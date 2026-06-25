from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from acta.agents.base import Agent, AgentContext, CandidateAction
from acta.models.reranker_dataset import rank_candidates as rank_feature_candidates


class InterventionVerifier(Protocol):
    def allow(
        self,
        context: AgentContext,
        raw_action: str,
        proposed_action: str,
        candidates: tuple[str, ...],
    ) -> tuple[bool, str]:
        ...


class LearnedRerankerAgent:
    """Apply a trained ReCAP reranker to a base agent's candidates."""

    def __init__(
        self,
        base_agent: Agent,
        model: Mapping[str, Any],
        abstain_margin: float = 0.0,
        safety_mode: str = "none",
        gate_model: Mapping[str, Any] | None = None,
        gate_threshold: float | None = None,
        intervention_verifier: InterventionVerifier | None = None,
    ) -> None:
        self.base_agent = base_agent
        self.model = model
        self.model_type = str(model.get("model_type", "recap_feature_reranker"))
        self.weights = _feature_weights(model)
        self.abstain_margin = abstain_margin
        self.safety_mode = safety_mode
        self.gate_model = gate_model
        self.gate_threshold = gate_threshold
        self.intervention_verifier = intervention_verifier

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        raw_candidates = tuple(self.base_agent.candidates(context))
        if len(raw_candidates) <= 1:
            return raw_candidates

        raw_actions = tuple(candidate.action for candidate in raw_candidates)
        preference_like = {
            "task_id": context.task_id,
            "seed": context.seed,
            "step_index": context.step_index,
            "history": tuple(context.history),
            "observation": context.observation,
            "initial_observation": context.initial_observation,
            "candidates": raw_actions,
            "rejected_action": raw_actions[0],
        }
        ranked = self._rank(preference_like)
        if not ranked:
            return raw_candidates

        scores = {action: score for action, score in ranked}
        learned_top = ranked[0][0]
        raw_top = raw_actions[0]
        margin = scores.get(learned_top, 0.0) - scores.get(raw_top, 0.0)

        # Verifier-free loop-break: act only when the base agent's top action is
        # provably stuck in a recent loop, and then promote the reranker's best
        # non-repeating candidate. This needs no learned gate or replay verifier,
        # and it confines interventions to steps where the raw action is almost
        # certainly wrong, so it lifts intervention coverage without the
        # gold-demotion harm that forces the probability gate to be conservative.
        if self.safety_mode == "loop-break":
            return self._loop_break_decision(
                context, raw_candidates, raw_actions, ranked, scores, margin
            )

        safety_reason = ""
        abstain = False
        if learned_top != raw_top and margin < self.abstain_margin:
            abstain = True
            safety_reason = "below_margin"
        elif learned_top != raw_top:
            safety_reason = self._safety_reason(context, raw_top, learned_top)
            abstain = bool(safety_reason)
        if not abstain and learned_top != raw_top and self.gate_model is not None:
            from acta.models.intervention_gate import intervention_probability

            prediction_like = {
                "task_id": context.task_id,
                "seed": context.seed,
                "step_index": context.step_index,
                "ranked_actions": [action for action, _score in ranked],
                "scores": scores,
                "margin_over_raw_top1": margin,
            }
            cutoff = (
                float(self.gate_model.get("threshold", 0.5))
                if self.gate_threshold is None
                else float(self.gate_threshold)
            )
            gate_probability = intervention_probability(
                preference_like,
                prediction_like,
                self.gate_model,
            )
            if gate_probability < cutoff:
                abstain = True
                safety_reason = "learned_intervention_gate"
        if not abstain and learned_top != raw_top and self.intervention_verifier is not None:
            allowed, verifier_reason = self.intervention_verifier.allow(
                context=context,
                raw_action=raw_top,
                proposed_action=learned_top,
                candidates=raw_actions,
            )
            if not allowed:
                abstain = True
                safety_reason = f"llm_verifier:{verifier_reason}"
        if abstain:
            return tuple(
                attach_reranker_metadata(
                    candidate,
                    score=scores.get(candidate.action),
                    margin=margin,
                    abstained=True,
                    raw_rank=index + 1,
                    intervened=False,
                    model_type=self.model_type,
                    safety_mode=self.safety_mode,
                    safety_reason=safety_reason,
                    gate_model_type=gate_model_type(self.gate_model),
                )
                for index, candidate in enumerate(raw_candidates)
            )

        by_action = {candidate.action: candidate for candidate in raw_candidates}
        return tuple(
            attach_reranker_metadata(
                by_action[action],
                score=score,
                margin=margin,
                abstained=False,
                raw_rank=raw_actions.index(action) + 1,
                intervened=action == learned_top and learned_top != raw_top,
                model_type=self.model_type,
                safety_mode=self.safety_mode,
                safety_reason=safety_reason,
                gate_model_type=gate_model_type(self.gate_model),
            )
            for action, score in ranked
            if action in by_action
        )

    def _loop_break_decision(
        self,
        context: AgentContext,
        raw_candidates: tuple[CandidateAction, ...],
        raw_actions: tuple[str, ...],
        ranked: tuple[tuple[str, float], ...],
        scores: Mapping[str, float],
        margin: float,
    ) -> tuple[CandidateAction, ...]:
        raw_top = raw_actions[0]
        by_action = {candidate.action: candidate for candidate in raw_candidates}

        def _abstain(reason: str) -> tuple[CandidateAction, ...]:
            return tuple(
                attach_reranker_metadata(
                    candidate,
                    score=scores.get(candidate.action),
                    margin=margin,
                    abstained=True,
                    raw_rank=index + 1,
                    intervened=False,
                    model_type=self.model_type,
                    safety_mode=self.safety_mode,
                    safety_reason=reason,
                    gate_model_type=gate_model_type(self.gate_model),
                )
                for index, candidate in enumerate(raw_candidates)
            )

        # Only intervene when the agent is provably stuck: the current state has
        # already been visited this episode (a true cycle / no-op loop). A
        # repeated action string alone is too loose -- it fires on benign repeats
        # in successful trajectories -- so we trigger on a state revisit, which
        # means the previous actions failed to make progress.
        revisited = context.seen_signatures.count(context.state_signature) >= 2
        if not revisited:
            return _abstain("state_not_revisited")

        # Promote the highest-ranked candidate that breaks the loop: it must
        # differ from the raw action and not itself be a recent repeat.
        chosen = next(
            (
                action
                for action, _score in ranked
                if action != raw_top
                and action in by_action
                and not action_is_recent_repeat(action, context.history)
            ),
            None,
        )
        if chosen is None:
            return _abstain("no_loop_breaking_candidate")

        chosen_candidate = attach_reranker_metadata(
            by_action[chosen],
            score=scores.get(chosen),
            margin=scores.get(chosen, 0.0) - scores.get(raw_top, 0.0),
            abstained=False,
            raw_rank=raw_actions.index(chosen) + 1,
            intervened=True,
            model_type=self.model_type,
            safety_mode=self.safety_mode,
            safety_reason="loop_break",
            gate_model_type=gate_model_type(self.gate_model),
        )
        rest = tuple(
            attach_reranker_metadata(
                by_action[action],
                score=score,
                margin=margin,
                abstained=False,
                raw_rank=raw_actions.index(action) + 1,
                intervened=False,
                model_type=self.model_type,
                safety_mode=self.safety_mode,
                safety_reason="loop_break",
                gate_model_type=gate_model_type(self.gate_model),
            )
            for action, score in ranked
            if action in by_action and action != chosen
        )
        return (chosen_candidate,) + rest

    def _rank(self, preference_like: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
        if self.model_type in {
            "recap_offline_policy_gradient_reranker",
            "recap_support_constrained_policy",
        }:
            from acta.models.policy_reranker import rank_candidates

            return rank_candidates(preference_like, self.model)
        if self.model_type.startswith("recap_sklearn_"):
            from acta.models.sklearn_reranker import rank_candidates

            return rank_candidates(preference_like, self.model)
        if self.model_type == "recap_cross_encoder_pointwise":
            from acta.models.cross_encoder_reranker import rank_candidates

            return rank_candidates(
                preference_like,
                self.model["model"],
                batch_size=int(self.model.get("batch_size", 16)),
                max_history=int(self.model.get("max_history", 8)),
                max_observation_chars=int(self.model.get("max_observation_chars", 0)),
            )
        return rank_feature_candidates(preference_like, self.weights)

    def _safety_reason(self, context: AgentContext, raw_top: str, learned_top: str) -> str:
        if self.safety_mode == "none":
            return ""
        if action_is_recent_repeat(learned_top, context.history):
            return "learned_top_recent_repeat"
        if self.safety_mode == "no-repeat":
            return ""
        if self.safety_mode in {"conservative", "loop-safe"}:
            if (
                self.safety_mode == "loop-safe"
                and context.history
                and raw_top == context.history[-1]
                and action_is_navigation(raw_top)
                and action_is_navigation(learned_top)
            ):
                return "protect_immediate_navigation_repeat"
            raw_suspicious = (
                action_is_recent_repeat(raw_top, context.history)
                or action_is_static(raw_top)
            )
            if raw_suspicious and not action_is_static(learned_top):
                return ""
            if action_is_manipulation(raw_top) and action_is_navigation(learned_top):
                return "would_demote_manipulation"
            return "raw_top_not_suspicious"
        raise ValueError(f"unknown reranker safety mode: {self.safety_mode}")


def attach_reranker_metadata(
    candidate: CandidateAction,
    score: float | None,
    margin: float,
    abstained: bool,
    raw_rank: int,
    intervened: bool = False,
    model_type: str = "recap_feature_reranker",
    safety_mode: str = "none",
    safety_reason: str = "",
    gate_model_type: str = "",
) -> CandidateAction:
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "recap_reranker_score": score,
            "recap_reranker_margin": margin,
            "recap_reranker_abstained": abstained,
            "recap_reranker_intervened": intervened,
            "recap_raw_rank": raw_rank,
            "recap_reranker_model_type": model_type,
            "recap_reranker_safety_mode": safety_mode,
            "recap_reranker_safety_reason": safety_reason,
            "recap_intervention_gate_model_type": gate_model_type,
        }
    )
    return CandidateAction(
        action=candidate.action,
        score=candidate.score if score is None else score,
        source=candidate.source,
        metadata=metadata,
    )


def load_reranker_model(path: Path, device: str | None = None) -> Mapping[str, Any]:
    if path.is_dir():
        from sentence_transformers import CrossEncoder

        summary_path = path / "recap_training_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "model_type": "recap_cross_encoder_pointwise",
            "model": CrossEncoder(str(path), device=device),
            "model_path": str(path),
            "max_history": int(summary.get("max_history", 8)),
            "max_observation_chars": int(summary.get("max_observation_chars", 0)),
        }
    if path.suffix == ".pt":
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".joblib":
        import joblib

        return joblib.load(path)
    return json.loads(path.read_text(encoding="utf-8"))


def gate_model_type(gate_model: Mapping[str, Any] | None) -> str:
    if gate_model is None:
        return ""
    return str(gate_model.get("model_type", "recap_intervention_gate"))


def _feature_weights(model: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in dict(model.get("weights", {})).items()
    }


def action_verb(action: str) -> str:
    return action.strip().split(maxsplit=1)[0].lower() if action.strip() else ""


def action_is_navigation(action: str) -> bool:
    return action_verb(action) in {"go", "north", "south", "east", "west"}


def action_is_static(action: str) -> bool:
    return action_verb(action) in {"look", "inventory", "examine"}


def action_is_manipulation(action: str) -> bool:
    return action_verb(action) in {
        "take",
        "drop",
        "open",
        "close",
        "unlock",
        "lock",
        "put",
        "insert",
        "eat",
    }


def action_is_recent_repeat(action: str, history: tuple[str, ...]) -> bool:
    return action in history[-4:]
