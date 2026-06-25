from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

from recap.agents import AgentContext, CandidateAction
from recap.envs.base import EnvAdapter
from recap.probe import PairProbe, ProbeConfig


@dataclass(frozen=True)
class ControllerConfig:
    """Soft penalties used for ReCAP candidate reranking."""

    equivalence_mode: str = "full"
    invalid_penalty: float = 100.0
    noop_penalty: float = 6.0
    seen_state_penalty: float = 4.0
    recent_repeat_penalty: float = 2.0
    absorbed_penalty: float = 3.0
    inverse_penalty: float = 4.0
    recent_window: int = 4
    enable_pair_penalties: bool = True


@dataclass(frozen=True)
class ControllerDecision:
    """Reranked candidates and diagnostics for one decision."""

    candidates: tuple[CandidateAction, ...]
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def selected(self) -> CandidateAction:
        if not self.candidates:
            raise ValueError("no candidates available")
        return self.candidates[0]


@dataclass(frozen=True)
class CachedReplay:
    valid: bool
    signature: Hashable | None


class PriorController:
    """Replay-backed soft controller for action candidates."""

    def __init__(
        self,
        adapter: EnvAdapter,
        env_name: str,
        config: ControllerConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.env_name = env_name
        self.config = config or ControllerConfig()
        self.probe = PairProbe(
            adapter=adapter,
            env_name=env_name,
            config=ProbeConfig(equivalence_mode=self.config.equivalence_mode),
        )
        self._replay_cache: dict[tuple[str, int, tuple[str, ...]], CachedReplay] = {}
        self._pair_cache: dict[
            tuple[str, int, tuple[str, ...], str, str],
            dict[str, bool | None],
        ] = {}
        self.cache_hits: dict[str, int] = {"replay": 0, "pair": 0}
        self.cache_misses: dict[str, int] = {"replay": 0, "pair": 0}

    def clear_cache(self) -> None:
        self._replay_cache.clear()
        self._pair_cache.clear()
        self.cache_hits = {"replay": 0, "pair": 0}
        self.cache_misses = {"replay": 0, "pair": 0}

    def rerank(
        self,
        context: AgentContext,
        candidates: Sequence[CandidateAction],
    ) -> ControllerDecision:
        scored: list[CandidateAction] = []
        reasons: dict[str, tuple[str, ...]] = {}
        for candidate in candidates:
            score, candidate_reasons = self._score_candidate(context, candidate)
            metadata = dict(candidate.metadata)
            metadata["recap_reasons"] = candidate_reasons
            scored.append(
                CandidateAction(
                    action=candidate.action,
                    score=score,
                    source=candidate.source,
                    metadata=metadata,
                )
            )
            reasons[candidate.action] = tuple(candidate_reasons)

        scored.sort(key=lambda item: (-item.score, item.action))
        return ControllerDecision(candidates=tuple(scored), reasons=reasons)

    def _score_candidate(
        self,
        context: AgentContext,
        candidate: CandidateAction,
    ) -> tuple[float, list[str]]:
        score = candidate.score
        reasons: list[str] = []
        action = candidate.action
        admissible = set(context.admissible_actions)

        if admissible and action not in admissible:
            return score - self.config.invalid_penalty, ["invalid"]

        next_signature = self._next_signature(context, action)
        if next_signature is not None:
            if self.adapter.is_equivalent(
                context.state_signature,
                next_signature,
                mode=self.config.equivalence_mode,
            ):
                score -= self.config.noop_penalty
                reasons.append("noop")
            if self._signature_seen(next_signature, context.seen_signatures):
                score -= self.config.seen_state_penalty
                reasons.append("seen_state")

        if action in context.history[-self.config.recent_window :]:
            score -= self.config.recent_repeat_penalty
            reasons.append("recent_repeat")

        if self.config.enable_pair_penalties and context.history:
            prefix = context.history[:-1]
            previous = context.history[-1]
            relations = self._pair_relations(
                context=context,
                prefix=prefix,
                action_a=previous,
                action_b=action,
            )
            if relations["absorb_ab"] is True:
                score -= self.config.absorbed_penalty
                reasons.append("absorbed_after_previous")
            if relations["inverse_ab"] is True:
                score -= self.config.inverse_penalty
                reasons.append("inverse_with_previous")

        return score, reasons

    def _next_signature(
        self,
        context: AgentContext,
        action: str,
    ) -> Hashable | None:
        cached = self._cached_replay(
            task_id=context.task_id,
            seed=context.seed,
            actions=context.history + (action,),
        )
        return cached.signature if cached.valid else None

    def _pair_relations(
        self,
        context: AgentContext,
        prefix: tuple[str, ...],
        action_a: str,
        action_b: str,
    ) -> dict[str, bool | None]:
        key = (context.task_id, context.seed, prefix, action_a, action_b)
        if key in self._pair_cache:
            self.cache_hits["pair"] += 1
            return self._pair_cache[key]

        self.cache_misses["pair"] += 1
        record = self.probe.probe_pair(
            task_id=context.task_id,
            seed=context.seed,
            prefix_actions=prefix,
            action_a=action_a,
            action_b=action_b,
        )
        relations = dict(record.relations)
        self._pair_cache[key] = relations
        return relations

    def _cached_replay(
        self,
        task_id: str,
        seed: int,
        actions: tuple[str, ...],
    ) -> CachedReplay:
        key = (task_id, seed, actions)
        if key in self._replay_cache:
            self.cache_hits["replay"] += 1
            return self._replay_cache[key]

        self.cache_misses["replay"] += 1
        replay = self.adapter.replay(task_id=task_id, prefix_actions=actions, seed=seed)
        signature = (
            self.adapter.signature(replay.state, mode=self.config.equivalence_mode)
            if replay.valid
            else None
        )
        cached = CachedReplay(valid=replay.valid, signature=signature)
        self._replay_cache[key] = cached
        return cached

    def _signature_seen(
        self,
        signature: Hashable,
        seen_signatures: tuple[Hashable, ...],
    ) -> bool:
        return any(
            self.adapter.is_equivalent(signature, seen, mode=self.config.equivalence_mode)
            for seen in seen_signatures
        )
