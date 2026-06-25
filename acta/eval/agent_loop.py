from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping

from acta.agents import Agent, AgentContext, CandidateAction
from acta.controllers import ActAController, ControllerDecision
from acta.envs.base import EnvAdapter


@dataclass(frozen=True)
class AgentStepLog:
    step_index: int
    action: str
    valid: bool
    reward: float
    done: bool
    selected_score: float
    controller_reasons: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    candidates_before: tuple[str, ...] = ()
    gold_action: str | None = None
    gold_in_candidates: bool = False
    gold_rank_before: int | None = None
    gold_rank_after: int | None = None
    selected_is_gold: bool = False
    acta_recovered_gold: bool = False
    acta_demoted_gold: bool = False
    top1_action: str | None = None
    top1_structural_reasons: tuple[str, ...] = ()
    top1_structural_bad: bool = False
    acta_blocked_bad_top1: bool = False
    reranker_used: bool = False
    reranker_abstained: bool = False
    reranker_intervened: bool = False
    reranker_margin: float | None = None
    reranker_safety_reason: str = ""


@dataclass(frozen=True)
class AgentEpisodeResult:
    task_id: str
    seed: int
    success: bool
    total_reward: float
    steps: tuple[AgentStepLog, ...]
    final_signature: Hashable
    metrics: Mapping[str, float] = field(default_factory=dict)
    error: str | None = None


def run_episode(
    adapter: EnvAdapter,
    agent: Agent,
    task_id: str,
    seed: int = 0,
    max_steps: int = 30,
    controller: ActAController | None = None,
    equivalence_mode: str = "full",
) -> AgentEpisodeResult:
    reset = adapter.reset(task_id=task_id, seed=seed)
    state = reset.state
    observation = reset.observation
    initial_observation = reset.observation
    history: list[str] = []
    seen_signatures: list[Hashable] = [adapter.signature(state, mode=equivalence_mode)]
    logs: list[AgentStepLog] = []
    total_reward = 0.0
    done = reset.done

    for step_index in range(max_steps):
        if done:
            break

        state_signature = adapter.signature(state, mode=equivalence_mode)
        admissible = tuple(adapter.admissible_actions(state))
        context = AgentContext(
            task_id=task_id,
            seed=seed,
            step_index=step_index,
            observation=observation,
            admissible_actions=admissible,
            history=tuple(history),
            state_signature=state_signature,
            seen_signatures=tuple(seen_signatures),
            initial_observation=initial_observation,
        )
        candidates = tuple(agent.candidates(context))
        if not candidates and admissible:
            candidates = tuple(CandidateAction(action=action) for action in admissible)
        if not candidates:
            break

        candidate_actions_before = candidate_actions_in_raw_order(candidates)
        raw_top_action = candidate_actions_before[0] if candidate_actions_before else None
        gold_action = first_policy_command(adapter.policy_commands(state))
        gold_rank_before = rank_action(candidate_actions_before, gold_action)

        decision: ControllerDecision | None = None
        if controller is not None:
            decision = controller.rerank(context, candidates)
            selected = decision.selected
            reasons = decision.reasons.get(selected.action, ())
            candidate_actions = tuple(candidate.action for candidate in decision.candidates)
            top1_reasons = decision.reasons.get(raw_top_action, ()) if raw_top_action else ()
        else:
            selected = candidates[0]
            reasons = ()
            candidate_actions = tuple(candidate.action for candidate in candidates)
            top1_reasons = structural_reasons_for_action(
                adapter=adapter,
                context=context,
                action=raw_top_action or selected.action,
                equivalence_mode=equivalence_mode,
            )

        gold_rank_after = rank_action(candidate_actions, gold_action)
        selected_is_gold = gold_action is not None and selected.action == gold_action
        top1_action = raw_top_action
        top1_structural_bad = bool(top1_reasons)
        selected_metadata = dict(selected.metadata)
        reranker_used = "recap_reranker_score" in selected_metadata
        reranker_abstained = bool(selected_metadata.get("recap_reranker_abstained", False))
        reranker_intervened = bool(selected_metadata.get("recap_reranker_intervened", False))
        reranker_margin = selected_metadata.get("recap_reranker_margin")
        reranker_margin_float = (
            float(reranker_margin)
            if isinstance(reranker_margin, int | float)
            else None
        )
        reranker_safety_reason = str(
            selected_metadata.get("recap_reranker_safety_reason", "")
        )

        restored = adapter.replay(task_id=task_id, prefix_actions=tuple(history), seed=seed)
        state = restored.state
        observation = restored.observation
        step = adapter.step(selected.action)
        total_reward += step.reward
        history.append(selected.action)
        state = step.state
        observation = step.observation
        done = step.done
        signature = adapter.signature(state, mode=equivalence_mode)
        seen_signatures.append(signature)
        logs.append(
            AgentStepLog(
                step_index=step_index,
                action=selected.action,
                valid=step.valid,
                reward=step.reward,
                done=step.done,
                selected_score=selected.score,
                controller_reasons=tuple(reasons),
                candidates=candidate_actions,
                candidates_before=candidate_actions_before,
                gold_action=gold_action,
                gold_in_candidates=gold_rank_before is not None,
                gold_rank_before=gold_rank_before,
                gold_rank_after=gold_rank_after,
                selected_is_gold=selected_is_gold,
                acta_recovered_gold=(
                    decision is not None
                    and gold_rank_before is not None
                    and gold_rank_before != 1
                    and selected_is_gold
                ),
                acta_demoted_gold=(
                    decision is not None
                    and gold_rank_before == 1
                    and not selected_is_gold
                ),
                top1_action=top1_action,
                top1_structural_reasons=tuple(top1_reasons),
                top1_structural_bad=top1_structural_bad,
                acta_blocked_bad_top1=(
                    decision is not None
                    and top1_structural_bad
                    and top1_action is not None
                    and selected.action != top1_action
                ),
                reranker_used=reranker_used,
                reranker_abstained=reranker_abstained,
                reranker_intervened=reranker_intervened,
                reranker_margin=reranker_margin_float,
                reranker_safety_reason=reranker_safety_reason,
            )
        )

    final_signature = adapter.signature(state, mode=equivalence_mode)
    return AgentEpisodeResult(
        task_id=task_id,
        seed=seed,
        success=done,
        total_reward=total_reward,
        steps=tuple(logs),
        final_signature=final_signature,
        metrics=episode_metrics(tuple(logs)),
    )


def episode_metrics(steps: tuple[AgentStepLog, ...]) -> dict[str, float]:
    total = len(steps)
    invalid = sum(1 for step in steps if not step.valid)
    repeated = sum(
        1
        for index, step in enumerate(steps)
        if step.action in {previous.action for previous in steps[max(0, index - 4) : index]}
    )
    noop_penalized = sum(1 for step in steps if "noop" in step.controller_reasons)
    seen_penalized = sum(1 for step in steps if "seen_state" in step.controller_reasons)
    inverse_penalized = sum(1 for step in steps if "inverse_with_previous" in step.controller_reasons)
    absorbed_penalized = sum(1 for step in steps if "absorbed_after_previous" in step.controller_reasons)
    gold_steps = sum(1 for step in steps if step.gold_action is not None)
    gold_rank_before_values = [
        step.gold_rank_before for step in steps if step.gold_rank_before is not None
    ]
    gold_rank_after_values = [
        step.gold_rank_after for step in steps if step.gold_rank_after is not None
    ]
    top1_bad = sum(1 for step in steps if step.top1_structural_bad)
    blocked_bad_top1 = sum(1 for step in steps if step.acta_blocked_bad_top1)
    reranker_steps = sum(1 for step in steps if step.reranker_used)
    reranker_interventions = sum(1 for step in steps if step.reranker_intervened)
    reranker_abstentions = sum(1 for step in steps if step.reranker_abstained)
    reranker_recovered_gold = sum(
        1
        for step in steps
        if step.reranker_intervened
        and step.gold_rank_before is not None
        and step.gold_rank_before != 1
        and step.selected_is_gold
    )
    reranker_demoted_gold = sum(
        1
        for step in steps
        if step.reranker_intervened
        and step.gold_rank_before == 1
        and not step.selected_is_gold
    )
    reranker_margins = [
        float(step.reranker_margin)
        for step in steps
        if step.reranker_margin is not None
    ]
    return {
        "steps": float(total),
        "invalid_actions": float(invalid),
        "invalid_action_rate": invalid / total if total else 0.0,
        "repeated_actions": float(repeated),
        "repeated_action_rate": repeated / total if total else 0.0,
        "noop_penalized": float(noop_penalized),
        "seen_state_penalized": float(seen_penalized),
        "inverse_penalized": float(inverse_penalized),
        "absorbed_penalized": float(absorbed_penalized),
        "gold_steps": float(gold_steps),
        "gold_in_candidates": float(sum(1 for step in steps if step.gold_in_candidates)),
        "gold_in_candidate_rate": (
            sum(1 for step in steps if step.gold_in_candidates) / gold_steps
            if gold_steps
            else 0.0
        ),
        "selected_gold": float(sum(1 for step in steps if step.selected_is_gold)),
        "selected_gold_rate": (
            sum(1 for step in steps if step.selected_is_gold) / gold_steps
            if gold_steps
            else 0.0
        ),
        "acta_recovered_gold": float(sum(1 for step in steps if step.acta_recovered_gold)),
        "acta_demoted_gold": float(sum(1 for step in steps if step.acta_demoted_gold)),
        "avg_gold_rank_before": (
            sum(gold_rank_before_values) / len(gold_rank_before_values)
            if gold_rank_before_values
            else 0.0
        ),
        "avg_gold_rank_after": (
            sum(gold_rank_after_values) / len(gold_rank_after_values)
            if gold_rank_after_values
            else 0.0
        ),
        "top1_structural_bad": float(top1_bad),
        "top1_structural_bad_rate": top1_bad / total if total else 0.0,
        "acta_blocked_bad_top1": float(blocked_bad_top1),
        "acta_blocked_bad_top1_rate": blocked_bad_top1 / top1_bad if top1_bad else 0.0,
        "reranker_steps": float(reranker_steps),
        "reranker_interventions": float(reranker_interventions),
        "reranker_intervention_rate": (
            reranker_interventions / reranker_steps if reranker_steps else 0.0
        ),
        "reranker_abstentions": float(reranker_abstentions),
        "reranker_abstain_rate": (
            reranker_abstentions / reranker_steps if reranker_steps else 0.0
        ),
        "reranker_recovered_gold": float(reranker_recovered_gold),
        "reranker_demoted_gold": float(reranker_demoted_gold),
        "reranker_gold_demotion_rate": (
            reranker_demoted_gold / gold_steps if gold_steps else 0.0
        ),
        "avg_reranker_margin": (
            sum(reranker_margins) / len(reranker_margins)
            if reranker_margins
            else 0.0
        ),
    }


def first_policy_command(policy_commands: tuple[str, ...] | list[str] | object) -> str | None:
    if not isinstance(policy_commands, tuple | list) or not policy_commands:
        return None
    return str(policy_commands[0])


def rank_action(actions: tuple[str, ...], target: str | None) -> int | None:
    if target is None:
        return None
    try:
        return actions.index(target) + 1
    except ValueError:
        return None


def candidate_actions_in_raw_order(candidates: tuple[CandidateAction, ...]) -> tuple[str, ...]:
    if not candidates:
        return ()
    if not any("recap_raw_rank" in candidate.metadata for candidate in candidates):
        return tuple(candidate.action for candidate in candidates)
    indexed: list[tuple[int, int, str]] = []
    missing_offset = len(candidates) + 1
    for index, candidate in enumerate(candidates):
        rank = candidate.metadata.get("recap_raw_rank")
        rank_value = int(rank) if isinstance(rank, int | float) else missing_offset + index
        indexed.append((rank_value, index, candidate.action))
    indexed.sort()
    return tuple(action for _rank, _index, action in indexed)


def structural_reasons_for_action(
    adapter: EnvAdapter,
    context: AgentContext,
    action: str,
    equivalence_mode: str,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if action in context.history[-4:]:
        reasons.append("recent_repeat")

    replay = adapter.replay(
        task_id=context.task_id,
        prefix_actions=context.history + (action,),
        seed=context.seed,
    )
    if not replay.valid:
        return tuple(reasons + ["invalid"])

    next_signature = adapter.signature(replay.state, mode=equivalence_mode)
    if adapter.is_equivalent(context.state_signature, next_signature, mode=equivalence_mode):
        reasons.append("noop")
    if any(
        adapter.is_equivalent(next_signature, seen, mode=equivalence_mode)
        for seen in context.seen_signatures
    ):
        reasons.append("seen_state")
    return tuple(reasons)
