from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acta.agents import Agent, AgentContext, CandidateAction, NoisyCandidateAgent
from acta.agents.llm_agent import load_env_file
from acta.controllers import ActAController, ControllerConfig
from acta.envs.base import EnvAdapter
from acta.envs.textworld_adapter import TextWorldAdapter
from acta.eval.agent_loop import first_policy_command, rank_action
from acta.eval.eval_agent import build_agent, build_controller_config


@dataclass(frozen=True)
class GoldSelectorRecord:
    task_id: str
    seed: int
    step_index: int
    gold_action: str
    candidates_before: tuple[str, ...]
    candidates_after: tuple[str, ...]
    base_action: str | None
    acta_action: str | None
    gold_rank_before: int | None
    gold_rank_after: int | None
    base_selected_gold: bool
    acta_selected_gold: bool
    acta_recovered_gold: bool
    acta_demoted_gold: bool
    top1_action: str | None
    top1_reasons: tuple[str, ...]
    top1_structural_bad: bool
    acta_blocked_bad_top1: bool
    selected_reasons: tuple[str, ...]
    valid_gold_step: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline top-1 vs ActA reranking on oracle TextWorld states."
    )
    parser.add_argument("games", type=Path, nargs="+")
    parser.add_argument("--agent", choices=["mock-llm", "openai"], default="mock-llm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument(
        "--candidate-noise",
        choices=["none", "frontload-existing-structural", "frontload-structural"],
        default="none",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm-max-retries", type=int, default=4)
    parser.add_argument("--llm-retry-base-delay", type=float, default=2.0)
    parser.add_argument("--llm-call-delay", type=float, default=0.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--fast-controller", action="store_true")
    parser.add_argument("--noop-penalty", type=float, default=6.0)
    parser.add_argument("--seen-state-penalty", type=float, default=4.0)
    parser.add_argument("--recent-repeat-penalty", type=float, default=2.0)
    parser.add_argument("--absorbed-penalty", type=float, default=3.0)
    parser.add_argument("--inverse-penalty", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("analysis/gold_selector_eval.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    adapter = TextWorldAdapter()
    controller = ActAController(
        adapter=adapter,
        env_name="textworld",
        config=build_controller_config(args),
    )

    records: list[GoldSelectorRecord] = []
    for index, game in enumerate(args.games):
        agent = build_agent(args, index)
        game_records = evaluate_game(
            adapter=adapter,
            agent=agent,
            controller=controller,
            task_id=str(game),
            seed=args.seed,
            max_steps=args.max_steps,
        )
        records.extend(game_records)
        if args.progress:
            selected = sum(record.acta_selected_gold for record in game_records)
            print(
                f"game={index + 1}/{len(args.games)} file={game} "
                f"states={len(game_records)} acta_selected_gold={selected}",
                flush=True,
            )

    summary = summarize(records)
    summary["agent"] = args.agent
    if args.agent == "openai":
        summary["model"] = args.model or os.environ.get("ACTA_LLM_MODEL")
    summary["controller"] = "acta"
    summary["controller_cache_hits"] = dict(controller.cache_hits)
    summary["controller_cache_misses"] = dict(controller.cache_misses)

    payload = {"summary": summary, "records": records}
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={args.out}")


def evaluate_game(
    adapter: EnvAdapter,
    agent: Agent,
    controller: ActAController,
    task_id: str,
    seed: int,
    max_steps: int,
) -> list[GoldSelectorRecord]:
    reset = adapter.reset(task_id=task_id, seed=seed)
    state = reset.state
    observation = reset.observation
    initial_observation = reset.observation
    history: list[str] = []
    seen_signatures = [adapter.signature(state, mode="full")]
    records: list[GoldSelectorRecord] = []

    for step_index in range(max_steps):
        gold_action = first_policy_command(adapter.policy_commands(state))
        if gold_action is None:
            break

        state_signature = adapter.signature(state, mode="full")
        context = AgentContext(
            task_id=task_id,
            seed=seed,
            step_index=step_index,
            observation=observation,
            admissible_actions=tuple(adapter.admissible_actions(state)),
            history=tuple(history),
            state_signature=state_signature,
            seen_signatures=tuple(seen_signatures),
            initial_observation=initial_observation,
        )
        candidates = tuple(agent.candidates(context))
        if not candidates:
            candidates = tuple(CandidateAction(action=action) for action in context.admissible_actions)
        candidates_before = tuple(candidate.action for candidate in candidates)
        base_action = candidates_before[0] if candidates_before else None

        decision = controller.rerank(context, candidates)
        candidates_after = tuple(candidate.action for candidate in decision.candidates)
        acta_action = decision.selected.action if decision.candidates else None
        gold_rank_before = rank_action(candidates_before, gold_action)
        gold_rank_after = rank_action(candidates_after, gold_action)
        top1_reasons = decision.reasons.get(base_action, ()) if base_action is not None else ()
        selected_reasons = decision.reasons.get(acta_action, ()) if acta_action is not None else ()
        base_selected_gold = base_action == gold_action
        acta_selected_gold = acta_action == gold_action

        restored = adapter.replay(task_id=task_id, prefix_actions=tuple(history), seed=seed)
        state = restored.state
        observation = restored.observation
        step = adapter.step(gold_action)
        history.append(gold_action)
        state = step.state
        observation = step.observation
        seen_signatures.append(adapter.signature(state, mode="full"))

        records.append(
            GoldSelectorRecord(
                task_id=task_id,
                seed=seed,
                step_index=step_index,
                gold_action=gold_action,
                candidates_before=candidates_before,
                candidates_after=candidates_after,
                base_action=base_action,
                acta_action=acta_action,
                gold_rank_before=gold_rank_before,
                gold_rank_after=gold_rank_after,
                base_selected_gold=base_selected_gold,
                acta_selected_gold=acta_selected_gold,
                acta_recovered_gold=(
                    gold_rank_before is not None
                    and gold_rank_before != 1
                    and acta_selected_gold
                ),
                acta_demoted_gold=gold_rank_before == 1 and not acta_selected_gold,
                top1_action=base_action,
                top1_reasons=tuple(top1_reasons),
                top1_structural_bad=bool(top1_reasons),
                acta_blocked_bad_top1=(
                    bool(top1_reasons)
                    and base_action is not None
                    and acta_action != base_action
                ),
                selected_reasons=tuple(selected_reasons),
                valid_gold_step=step.valid,
            )
        )
        if step.done:
            break

    return records


def summarize(records: list[GoldSelectorRecord]) -> dict[str, Any]:
    total = len(records)
    gold_in_topk = [record for record in records if record.gold_rank_before is not None]
    base_selected = [record for record in records if record.base_selected_gold]
    acta_selected = [record for record in records if record.acta_selected_gold]
    recovery_opps = [
        record
        for record in records
        if record.gold_rank_before is not None and record.gold_rank_before != 1
    ]
    recovered = [record for record in records if record.acta_recovered_gold]
    demotion_opps = [record for record in records if record.gold_rank_before == 1]
    demoted = [record for record in records if record.acta_demoted_gold]
    top1_bad = [record for record in records if record.top1_structural_bad]
    blocked_bad = [record for record in records if record.acta_blocked_bad_top1]
    rank_before_values = [
        record.gold_rank_before for record in records if record.gold_rank_before is not None
    ]
    rank_after_values = [
        record.gold_rank_after for record in records if record.gold_rank_after is not None
    ]
    return {
        "states": total,
        "gold_in_topk": len(gold_in_topk),
        "gold_in_topk_rate": rate(len(gold_in_topk), total),
        "base_selected_gold": len(base_selected),
        "base_selected_gold_rate": rate(len(base_selected), total),
        "acta_selected_gold": len(acta_selected),
        "acta_selected_gold_rate": rate(len(acta_selected), total),
        "recovery_opportunities": len(recovery_opps),
        "acta_recovered_gold": len(recovered),
        "acta_recovery_rate": rate(len(recovered), len(recovery_opps)),
        "demotion_opportunities": len(demotion_opps),
        "acta_demoted_gold": len(demoted),
        "acta_demotion_rate": rate(len(demoted), len(demotion_opps)),
        "top1_structural_bad": len(top1_bad),
        "top1_structural_bad_rate": rate(len(top1_bad), total),
        "acta_blocked_bad_top1": len(blocked_bad),
        "acta_blocked_bad_top1_rate": rate(len(blocked_bad), len(top1_bad)),
        "avg_gold_rank_before": average(rank_before_values),
        "avg_gold_rank_after": average(rank_after_values),
        "valid_gold_step_rate": rate(sum(record.valid_gold_step for record in records), total),
    }


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_jsonable(getattr(value, key))
            for key in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
