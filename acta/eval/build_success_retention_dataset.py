from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build success-step retention records for reranker gold-demotion evaluation."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    trajectories = tuple(read_jsonl(args.trajectories))
    records, summary = build_retention_dataset(
        trajectories=trajectories,
        include_failed=args.include_failed,
        max_records=args.max_records,
    )
    write_jsonl(args.out, records)
    summary["outputs"] = {"records": str(args.out)}
    summary_out = args.summary_out or args.out.with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")
    print(f"summary={summary_out}")


def build_retention_dataset(
    trajectories: tuple[Mapping[str, Any], ...],
    include_failed: bool = False,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = {
        "failed_trajectory": 0,
        "no_gold_action": 0,
        "no_candidates": 0,
        "gold_not_raw_top1": 0,
        "single_candidate": 0,
        "selected_not_gold": 0,
    }
    considered_steps = 0
    success_trajectories = 0
    failed_trajectories = 0

    for trajectory in trajectories:
        success = bool(trajectory.get("success", False))
        if success:
            success_trajectories += 1
        else:
            failed_trajectories += 1
        if not success and not include_failed:
            skipped["failed_trajectory"] += len(tuple(trajectory.get("steps", ())))
            continue

        history: list[str] = []
        for step in tuple(trajectory.get("steps", ())):
            considered_steps += 1
            candidate_actions = tuple(
                str(action)
                for action in step.get("candidates_before", step.get("candidates", ()))
            )
            gold_action = step.get("gold_action")
            if gold_action is None:
                skipped["no_gold_action"] += 1
                history.append(str(step.get("action", "")))
                continue
            gold = str(gold_action)
            if not candidate_actions:
                skipped["no_candidates"] += 1
                history.append(str(step.get("action", "")))
                continue
            if candidate_actions[0] != gold:
                skipped["gold_not_raw_top1"] += 1
                history.append(str(step.get("action", "")))
                continue
            if len(candidate_actions) < 2:
                skipped["single_candidate"] += 1
                history.append(str(step.get("action", "")))
                continue
            if str(step.get("action", "")) != gold:
                skipped["selected_not_gold"] += 1
                history.append(str(step.get("action", "")))
                continue

            rejected = first_non_gold(candidate_actions, gold)
            if rejected is None:
                skipped["single_candidate"] += 1
                history.append(str(step.get("action", "")))
                continue

            records.append(
                {
                    "task_id": str(trajectory["task_id"]),
                    "trajectory_id": str(trajectory.get("trajectory_id", "")),
                    "env": str(trajectory.get("env", "")),
                    "difficulty": str(trajectory.get("difficulty", "")),
                    "game_id": str(trajectory.get("game_id", "")),
                    "game_seed": trajectory.get("game_seed"),
                    "seed": int(trajectory.get("seed", trajectory.get("rollout_seed", 0))),
                    "rollout_seed": int(trajectory.get("rollout_seed", trajectory.get("seed", 0))),
                    "step_index": int(step.get("step_index", len(history))),
                    "history": tuple(history),
                    "candidates": candidate_actions,
                    "candidate_count": len(candidate_actions),
                    "preferred_action": gold,
                    "preferred_action_norm": normalize_action(gold),
                    "rejected_action": rejected,
                    "rejected_action_norm": normalize_action(rejected),
                    "preferred_rank_before": 1,
                    "rejected_rank_before": candidate_actions.index(rejected) + 1,
                    "source": "success_policy_retention",
                    "certificate_level": "K1_success_policy_gold",
                    "original_outcome": "success",
                    "retention_target": True,
                }
            )
            history.append(str(step.get("action", "")))
            if max_records is not None and len(records) >= max_records:
                return records, retention_summary(
                    trajectories=trajectories,
                    success_trajectories=success_trajectories,
                    failed_trajectories=failed_trajectories,
                    considered_steps=considered_steps,
                    records=records,
                    skipped=skipped,
                    include_failed=include_failed,
                    max_records=max_records,
                )

    return records, retention_summary(
        trajectories=trajectories,
        success_trajectories=success_trajectories,
        failed_trajectories=failed_trajectories,
        considered_steps=considered_steps,
        records=records,
        skipped=skipped,
        include_failed=include_failed,
        max_records=max_records,
    )


def retention_summary(
    trajectories: tuple[Mapping[str, Any], ...],
    success_trajectories: int,
    failed_trajectories: int,
    considered_steps: int,
    records: list[Mapping[str, Any]],
    skipped: Mapping[str, int],
    include_failed: bool,
    max_records: int | None,
) -> dict[str, Any]:
    return {
        "trajectories": len(trajectories),
        "success_trajectories": success_trajectories,
        "failed_trajectories": failed_trajectories,
        "include_failed": include_failed,
        "max_records": max_records,
        "considered_steps": considered_steps,
        "retention_records": len(records),
        "skipped": dict(skipped),
    }


def first_non_gold(candidates: tuple[str, ...], gold: str) -> str | None:
    for action in candidates:
        if action != gold:
            return action
    return None


def normalize_action(action: str) -> str:
    return " ".join(action.strip().lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
