from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ReCAP failure decomposition from a compilation ledger."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("analysis/recap_failure_decomposition.json"))
    args = parser.parse_args()

    ledger = tuple(read_jsonl(args.ledger))
    preferences = tuple(read_jsonl(args.preferences)) if args.preferences is not None else ()
    step_ledger = normalize_step_ledger(ledger)
    output = {
        "summary": summarize_decomposition(step_ledger, preferences),
        "by_trajectory": summarize_by_trajectory(step_ledger),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def summarize_decomposition(
    ledger: tuple[Mapping[str, Any], ...],
    preferences: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    step_ledger = normalize_step_ledger(ledger)
    failed = tuple(record for record in step_ledger if record.get("original_outcome") == "fail")
    target = failed if failed else step_ledger
    trajectory_count = len({str(record.get("trajectory_id", "")) for record in step_ledger})
    failed_trajectory_count = len({str(record.get("trajectory_id", "")) for record in failed})
    summarized_trajectory_count = len({str(record.get("trajectory_id", "")) for record in target})
    candidate_steps = sum(1 for record in target if has_candidates(record))
    certified_preferences = count_status(target, "certified_preference")
    candidate_absent = count_status(target, "repair_not_in_candidates")
    repair_same = count_status(target, "repair_same_as_executed")
    no_repair = count_status(target, "no_repair_found")
    suffix_too_long = count_status(target, "suffix_too_long")
    invalid_replay = count_status(target, "invalid_replay")
    no_candidates = count_status(target, "no_candidates_logged")
    preferred_ranks = [
        int(record["preferred_rank_before"])
        for record in preferences
        if record.get("preferred_rank_before") is not None
    ]
    if not preferred_ranks:
        preferred_ranks = [
            int(record["repair_rank_before"])
            for record in target
            if record.get("status") == "certified_preference"
            and record.get("repair_rank_before") is not None
        ]
    return {
        "trajectories": trajectory_count,
        "failed_trajectories": failed_trajectory_count,
        "summarized_trajectories": summarized_trajectory_count,
        "steps": len(target),
        "candidate_steps": candidate_steps,
        "denominators": {
            "trajectory_level": summarized_trajectory_count,
            "candidate_step_level": candidate_steps,
            "certified_preference_level": certified_preferences,
        },
        "certified_preferences": certified_preferences,
        "candidate_absent": candidate_absent,
        "no_repair_found": no_repair,
        "repair_same_as_executed": repair_same,
        "suffix_too_long": suffix_too_long,
        "invalid_replay": invalid_replay,
        "no_candidates_logged": no_candidates,
        "certified_misranking_step_rate": safe_div(
            certified_preferences,
            candidate_steps,
        ),
        "candidate_absent_step_rate": safe_div(candidate_absent, candidate_steps),
        "repair_same_as_executed_step_rate": safe_div(repair_same, candidate_steps),
        "no_repair_found_step_rate": safe_div(no_repair, candidate_steps),
        "suffix_too_long_step_rate": safe_div(suffix_too_long, candidate_steps),
        "invalid_replay_step_rate": safe_div(invalid_replay, candidate_steps),
        "no_candidates_logged_step_rate": safe_div(no_candidates, len(target)),
        "avg_preferred_rank_on_certified_preferences": (
            sum(preferred_ranks) / len(preferred_ranks) if preferred_ranks else 0.0
        ),
        "candidate_misranking_rate": safe_div(certified_preferences, candidate_steps),
        "candidate_absent_rate": safe_div(candidate_absent, candidate_steps),
        "avg_preferred_rank": (
            sum(preferred_ranks) / len(preferred_ranks) if preferred_ranks else 0.0
        ),
    }


def normalize_step_ledger(
    ledger: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    if not ledger:
        return ()
    if any("step_statuses" in record for record in ledger):
        rows: list[Mapping[str, Any]] = []
        for record in ledger:
            for step in record.get("step_statuses", ()):
                row = dict(step)
                row.setdefault("trajectory_id", record.get("trajectory_id"))
                row.setdefault("task_id", record.get("task_id"))
                row.setdefault("seed", record.get("seed"))
                row.setdefault("original_outcome", record.get("outcome"))
                row.setdefault("candidate_count", step.get("candidate_count", 0))
                rows.append(row)
        return tuple(rows)
    return ledger


def summarize_by_trajectory(
    step_ledger: tuple[Mapping[str, Any], ...],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in step_ledger:
        grouped.setdefault(str(record.get("trajectory_id", "")), []).append(record)
    summaries: list[dict[str, Any]] = []
    for trajectory_id, records in sorted(grouped.items()):
        candidate_steps = sum(1 for record in records if has_candidates(record))
        summaries.append(
            {
                "trajectory_id": trajectory_id,
                "task_id": records[0].get("task_id") if records else None,
                "seed": records[0].get("seed") if records else None,
                "outcome": records[0].get("original_outcome") if records else None,
                "num_steps": len(records),
                "num_candidate_steps": candidate_steps,
                "num_certified_preferences": count_status(records, "certified_preference"),
                "num_candidate_absent": count_status(records, "repair_not_in_candidates"),
                "num_no_repair_found": count_status(records, "no_repair_found"),
                "num_repair_same_as_executed": count_status(records, "repair_same_as_executed"),
                "num_suffix_too_long": count_status(records, "suffix_too_long"),
                "num_invalid_replay": count_status(records, "invalid_replay"),
                "num_no_candidates_logged": count_status(records, "no_candidates_logged"),
            }
        )
    return summaries


def count_status(records: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]], status: str) -> int:
    return sum(1 for record in records if record.get("status") == status)


def has_candidates(record: Mapping[str, Any]) -> bool:
    if record.get("has_logged_candidates") is not None:
        return bool(record.get("has_logged_candidates"))
    return int(record.get("candidate_count", 0) or 0) > 0


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
