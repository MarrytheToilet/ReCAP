from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from recap.eval.eval_recap_failure_decomposition import has_candidates, normalize_step_ledger, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ReCAP sensitivity to candidate-list width and suffix budget."
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=(2, 3, 4, 5))
    parser.add_argument("--suffix-budget", type=int, nargs="+", default=(3, 5, 7))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ledger = tuple(read_jsonl(args.ledger))
    output = evaluate_sensitivity(
        ledger=ledger,
        top_k_values=tuple(args.top_k),
        suffix_budgets=tuple(args.suffix_budget),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def evaluate_sensitivity(
    ledger: Sequence[Mapping[str, Any]],
    top_k_values: Sequence[int] = (2, 3, 4, 5),
    suffix_budgets: Sequence[int] = (3, 5, 7),
) -> dict[str, Any]:
    rows = tuple(normalize_step_ledger(tuple(ledger)))
    failed = tuple(row for row in rows if row.get("original_outcome") == "fail")
    target = failed if failed else rows
    candidate_steps = sum(1 for row in target if has_candidates(row))
    certified = tuple(row for row in target if row.get("status") == "certified_preference")
    base_candidate_absent = count_status(target, "repair_not_in_candidates")
    base_suffix_too_long = count_status(target, "suffix_too_long")

    top_k_rows = [
        top_k_summary(
            certified=certified,
            candidate_steps=candidate_steps,
            base_candidate_absent=base_candidate_absent,
            k=k,
        )
        for k in top_k_values
    ]
    suffix_rows = [
        suffix_budget_summary(
            certified=certified,
            candidate_steps=candidate_steps,
            base_suffix_too_long=base_suffix_too_long,
            budget=budget,
        )
        for budget in suffix_budgets
    ]
    suffix_rows.append(
        suffix_budget_summary(
            certified=certified,
            candidate_steps=candidate_steps,
            base_suffix_too_long=base_suffix_too_long,
            budget=None,
        )
    )
    return {
        "summary": {
            "failed_candidate_steps": candidate_steps,
            "certified_preferences": len(certified),
            "base_candidate_absent": base_candidate_absent,
            "base_suffix_too_long": base_suffix_too_long,
        },
        "top_k_truncation": top_k_rows,
        "suffix_budget": suffix_rows,
    }


def top_k_summary(
    certified: Sequence[Mapping[str, Any]],
    candidate_steps: int,
    base_candidate_absent: int,
    k: int,
) -> dict[str, Any]:
    retained = tuple(
        row for row in certified if rank_or_large(row.get("repair_rank_before")) <= k
    )
    downgraded_absent = len(certified) - len(retained)
    return {
        "top_k": k,
        "certified_preferences_retained": len(retained),
        "certified_preferences_lost_vs_top5": downgraded_absent,
        "certified_misranking_step_rate": safe_div(len(retained), candidate_steps),
        "candidate_absent_step_rate": safe_div(
            base_candidate_absent + downgraded_absent,
            candidate_steps,
        ),
        "recappable_trajectories": len({trajectory_id(row) for row in retained}),
        "avg_preferred_rank": mean(
            [rank_or_large(row.get("repair_rank_before")) for row in retained]
        ),
    }


def suffix_budget_summary(
    certified: Sequence[Mapping[str, Any]],
    candidate_steps: int,
    base_suffix_too_long: int,
    budget: int | None,
) -> dict[str, Any]:
    retained = tuple(
        row
        for row in certified
        if budget is None or length_or_large(row.get("repair_suffix_len")) <= budget
    )
    newly_too_long = len(certified) - len(retained)
    return {
        "suffix_budget": "full" if budget is None else budget,
        "certified_preferences_retained": len(retained),
        "certified_preferences_lost_vs_full": newly_too_long,
        "certified_misranking_step_rate": safe_div(len(retained), candidate_steps),
        "suffix_too_long_step_rate": safe_div(
            base_suffix_too_long + newly_too_long,
            candidate_steps,
        ),
        "recappable_trajectories": len({trajectory_id(row) for row in retained}),
    }


def count_status(rows: Sequence[Mapping[str, Any]], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def rank_or_large(value: object) -> int:
    if value is None:
        return 10**9
    return int(value)


def length_or_large(value: object) -> int:
    if value is None:
        return 10**9
    return int(value)


def trajectory_id(row: Mapping[str, Any]) -> str:
    return f"{row.get('task_id', '')}:seed{row.get('seed', 0)}"


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
