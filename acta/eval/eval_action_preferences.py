from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from acta.agents.preference_agent import ActionPreference, load_action_preferences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate compiled action preferences against recorded candidate lists."
    )
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--run-key", default=None)
    parser.add_argument("--preference-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("analysis/action_preference_eval.json"))
    args = parser.parse_args()

    run_payload = json.loads(args.run_json.read_text(encoding="utf-8"))
    if args.run_key is not None:
        run_payload = run_payload[args.run_key]
    preferences = load_action_preferences(args.preference_data)
    records = evaluate_preferences(run_payload, preferences)
    output = {
        "summary": summarize_records(records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def evaluate_preferences(
    run_payload: Mapping[str, Any],
    preferences: tuple[ActionPreference, ...],
) -> list[dict[str, Any]]:
    steps = recorded_steps(run_payload)
    records: list[dict[str, Any]] = []
    for preference in preferences:
        key = (preference.task_id, preference.seed, len(preference.history))
        step = steps.get(key)
        if step is None:
            key = (Path(preference.task_id).stem, preference.seed, len(preference.history))
            step = steps.get(key)
        if step is None:
            records.append(record_missing(preference))
            continue

        candidates = tuple(str(action) for action in step_candidates(step))
        preferred_rank = rank(candidates, preference.preferred_action)
        rejected_rank = rank(candidates, preference.rejected_action)
        selected = str(step.get("action", ""))
        records.append(
            {
                "task_id": preference.task_id,
                "seed": preference.seed,
                "step_index": len(preference.history),
                "preferred_action": preference.preferred_action,
                "rejected_action": preference.rejected_action,
                "selected_action": selected,
                "preferred_rank_before": preferred_rank,
                "rejected_rank_before": rejected_rank,
                "selected_was_rejected": selected == preference.rejected_action,
                "preferred_in_candidates": preferred_rank is not None,
                "would_make_preferred_top1": preferred_rank is not None,
                "preferred_misranked": (
                    preferred_rank is not None and preferred_rank > 1
                ),
                "raw_mrr": reciprocal_rank(preferred_rank),
                "oracle_recap_mrr": 1.0 if preferred_rank is not None else 0.0,
                "raw_ndcg": single_relevant_ndcg(preferred_rank),
                "oracle_recap_ndcg": 1.0 if preferred_rank is not None else 0.0,
                "source": preference.source,
            }
        )
    return records


def recorded_steps(run_payload: Mapping[str, Any]) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    steps: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for episode in run_payload.get("episodes", ()):
        task_id = str(episode["task_id"])
        stem = Path(task_id).stem
        seed = int(episode.get("seed", 0))
        for step in episode.get("steps", ()):
            step_index = int(step.get("step_index", 0))
            steps[(task_id, seed, step_index)] = step
            steps[(stem, seed, step_index)] = step
    return steps


def record_missing(preference: ActionPreference) -> dict[str, Any]:
    return {
        "task_id": preference.task_id,
        "seed": preference.seed,
        "step_index": len(preference.history),
        "preferred_action": preference.preferred_action,
        "rejected_action": preference.rejected_action,
        "selected_action": None,
        "preferred_rank_before": None,
        "rejected_rank_before": None,
        "selected_was_rejected": False,
        "preferred_in_candidates": False,
        "would_make_preferred_top1": False,
        "preferred_misranked": False,
        "raw_mrr": 0.0,
        "oracle_recap_mrr": 0.0,
        "raw_ndcg": 0.0,
        "oracle_recap_ndcg": 0.0,
        "source": preference.source,
        "missing_step": True,
    }


def step_candidates(step: Mapping[str, Any]) -> tuple[str, ...]:
    raw = step.get("candidates_before") or step.get("candidates") or ()
    return tuple(str(action) for action in raw)


def rank(candidates: tuple[str, ...], action: str) -> int | None:
    try:
        return candidates.index(action) + 1
    except ValueError:
        return None


def reciprocal_rank(rank_value: int | None) -> float:
    return 1.0 / rank_value if rank_value is not None and rank_value > 0 else 0.0


def single_relevant_ndcg(rank_value: int | None) -> float:
    if rank_value is None or rank_value <= 0:
        return 0.0
    return 1.0 / math.log2(rank_value + 1)


def summarize_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    matched = sum(1 for record in records if not record.get("missing_step"))
    preferred_in_candidates = sum(1 for record in records if record["preferred_in_candidates"])
    would_make_preferred_top1 = sum(
        1 for record in records if record["would_make_preferred_top1"]
    )
    preferred_ranks = [
        int(record["preferred_rank_before"])
        for record in records
        if record.get("preferred_rank_before") is not None
    ]
    raw_mrr = sum(float(record.get("raw_mrr", 0.0)) for record in records)
    oracle_recap_mrr = sum(float(record.get("oracle_recap_mrr", 0.0)) for record in records)
    raw_ndcg = sum(float(record.get("raw_ndcg", 0.0)) for record in records)
    oracle_recap_ndcg = sum(float(record.get("oracle_recap_ndcg", 0.0)) for record in records)
    return {
        "preferences": total,
        "matched_steps": matched,
        "preferred_in_candidates": preferred_in_candidates,
        "candidate_coverage_rate": preferred_in_candidates / total if total else 0.0,
        "candidate_absent_rate": (
            (matched - preferred_in_candidates) / matched if matched else 0.0
        ),
        "selected_was_rejected": sum(1 for record in records if record["selected_was_rejected"]),
        "selected_is_rejected_rate": (
            sum(1 for record in records if record["selected_was_rejected"]) / matched
            if matched
            else 0.0
        ),
        "preferred_top1_before": sum(
            1 for record in records if record.get("preferred_rank_before") == 1
        ),
        "preferred_misranked": sum(
            1 for record in records if record.get("preferred_misranked")
        ),
        "misranking_rate": (
            sum(1 for record in records if record.get("preferred_misranked"))
            / preferred_in_candidates
            if preferred_in_candidates
            else 0.0
        ),
        "would_make_preferred_top1": would_make_preferred_top1,
        "oracle_top1_repairable_rate": (
            would_make_preferred_top1 / preferred_in_candidates
            if preferred_in_candidates
            else 0.0
        ),
        "avg_preferred_rank_before": (
            sum(preferred_ranks) / len(preferred_ranks) if preferred_ranks else 0.0
        ),
        "raw_mrr": raw_mrr / total if total else 0.0,
        "oracle_recap_mrr": oracle_recap_mrr / total if total else 0.0,
        "learned_mrr": None,
        "raw_ndcg": raw_ndcg / total if total else 0.0,
        "oracle_recap_ndcg": oracle_recap_ndcg / total if total else 0.0,
        "learned_ndcg": None,
        "learned_top1_correction_rate": None,
        "learned_abstain_rate": None,
        "learned_harm_rate": None,
    }


if __name__ == "__main__":
    main()
