from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate raw, oracle ReCAP, and optional learned candidate ranking."
    )
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("analysis/recap_candidate_ranking.json"))
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.preferences))
    predictions = (
        index_predictions(tuple(read_jsonl(args.predictions)))
        if args.predictions is not None
        else {}
    )
    records = evaluate_candidate_ranking(preferences, predictions)
    output = {
        "summary": summarize_ranking(records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def evaluate_candidate_ranking(
    preferences: tuple[Mapping[str, Any], ...],
    predictions: Mapping[tuple[str, int, int], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    predictions = predictions or {}
    records: list[dict[str, Any]] = []
    for preference in preferences:
        candidates = tuple(str(action) for action in preference.get("candidates", ()))
        preferred = str(preference["preferred_action"])
        rejected = str(preference["rejected_action"])
        raw_rank = rank(candidates, preferred)
        rejected_rank = rank(candidates, rejected)
        key = preference_key(preference)
        prediction = predictions.get(key)
        learned_rank = None
        learned_top1 = None
        learned_abstained = None
        if prediction is not None:
            learned_abstained = bool(
                prediction.get("abstain", prediction.get("abstained", False))
            )
            if not learned_abstained:
                learned_order = learned_candidate_order(candidates, prediction)
                learned_rank = rank(learned_order, preferred)
                learned_top1 = learned_order[0] if learned_order else None

        raw_top1_is_preferred = raw_rank == 1
        learned_top1_is_preferred = learned_rank == 1
        learned_demoted_raw_gold = (
            bool(raw_top1_is_preferred)
            and prediction is not None
            and learned_abstained is False
            and not learned_top1_is_preferred
        )
        records.append(
            {
                "task_id": str(preference["task_id"]),
                "seed": int(preference.get("seed", 0)),
                "step_index": int(preference.get("step_index", len(preference.get("history", ())))),
                "preferred_action": preferred,
                "rejected_action": rejected,
                "preferred_rank_before": raw_rank,
                "rejected_rank_before": rejected_rank,
                "preferred_in_candidates": raw_rank is not None,
                "raw_top1_is_preferred": raw_top1_is_preferred,
                "raw_mrr": reciprocal_rank(raw_rank),
                "oracle_recap_mrr": 1.0 if raw_rank is not None else 0.0,
                "raw_ndcg": single_relevant_ndcg(raw_rank),
                "oracle_recap_ndcg": 1.0 if raw_rank is not None else 0.0,
                "has_learned_prediction": prediction is not None,
                "learned_abstained": learned_abstained,
                "learned_rank": learned_rank,
                "learned_top1": learned_top1,
                "learned_mrr": (
                    reciprocal_rank(learned_rank)
                    if prediction is not None and learned_abstained is False
                    else None
                ),
                "learned_ndcg": (
                    single_relevant_ndcg(learned_rank)
                    if prediction is not None and learned_abstained is False
                    else None
                ),
                "learned_top1_is_preferred": (
                    learned_top1_is_preferred
                    if prediction is not None and learned_abstained is False
                    else None
                ),
                "learned_demoted_raw_gold": learned_demoted_raw_gold,
            }
        )
    return records


def learned_candidate_order(
    candidates: tuple[str, ...],
    prediction: Mapping[str, Any],
) -> tuple[str, ...]:
    ranked_actions = prediction.get("ranked_actions")
    if ranked_actions is not None:
        order: list[str] = []
        for action in ranked_actions:
            action_str = str(action)
            if action_str in candidates and action_str not in order:
                order.append(action_str)
        order.extend(action for action in candidates if action not in order)
        return tuple(order)

    scores = prediction_scores(prediction)
    indexed = tuple(enumerate(candidates))
    return tuple(
        action
        for _index, action in sorted(
            indexed,
            key=lambda item: (-scores.get(item[1], float("-inf")), item[0]),
        )
    )


def prediction_scores(prediction: Mapping[str, Any]) -> dict[str, float]:
    raw_scores = prediction.get("scores", prediction.get("candidate_scores", {}))
    if isinstance(raw_scores, Mapping):
        return {str(action): float(score) for action, score in raw_scores.items()}
    scores: dict[str, float] = {}
    if isinstance(raw_scores, list):
        for item in raw_scores:
            if isinstance(item, Mapping) and "action" in item and "score" in item:
                scores[str(item["action"])] = float(item["score"])
    return scores


def summarize_ranking(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    predicted = [record for record in records if record.get("has_learned_prediction")]
    learned = [record for record in predicted if record.get("learned_abstained") is False]
    raw_gold = [
        record
        for record in learned
        if record.get("raw_top1_is_preferred")
    ]
    learned_demotions = sum(1 for record in raw_gold if record.get("learned_demoted_raw_gold"))
    learned_top1 = sum(1 for record in learned if record.get("learned_top1_is_preferred"))
    return {
        "preferences": total,
        "preferred_in_candidates": sum(1 for record in records if record["preferred_in_candidates"]),
        "candidate_coverage_rate": safe_div(
            sum(1 for record in records if record["preferred_in_candidates"]),
            total,
        ),
        "preferred_top1_before": sum(
            1 for record in records if record.get("preferred_rank_before") == 1
        ),
        "avg_preferred_rank_before": average(
            record["preferred_rank_before"]
            for record in records
            if record.get("preferred_rank_before") is not None
        ),
        "raw_mrr": average(record["raw_mrr"] for record in records),
        "oracle_recap_mrr": average(record["oracle_recap_mrr"] for record in records),
        "learned_mrr": average_or_none(record["learned_mrr"] for record in learned),
        "raw_ndcg": average(record["raw_ndcg"] for record in records),
        "oracle_recap_ndcg": average(record["oracle_recap_ndcg"] for record in records),
        "learned_ndcg": average_or_none(record["learned_ndcg"] for record in learned),
        "learned_predictions": len(predicted),
        "learned_evaluated_preferences": len(learned),
        "learned_prediction_coverage_rate": safe_div(len(predicted), total),
        "learned_abstain_rate": (
            safe_div(
                sum(1 for record in predicted if record.get("learned_abstained") is True),
                len(predicted),
            )
            if predicted
            else None
        ),
        "learned_top1_correction_rate": (
            safe_div(learned_top1, len(learned)) if learned else None
        ),
        "gold_demotion_rate": (
            safe_div(learned_demotions, len(raw_gold)) if raw_gold else None
        ),
        "learned_harm_rate": (
            safe_div(learned_demotions, len(raw_gold)) if raw_gold else None
        ),
    }


def read_jsonl(path: Path | None) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    records: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def index_predictions(
    predictions: tuple[Mapping[str, Any], ...],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    return {preference_key(record): record for record in predictions}


def preference_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["task_id"]),
        int(record.get("seed", 0)),
        int(record.get("step_index", len(record.get("history", ())))),
    )


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


def average(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def average_or_none(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    return sum(items) / len(items) if items else None


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
