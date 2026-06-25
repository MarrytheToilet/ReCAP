from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

from recap.eval.bootstrap_recap_metrics import percentile
from recap.eval.eval_candidate_ranking import (
    evaluate_candidate_ranking,
    index_predictions,
    summarize_ranking,
)


BOOTSTRAP_METRICS = (
    "raw_mrr",
    "learned_mrr",
    "raw_ndcg",
    "learned_ndcg",
    "learned_top1_correction_rate",
    "learned_abstain_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trajectory-cluster bootstrap for learned candidate ranking metrics."
    )
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.preferences))
    predictions = tuple(read_jsonl(args.predictions))
    output = bootstrap_candidate_ranking(
        preferences=preferences,
        predictions=predictions,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def bootstrap_candidate_ranking(
    preferences: tuple[Mapping[str, Any], ...],
    predictions: tuple[Mapping[str, Any], ...],
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    prediction_index = index_predictions(predictions)
    point = summarize_ranking(evaluate_candidate_ranking(preferences, prediction_index))
    groups = group_preferences_by_trajectory(preferences)
    group_names = sorted(groups)
    samples: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    rng = random.Random(seed)
    for _ in range(iterations):
        sampled_preferences: list[Mapping[str, Any]] = []
        for _group in group_names:
            sampled_preferences.extend(groups[rng.choice(group_names)])
        summary = summarize_ranking(
            evaluate_candidate_ranking(tuple(sampled_preferences), prediction_index)
        )
        for metric in BOOTSTRAP_METRICS:
            value = summary.get(metric)
            if value is not None:
                samples[metric].append(float(value))
    return {
        "summary": {
            "iterations": iterations,
            "bootstrap_unit": "trajectory_id",
            "trajectory_groups": len(group_names),
            "preferences": len(preferences),
            "predictions": len(predictions),
        },
        "metrics": {
            metric: {
                "point": point.get(metric),
                "ci_low": percentile(values, 0.025) if values else None,
                "ci_high": percentile(values, 0.975) if values else None,
            }
            for metric, values in samples.items()
        },
    }


def group_preferences_by_trajectory(
    records: tuple[Mapping[str, Any], ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        trajectory_id = f"{record.get('task_id', '')}:seed{record.get('seed', 0)}"
        grouped.setdefault(trajectory_id, []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
