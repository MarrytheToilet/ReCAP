from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

from acta.eval.eval_candidate_ranking import evaluate_candidate_ranking, summarize_ranking
from acta.eval.eval_recap_failure_decomposition import (
    normalize_step_ledger,
    read_jsonl,
    summarize_decomposition,
)


DEFAULT_DECOMPOSITION_METRICS = (
    "certified_misranking_step_rate",
    "candidate_absent_step_rate",
    "repair_same_as_executed_step_rate",
    "no_repair_found_step_rate",
    "avg_preferred_rank_on_certified_preferences",
)

DEFAULT_RANKING_METRICS = (
    "raw_mrr",
    "oracle_recap_mrr",
    "raw_ndcg",
    "oracle_recap_ndcg",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap ReCAP decomposition and ranking metrics.")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("analysis/recap_bootstrap_metrics.json"))
    args = parser.parse_args()

    ledger = tuple(read_jsonl(args.ledger))
    preferences = tuple(read_jsonl(args.preferences))
    output = bootstrap_metrics(
        ledger=ledger,
        preferences=preferences,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def bootstrap_metrics(
    ledger: tuple[Mapping[str, Any], ...],
    preferences: tuple[Mapping[str, Any], ...],
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    step_ledger = normalize_step_ledger(ledger)
    ledger_groups = group_by_trajectory(step_ledger)
    preference_groups = group_by_trajectory(preferences)
    trajectory_ids = sorted(set(ledger_groups) | set(preference_groups))
    rng = random.Random(seed)

    decomposition_point = summarize_decomposition(step_ledger, preferences)
    ranking_point = summarize_ranking(evaluate_candidate_ranking(preferences))
    decomposition_samples: dict[str, list[float]] = {
        name: [] for name in DEFAULT_DECOMPOSITION_METRICS
    }
    ranking_samples: dict[str, list[float]] = {name: [] for name in DEFAULT_RANKING_METRICS}

    if trajectory_ids:
        for _ in range(iterations):
            sampled_ids = [rng.choice(trajectory_ids) for _ in trajectory_ids]
            sampled_ledger: list[Mapping[str, Any]] = []
            sampled_preferences: list[Mapping[str, Any]] = []
            for trajectory_id in sampled_ids:
                sampled_ledger.extend(ledger_groups.get(trajectory_id, ()))
                sampled_preferences.extend(preference_groups.get(trajectory_id, ()))

            decomposition = summarize_decomposition(tuple(sampled_ledger), tuple(sampled_preferences))
            ranking = summarize_ranking(evaluate_candidate_ranking(tuple(sampled_preferences)))
            for name in DEFAULT_DECOMPOSITION_METRICS:
                decomposition_samples[name].append(float(decomposition.get(name, 0.0)))
            for name in DEFAULT_RANKING_METRICS:
                ranking_samples[name].append(float(ranking.get(name, 0.0)))

    return {
        "summary": {
            "iterations": iterations,
            "bootstrap_unit": "trajectory_id",
            "trajectory_groups": len(trajectory_ids),
            "preferences": len(preferences),
        },
        "decomposition": interval_block(decomposition_point, decomposition_samples),
        "ranking": interval_block(ranking_point, ranking_samples),
    }


def group_by_trajectory(
    records: tuple[Mapping[str, Any], ...],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        trajectory_id = f"{record.get('task_id', '')}:seed{record.get('seed', 0)}"
        grouped.setdefault(trajectory_id, []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def interval_block(
    point: Mapping[str, Any],
    samples: Mapping[str, list[float]],
) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            "point": float(point.get(name, 0.0) or 0.0),
            "ci_low": percentile(values, 0.025) if values else None,
            "ci_high": percentile(values, 0.975) if values else None,
        }
        for name, values in samples.items()
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


if __name__ == "__main__":
    main()
