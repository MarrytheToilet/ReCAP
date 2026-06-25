#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acta.eval.eval_candidate_ranking import evaluate_candidate_ranking, index_predictions, summarize_ranking
from acta.eval.make_recap_splits import make_splits
from acta.models.eval_action_reranker import predict_preferences as predict_feature_preferences
from acta.models.eval_policy_reranker import predict_preferences as predict_policy_preferences
from acta.models.reranker_dataset import read_jsonl, write_jsonl
from acta.models.policy_reranker import train_policy_reranker
from acta.models.train_action_reranker import train_feature_reranker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ReCAP reranker robustness across repeated group-safe splits."
    )
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--split-by", choices=["seed", "game_id", "trajectory_id"], default="seed")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--reranker", choices=["feature", "policy"], default="feature")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.preferences))
    rows = []
    for split_seed in range(args.seed_start, args.seed_start + args.num_splits):
        splits = make_splits(
            records=preferences,
            split_by=args.split_by,
            train_frac=args.train_frac,
            valid_frac=args.valid_frac,
            test_frac=args.test_frac,
            seed=split_seed,
        )
        if args.reranker == "policy":
            model = train_policy_reranker(
                tuple(splits["train"]),
                validation_preferences=tuple(splits["valid"]),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                entropy_coef=args.entropy_coef,
                seed=split_seed,
            )
            predictions = predict_policy_preferences(tuple(splits["test"]), model)
        else:
            model = train_feature_reranker(
                train_records=tuple(splits["train"]),
                valid_records=tuple(splits["valid"]),
            )
            predictions = predict_feature_preferences(tuple(splits["test"]), model)
        summary = summarize_ranking(
            evaluate_candidate_ranking(tuple(splits["test"]), index_predictions(tuple(predictions)))
        )
        row = {
            "split_seed": split_seed,
            "reranker": args.reranker,
            "train_preferences": len(splits["train"]),
            "valid_preferences": len(splits["valid"]),
            "test_preferences": len(splits["test"]),
            "raw_mrr": summary["raw_mrr"],
            "learned_mrr": summary["learned_mrr"],
            "raw_ndcg": summary["raw_ndcg"],
            "learned_ndcg": summary["learned_ndcg"],
            "learned_top1_correction_rate": summary["learned_top1_correction_rate"],
        }
        rows.append(row)
        if args.out_dir is not None:
            split_dir = args.out_dir / f"split_seed_{split_seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            for name, records in splits.items():
                write_jsonl(split_dir / f"{name}.jsonl", records)
            if args.reranker == "policy":
                import torch

                torch.save(model, split_dir / "model.pt")
            else:
                (split_dir / "model.json").write_text(
                    json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            write_jsonl(split_dir / "predictions.jsonl", predictions)
    output = {
        "summary": summarize_rows(rows),
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "train_preferences",
        "valid_preferences",
        "test_preferences",
        "raw_mrr",
        "learned_mrr",
        "raw_ndcg",
        "learned_ndcg",
        "learned_top1_correction_rate",
    )
    return {
        "splits": len(rows),
        **{
            metric: {
                "mean": mean_float(row[metric] for row in rows),
                "std": std_float(row[metric] for row in rows),
                "min": min(float(row[metric]) for row in rows),
                "max": max(float(row[metric]) for row in rows),
            }
            for metric in metrics
        },
    }


def mean_float(values: Any) -> float:
    items = [float(value) for value in values]
    return mean(items) if items else 0.0


def std_float(values: Any) -> float:
    items = [float(value) for value in values]
    return pstdev(items) if len(items) > 1 else 0.0


if __name__ == "__main__":
    main()
