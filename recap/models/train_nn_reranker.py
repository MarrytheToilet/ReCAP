from __future__ import annotations

import argparse
import json
from pathlib import Path

from recap.models.nn_memory_reranker import build_memory_model, pairwise_accuracy
from recap.models.reranker_dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ReCAP nearest-neighbor memory reranker.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, default=None)
    parser.add_argument(
        "--negative-scope",
        choices=["rejected", "all-candidates"],
        default="rejected",
    )
    parser.add_argument("--recent-history", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--max-exemplars", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_records = tuple(read_jsonl(args.train))
    valid_records = tuple(read_jsonl(args.valid)) if args.valid is not None else ()
    model = build_memory_model(
        train_records=train_records,
        negative_scope=args.negative_scope,
        recent_history=args.recent_history,
        max_exemplars=args.max_exemplars,
        neighbors=args.neighbors,
    )
    model["summary"]["valid_preferences"] = len(valid_records)
    model["summary"]["valid_pairwise_accuracy"] = (
        pairwise_accuracy(
            valid_records,
            model,
            neighbors=args.neighbors,
            recent_history=args.recent_history,
        )
        if valid_records
        else None
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(model["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
