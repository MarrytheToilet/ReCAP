from __future__ import annotations

import argparse
import json
from pathlib import Path

from recap.models.exact_memory_reranker import build_exact_memory_model
from recap.models.reranker_dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an exact-memory ReCAP reranker baseline.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--negative-scope",
        choices=["rejected", "all-candidates"],
        default="rejected",
    )
    parser.add_argument("--recent-history", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_records = tuple(read_jsonl(args.train))
    model = build_exact_memory_model(
        train_records=train_records,
        negative_scope=args.negative_scope,
        recent_history=args.recent_history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(model["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
