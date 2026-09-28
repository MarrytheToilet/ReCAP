from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from recap.models.embedding_reranker import DEFAULT_EMBEDDING_MODEL, fit_embedding_reranker
from recap.models.reranker_dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sentence-embedding ReCAP reranker baseline.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_records = tuple(read_jsonl(args.train))
    model = fit_embedding_reranker(
        train_records,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        max_history=args.max_history,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(model["summary"])
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
