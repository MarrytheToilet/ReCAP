from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from recap.models.reranker_dataset import read_jsonl
from recap.models.sklearn_reranker import MODEL_TYPES, fit_sklearn_reranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a nonlinear sklearn ReCAP reranker baseline.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.train))
    model = fit_sklearn_reranker(
        preferences=preferences,
        model_type=args.model_type,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(model["summary"])
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
