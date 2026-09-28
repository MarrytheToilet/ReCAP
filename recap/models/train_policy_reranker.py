from __future__ import annotations

import argparse
from pathlib import Path

import torch

from recap.models.policy_reranker import train_policy_reranker
from recap.models.reranker_dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a support-constrained ReCAP candidate policy.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-set", default="all")
    parser.add_argument("--soft-target-weight", type=float, default=0.0)
    parser.add_argument("--reward-temperature", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.train))
    validation_preferences = tuple(read_jsonl(args.valid)) if args.valid is not None else None
    model = train_policy_reranker(
        preferences,
        validation_preferences=validation_preferences,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        weight_decay=args.weight_decay,
        feature_set=args.feature_set,
        soft_target_weight=args.soft_target_weight,
        reward_temperature=args.reward_temperature,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.out)
    print(model["summary"])
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
