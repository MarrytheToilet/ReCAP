from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from torch.utils.data import DataLoader

from recap.models.cross_encoder_reranker import pointwise_training_rows
from recap.models.reranker_dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sentence-transformers cross-encoder ReCAP reranker.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-observation-chars", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from sentence_transformers import CrossEncoder, InputExample

    train_records = tuple(read_jsonl(args.train))
    rows = pointwise_training_rows(
        train_records,
        max_history=args.max_history,
        max_observation_chars=args.max_observation_chars,
    )
    examples = [InputExample(texts=[context, action], label=label) for context, action, label in rows]
    dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    warmup_steps = max(1, int(len(dataloader) * args.epochs * 0.1))
    model.fit(
        train_dataloader=dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.out),
        save_best_model=False,
    )
    model.save(str(args.out))

    summary = {
        "model_type": "recap_cross_encoder_pointwise",
        "base_model": args.base_model,
        "train_preferences": len(train_records),
        "train_candidate_rows": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_history": args.max_history,
        "max_observation_chars": args.max_observation_chars,
        "device": args.device,
    }
    summary_path = args.summary_out or (args.out / "recap_training_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    model_summary_path = args.out / "recap_training_summary.json"
    if model_summary_path != summary_path:
        model_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
