from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from recap.eval.eval_candidate_ranking import evaluate_candidate_ranking, summarize_ranking
from recap.models.lm_candidate_policy import (
    candidate_scores,
    encode_candidate_batch,
    make_prediction,
    yes_no_token_ids,
)
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA LM ReCAP candidate policy.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--max-observation-chars", type=int, default=360)
    parser.add_argument("--candidate-chunk-size", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter or args.base_model), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(str(args.base_model), **model_kwargs)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    if not args.load_in_4bit:
        model.to(args.device)
    model.eval()

    yes_id, no_id = yes_no_token_ids(tokenizer)
    records = tuple(read_jsonl(args.test))
    if args.limit is not None:
        records = records[: args.limit]
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            batch = encode_candidate_batch(
                tokenizer,
                record,
                max_length=args.max_length,
                max_history=args.max_history,
                max_observation_chars=args.max_observation_chars,
                device=args.device,
            )
            if not batch.candidates:
                continue
            scores_tensor = candidate_scores(
                model,
                batch,
                yes_id=yes_id,
                no_id=no_id,
                chunk_size=args.candidate_chunk_size,
            )
            scores = {
                action: float(score)
                for action, score in zip(batch.candidates, scores_tensor.detach().float().cpu().tolist())
            }
            ranked = sorted(batch.candidates, key=lambda action: (-scores[action], batch.candidates.index(action)))
            raw_top = batch.candidates[0]
            learned_top = ranked[0]
            abstain = (
                args.abstain_margin > 0
                and learned_top != raw_top
                and scores[learned_top] - scores[raw_top] < args.abstain_margin
            )
            predictions.append(
                make_prediction(
                    record,
                    ranked_actions=ranked,
                    scores=scores,
                    abstain=abstain,
                    model_type="recap_lm_scpo_lora" if args.adapter else "recap_lm_zeroshot",
                )
            )

    write_jsonl(args.out_predictions, predictions)
    prediction_index = {
        (
            str(prediction["task_id"]),
            int(prediction.get("seed", 0)),
            int(prediction["step_index"]),
        ): prediction
        for prediction in predictions
    }
    ranking_records = evaluate_candidate_ranking(records, prediction_index)
    output = {
        "summary": summarize_ranking(ranking_records),
        "records": ranking_records,
        "predictions": str(args.out_predictions),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"predictions={args.out_predictions}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
