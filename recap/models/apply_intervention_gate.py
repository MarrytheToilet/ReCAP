from __future__ import annotations

import argparse
import json
from pathlib import Path

from recap.models.intervention_gate import (
    apply_gate_to_predictions,
    index_predictions,
    load_gate_model,
)
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a learned ReCAP intervention gate.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gate-model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = tuple(read_jsonl(args.records))
    predictions = index_predictions(read_jsonl(args.predictions))
    gate_model = load_gate_model(args.gate_model)
    if gate_model is None:
        raise SystemExit("--gate-model is required")
    gated = apply_gate_to_predictions(
        records=records,
        predictions=predictions,
        gate_model=gate_model,
        threshold=args.threshold,
    )
    write_jsonl(args.out_predictions, gated)
    summary = summarize_gated_predictions(gated)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"predictions={args.out_predictions}")


def summarize_gated_predictions(predictions: list[dict]) -> dict[str, float | int]:
    total = len(predictions)
    abstained = sum(1 for item in predictions if item.get("abstain") is True)
    gated = sum(1 for item in predictions if item.get("abstain_reason") == "learned_intervention_gate")
    probabilities = [
        float(item["intervention_gate_probability"])
        for item in predictions
        if "intervention_gate_probability" in item
    ]
    return {
        "predictions": total,
        "abstained": abstained,
        "abstain_rate": abstained / total if total else 0.0,
        "learned_gate_abstentions": gated,
        "mean_gate_probability": (
            sum(probabilities) / len(probabilities) if probabilities else 0.0
        ),
    }


if __name__ == "__main__":
    main()
