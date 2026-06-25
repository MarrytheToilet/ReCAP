from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from acta.eval.eval_candidate_ranking import evaluate_candidate_ranking, index_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reranker gold demotion on success-step retention records."
    )
    parser.add_argument("--retention", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    retention = tuple(read_jsonl(args.retention))
    predictions = tuple(read_jsonl(args.predictions))
    records = evaluate_candidate_ranking(retention, index_predictions(predictions))
    summary = summarize_gold_demotion(records)
    output = {
        "summary": summary,
        "records": records,
        "inputs": {
            "retention": str(args.retention),
            "predictions": str(args.predictions),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def summarize_gold_demotion(records: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    predicted = [record for record in records if record.get("has_learned_prediction")]
    abstained = [record for record in predicted if record.get("learned_abstained") is True]
    intervened = [record for record in predicted if record.get("learned_abstained") is False]
    demoted = [
        record
        for record in intervened
        if record.get("raw_top1_is_preferred") and not record.get("learned_top1_is_preferred")
    ]
    retained_when_intervened = [
        record
        for record in intervened
        if record.get("raw_top1_is_preferred") and record.get("learned_top1_is_preferred")
    ]
    effective_retained = len(abstained) + len(retained_when_intervened)
    return {
        "retention_records": total,
        "predicted_records": len(predicted),
        "prediction_coverage_rate": safe_div(len(predicted), total),
        "abstained": len(abstained),
        "abstain_rate": safe_div(len(abstained), len(predicted)) if predicted else None,
        "intervened": len(intervened),
        "intervention_rate": safe_div(len(intervened), len(predicted)) if predicted else None,
        "gold_demotions": len(demoted),
        "gold_demotion_rate_over_all_predicted": (
            safe_div(len(demoted), len(predicted)) if predicted else None
        ),
        "gold_demotion_rate_when_intervened": (
            safe_div(len(demoted), len(intervened)) if intervened else None
        ),
        "gold_retention_rate_with_abstention": (
            safe_div(effective_retained, len(predicted)) if predicted else None
        ),
        "gold_retention_rate_when_intervened": (
            safe_div(len(retained_when_intervened), len(intervened)) if intervened else None
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
