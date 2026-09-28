from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from recap.eval.eval_candidate_ranking import (
    evaluate_candidate_ranking,
    index_predictions,
    summarize_ranking,
)
from recap.eval.eval_gold_demotion import summarize_gold_demotion
from recap.models.intervention_gate import apply_gate_to_predictions, load_gate_model
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a learned intervention-gate threshold by maximizing validation "
            "correction under a retention demotion constraint."
        )
    )
    parser.add_argument("--valid-records", type=Path, required=True)
    parser.add_argument("--valid-predictions", type=Path, required=True)
    parser.add_argument("--retention-records", type=Path, required=True)
    parser.add_argument("--retention-predictions", type=Path, required=True)
    parser.add_argument("--gate-model", type=Path, required=True)
    parser.add_argument("--max-retention-demotion", type=float, default=0.025)
    parser.add_argument("--min-intervention-rate", type=float, default=0.0)
    parser.add_argument(
        "--tie-break",
        choices=["coverage", "safety"],
        default="coverage",
        help=(
            "When validation correction is tied, prefer either intervention "
            "coverage or the lowest retention demotion rate."
        ),
    )
    parser.add_argument("--grid-start", type=float, default=0.50)
    parser.add_argument("--grid-stop", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-valid-predictions", type=Path, default=None)
    parser.add_argument("--out-retention-predictions", type=Path, default=None)
    args = parser.parse_args()

    valid_records = tuple(read_jsonl(args.valid_records))
    valid_predictions = index_predictions(tuple(read_jsonl(args.valid_predictions)))
    retention_records = tuple(read_jsonl(args.retention_records))
    retention_predictions = index_predictions(tuple(read_jsonl(args.retention_predictions)))
    gate_model = load_gate_model(args.gate_model)
    if gate_model is None:
        raise SystemExit("--gate-model is required")

    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid(args.grid_start, args.grid_stop, args.grid_step):
        valid_gated = apply_gate_to_predictions(
            records=valid_records,
            predictions=valid_predictions,
            gate_model=gate_model,
            threshold=threshold,
        )
        retention_gated = apply_gate_to_predictions(
            records=retention_records,
            predictions=retention_predictions,
            gate_model=gate_model,
            threshold=threshold,
        )
        valid_records_ranked = evaluate_candidate_ranking(
            valid_records,
            index_predictions(tuple(valid_gated)),
        )
        valid_summary = summarize_ranking(valid_records_ranked)
        retention_records_ranked = evaluate_candidate_ranking(
            retention_records,
            index_predictions(tuple(retention_gated)),
        )
        retention_summary = summarize_gold_demotion(retention_records_ranked)
        valid_correct = sum(
            1
            for record in valid_records_ranked
            if record.get("learned_abstained") is False
            and record.get("learned_top1_is_preferred") is True
        )
        rows.append(
            {
                "threshold": threshold,
                "valid_overall_top1_correction": valid_correct / len(valid_records)
                if valid_records
                else 0.0,
                "valid_nonabstained_top1_correction": valid_summary[
                    "learned_top1_correction_rate"
                ],
                "valid_abstain_rate": valid_summary["learned_abstain_rate"],
                "valid_mrr": valid_summary["learned_mrr"],
                "retention_demotion_rate": retention_summary[
                    "gold_demotion_rate_over_all_predicted"
                ],
                "retention_intervention_rate": retention_summary["intervention_rate"],
            }
        )

    feasible = [
        row
        for row in rows
        if row["retention_demotion_rate"] <= args.max_retention_demotion
        and row["retention_intervention_rate"] >= args.min_intervention_rate
    ]
    selected = max(feasible or rows, key=lambda row: selection_key(row, args.tie_break))
    output = {
        "selected_threshold": selected["threshold"],
        "selection_rule": (
            "maximize validation overall top-1 correction subject to retention "
            "gold-demotion and minimum-intervention constraints"
        ),
        "tie_break": args.tie_break,
        "max_retention_demotion": args.max_retention_demotion,
        "min_intervention_rate": args.min_intervention_rate,
        "selected": selected,
        "grid": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_valid_predictions is not None:
        write_jsonl(
            args.out_valid_predictions,
            apply_gate_to_predictions(
                records=valid_records,
                predictions=valid_predictions,
                gate_model=gate_model,
                threshold=float(selected["threshold"]),
            ),
        )
    if args.out_retention_predictions is not None:
        write_jsonl(
            args.out_retention_predictions,
            apply_gate_to_predictions(
                records=retention_records,
                predictions=retention_predictions,
                gate_model=gate_model,
                threshold=float(selected["threshold"]),
            ),
        )
    print(json.dumps(output["selected"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 6))
        value += step
    return values


def selection_key(row: Mapping[str, Any], tie_break: str) -> tuple[float, float, float]:
    correction = float(row["valid_overall_top1_correction"])
    intervention = float(row["retention_intervention_rate"])
    demotion = float(row["retention_demotion_rate"])
    if tie_break == "safety":
        return (correction, -demotion, intervention)
    return (correction, intervention, -demotion)


if __name__ == "__main__":
    main()
