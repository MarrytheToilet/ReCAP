from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from recap.eval.eval_candidate_ranking import evaluate_candidate_ranking, summarize_ranking
from recap.models.exact_memory_reranker import rank_candidates
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an exact-memory ReCAP reranker baseline.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--out-predictions",
        type=Path,
        default=Path("analysis/recap_exact_memory_predictions.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/recap_exact_memory_eval.json"),
    )
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.test))
    model = json.loads(args.model.read_text(encoding="utf-8"))
    predictions = predict_preferences(preferences, model)
    write_jsonl(args.out_predictions, predictions)
    prediction_index = {
        (
            str(prediction["task_id"]),
            int(prediction.get("seed", 0)),
            int(prediction["step_index"]),
        ): prediction
        for prediction in predictions
    }
    records = evaluate_candidate_ranking(preferences, prediction_index)
    output = {
        "summary": summarize_ranking(records),
        "records": records,
        "predictions": str(args.out_predictions),
        "model": str(args.model),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"predictions={args.out_predictions}")
    print(f"wrote={args.out}")


def predict_preferences(
    preferences: tuple[Mapping[str, Any], ...],
    model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for preference in preferences:
        ranked = rank_candidates(preference, model)
        predictions.append(
            {
                "task_id": str(preference["task_id"]),
                "seed": int(preference.get("seed", 0)),
                "step_index": int(preference.get("step_index", len(preference.get("history", ())))),
                "ranked_actions": [action for action, _score in ranked],
                "scores": {action: score for action, score in ranked},
                "abstain": False,
                "model_type": str(model.get("model_type", "unknown")),
            }
        )
    return predictions


if __name__ == "__main__":
    main()
