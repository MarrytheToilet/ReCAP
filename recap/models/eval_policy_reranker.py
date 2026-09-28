from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from recap.eval.eval_candidate_ranking import evaluate_candidate_ranking, summarize_ranking
from recap.models.policy_reranker import load_policy_model, rank_candidates_with_model
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a support-constrained ReCAP candidate policy.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    parser.add_argument(
        "--out-predictions",
        type=Path,
        default=Path("analysis/recap_policy_predictions.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/recap_policy_eval.json"),
    )
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.test))
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    predictions = predict_preferences(preferences, model, abstain_margin=args.abstain_margin)
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
    abstain_margin: float = 0.0,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    loaded_model = load_policy_model(model)
    for preference in preferences:
        ranked = rank_candidates_with_model(preference, loaded_model, model)
        scores = {action: score for action, score in ranked}
        ranked_actions = [action for action, _score in ranked]
        candidates = tuple(str(action) for action in preference.get("candidates", ()))
        raw_top = candidates[0] if candidates else None
        learned_top = ranked_actions[0] if ranked_actions else None
        margin_over_raw = (
            scores.get(learned_top, 0.0) - scores.get(raw_top, 0.0)
            if learned_top is not None and raw_top is not None
            else 0.0
        )
        abstain = (
            bool(abstain_margin > 0)
            and learned_top != raw_top
            and margin_over_raw < abstain_margin
        )
        predictions.append(
            {
                "task_id": str(preference["task_id"]),
                "seed": int(preference.get("seed", 0)),
                "step_index": int(preference.get("step_index", len(preference.get("history", ())))),
                "ranked_actions": ranked_actions,
                "scores": scores,
                "abstain": abstain,
                "margin_over_raw_top1": margin_over_raw,
                "model_type": str(model.get("model_type", "unknown")),
            }
        )
    return predictions


if __name__ == "__main__":
    main()
