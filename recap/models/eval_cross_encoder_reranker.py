from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from recap.eval.eval_candidate_ranking import evaluate_candidate_ranking, summarize_ranking
from recap.models.cross_encoder_reranker import rank_candidates
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained cross-encoder ReCAP reranker.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-observation-chars", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    parser.add_argument(
        "--out-predictions",
        type=Path,
        default=Path("analysis/recap_cross_encoder_predictions.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/recap_cross_encoder_eval.json"),
    )
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    preferences = tuple(read_jsonl(args.test))
    model = CrossEncoder(str(args.model), max_length=args.max_length, device=args.device)
    predictions = predict_preferences(
        preferences=preferences,
        model=model,
        batch_size=args.batch_size,
        max_history=args.max_history,
        max_observation_chars=args.max_observation_chars,
        abstain_margin=args.abstain_margin,
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
    model: Any,
    batch_size: int = 16,
    max_history: int = 8,
    max_observation_chars: int = 0,
    abstain_margin: float = 0.0,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for preference in preferences:
        ranked = rank_candidates(
            preference,
            model,
            batch_size=batch_size,
            max_history=max_history,
            max_observation_chars=max_observation_chars,
        )
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
                "model_type": "recap_cross_encoder_pointwise",
            }
        )
    return predictions


if __name__ == "__main__":
    main()
