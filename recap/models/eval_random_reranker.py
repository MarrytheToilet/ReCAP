from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

from recap.eval.eval_candidate_ranking import evaluate_candidate_ranking, summarize_ranking
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a deterministic random reranking baseline.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-predictions",
        type=Path,
        default=Path("analysis/recap_random_predictions.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/recap_random_eval.json"),
    )
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.test))
    predictions = predict_random(preferences, seed=args.seed)
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
        "seed": args.seed,
        "model": "deterministic_random_reranker",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"predictions={args.out_predictions}")
    print(f"wrote={args.out}")


def predict_random(
    preferences: tuple[Mapping[str, Any], ...],
    seed: int = 0,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for index, preference in enumerate(preferences):
        candidates = [str(action) for action in preference.get("candidates", ())]
        rng = random.Random(seed + index)
        rng.shuffle(candidates)
        predictions.append(
            {
                "task_id": str(preference["task_id"]),
                "seed": int(preference.get("seed", 0)),
                "step_index": int(preference.get("step_index", len(preference.get("history", ())))),
                "ranked_actions": candidates,
                "scores": {action: float(len(candidates) - rank) for rank, action in enumerate(candidates)},
                "abstain": False,
                "model_type": "deterministic_random_reranker",
            }
        )
    return predictions


if __name__ == "__main__":
    main()
