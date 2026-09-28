from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from recap.eval.eval_candidate_ranking import evaluate_candidate_ranking, summarize_ranking
from recap.models.reranker_dataset import normalize_action, read_jsonl, tokenize, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple non-neural ReCAP reranker baselines.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument(
        "--strategy",
        choices=["navigation-prior", "anti-static", "learned-verb-prior"],
        required=True,
    )
    parser.add_argument(
        "--out-predictions",
        type=Path,
        default=Path("analysis/recap_heuristic_predictions.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/recap_heuristic_eval.json"),
    )
    args = parser.parse_args()

    test_records = tuple(read_jsonl(args.test))
    train_records = tuple(read_jsonl(args.train)) if args.train is not None else ()
    model = build_heuristic_model(args.strategy, train_records)
    predictions = predict_preferences(test_records, model)
    write_jsonl(args.out_predictions, predictions)
    prediction_index = {
        (
            str(prediction["task_id"]),
            int(prediction.get("seed", 0)),
            int(prediction["step_index"]),
        ): prediction
        for prediction in predictions
    }
    records = evaluate_candidate_ranking(test_records, prediction_index)
    output = {
        "summary": summarize_ranking(records),
        "records": records,
        "predictions": str(args.out_predictions),
        "model": model,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"predictions={args.out_predictions}")
    print(f"wrote={args.out}")


def build_heuristic_model(
    strategy: str,
    train_records: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    model: dict[str, Any] = {
        "model_type": f"recap_{strategy}",
        "strategy": strategy,
        "summary": {"train_preferences": len(train_records)},
    }
    if strategy == "learned-verb-prior":
        positive: Counter[str] = Counter()
        negative: Counter[str] = Counter()
        for preference in train_records:
            preferred = str(preference["preferred_action"])
            positive[action_verb(preferred)] += 1
            for candidate in preference.get("candidates", ()):
                candidate_str = str(candidate)
                if candidate_str != preferred:
                    negative[action_verb(candidate_str)] += 1
        verbs = sorted(set(positive) | set(negative))
        alpha = 1.0
        model["verb_log_odds"] = {
            verb: math.log((positive[verb] + alpha) / (negative[verb] + alpha))
            for verb in verbs
        }
        model["summary"].update(
            {
                "positive_verbs": dict(sorted(positive.items())),
                "negative_verbs": dict(sorted(negative.items())),
            }
        )
    return model


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


def rank_candidates(
    preference: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[tuple[str, float], ...]:
    candidates = tuple(str(candidate) for candidate in preference.get("candidates", ()))
    scored = [
        (
            index,
            action,
            score_candidate(preference, action, model, raw_index=index),
        )
        for index, action in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item[2], item[0]))
    return tuple((action, score) for _index, action, score in scored)


def score_candidate(
    preference: Mapping[str, Any],
    action: str,
    model: Mapping[str, Any],
    raw_index: int,
) -> float:
    strategy = str(model["strategy"])
    verb = action_verb(action)
    tokens = tokenize(normalize_action(action))
    raw_tie_break = -0.001 * raw_index
    if strategy == "navigation-prior":
        return (1.0 if verb == "go" else 0.0) + raw_tie_break
    if strategy == "anti-static":
        score = raw_tie_break
        if verb == "go":
            score += 1.0
        if verb in {"take", "drop", "put", "open", "close", "unlock", "lock", "insert"}:
            score += 0.35
        if verb in {"look", "inventory", "examine", "read"}:
            score -= 1.0
        if normalize_action(action) in {normalize_action(item) for item in preference.get("history", ())[-4:]}:
            score -= 0.4
        score += 0.02 * min(len(tokens), 6)
        return score
    if strategy == "learned-verb-prior":
        verb_scores = dict(model.get("verb_log_odds", {}))
        return float(verb_scores.get(verb, 0.0)) + raw_tie_break
    raise ValueError(f"unknown strategy: {strategy}")


def action_verb(action: str) -> str:
    tokens = tokenize(normalize_action(action))
    return tokens[0] if tokens else ""


if __name__ == "__main__":
    main()
