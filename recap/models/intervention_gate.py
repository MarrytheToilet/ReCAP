from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recap.models.reranker_dataset import action_verb, read_jsonl, write_jsonl


GATE_FEATURE_NAMES = (
    "margin_over_raw_top1",
    "learned_score",
    "raw_score",
    "score_gap_top2",
    "raw_rank_of_learned_top",
    "candidate_count",
    "step_index_log1p",
    "history_len_log1p",
    "learned_is_navigation",
    "learned_is_manipulation",
    "learned_is_static",
    "raw_is_navigation",
    "raw_is_manipulation",
    "raw_is_static",
    "learned_recent_repeat",
    "raw_recent_repeat",
    "raw_suspicious",
    "demotes_manipulation_to_navigation",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a learned abstention gate over proposed reranker interventions."
    )
    parser.add_argument("--positive-records", type=Path, required=True)
    parser.add_argument("--positive-predictions", type=Path, required=True)
    parser.add_argument("--negative-records", type=Path, required=True)
    parser.add_argument("--negative-predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    positives = tuple(read_jsonl(args.positive_records))
    positive_predictions = index_predictions(read_jsonl(args.positive_predictions))
    negatives = tuple(read_jsonl(args.negative_records))
    negative_predictions = index_predictions(read_jsonl(args.negative_predictions))

    x, y, rows = gate_training_rows(
        positive_records=positives,
        positive_predictions=positive_predictions,
        negative_records=negatives,
        negative_predictions=negative_predictions,
    )
    if len(set(y.tolist())) < 2:
        raise ValueError("intervention gate requires both safe and unsafe proposed interventions")

    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    estimator.fit(x, y)
    model = {
        "model_type": "recap_intervention_gate",
        "estimator": estimator,
        "feature_names": tuple(GATE_FEATURE_NAMES),
        "threshold": float(args.threshold),
        "summary": {
            "training_rows": int(x.shape[0]),
            "positive_rows": int(y.sum()),
            "negative_rows": int((1 - y).sum()),
            "threshold": float(args.threshold),
            "seed": int(args.seed),
            "positive_records": len(positives),
            "negative_records": len(negatives),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    summary = dict(model["summary"])
    summary["training_accuracy"] = float(estimator.score(x, y))
    summary["feature_names"] = list(GATE_FEATURE_NAMES)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def gate_training_rows(
    positive_records: tuple[Mapping[str, Any], ...],
    positive_predictions: Mapping[tuple[str, int, int], Mapping[str, Any]],
    negative_records: tuple[Mapping[str, Any], ...],
    negative_predictions: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    audit_rows: list[dict[str, Any]] = []
    for record in positive_records:
        prediction = positive_predictions.get(record_key(record))
        if prediction is None:
            continue
        label = safe_intervention_label(record, prediction)
        if label is None:
            continue
        features = gate_feature_vector(record, prediction)
        rows.append(features)
        labels.append(label)
        audit_rows.append(audit_row(record, prediction, label, "certified_preference"))
    for record in negative_records:
        prediction = negative_predictions.get(record_key(record))
        if prediction is None or not proposes_intervention(record, prediction):
            continue
        features = gate_feature_vector(record, prediction)
        rows.append(features)
        labels.append(0)
        audit_rows.append(audit_row(record, prediction, 0, "success_retention"))
    if not rows:
        return (
            np.zeros((0, len(GATE_FEATURE_NAMES)), dtype=float),
            np.zeros((0,), dtype=int),
            audit_rows,
        )
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), audit_rows


def apply_gate_to_predictions(
    records: tuple[Mapping[str, Any], ...],
    predictions: Mapping[tuple[str, int, int], Mapping[str, Any]],
    gate_model: Mapping[str, Any],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    gated: list[dict[str, Any]] = []
    cutoff = float(gate_model.get("threshold", 0.5) if threshold is None else threshold)
    for record in records:
        prediction = predictions.get(record_key(record))
        if prediction is None:
            continue
        output = dict(prediction)
        probability = intervention_probability(record, prediction, gate_model)
        output["intervention_gate_probability"] = probability
        output["intervention_gate_threshold"] = cutoff
        if proposes_intervention(record, prediction) and probability < cutoff:
            output["abstain"] = True
            output["abstain_reason"] = "learned_intervention_gate"
        gated.append(output)
    return gated


def intervention_probability(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    gate_model: Mapping[str, Any],
) -> float:
    estimator = gate_model["estimator"]
    x = np.asarray([gate_feature_vector(record, prediction)], dtype=float)
    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(x)[0]
        classes = list(getattr(estimator, "classes_", (0, 1)))
        if hasattr(estimator, "named_steps"):
            classes = list(estimator.named_steps["logistic"].classes_)
        return float(probability[classes.index(1)]) if 1 in classes else 0.0
    if hasattr(estimator, "decision_function"):
        logit = float(estimator.decision_function(x)[0])
        return 1.0 / (1.0 + math.exp(-logit))
    return float(estimator.predict(x)[0])


def load_gate_model(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return joblib.load(path)


def index_predictions(
    predictions: list[Mapping[str, Any]],
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    return {record_key(prediction): prediction for prediction in predictions}


def record_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["task_id"]),
        int(record.get("seed", 0)),
        int(record.get("step_index", len(record.get("history", ())))),
    )


def safe_intervention_label(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> int | None:
    if not proposes_intervention(record, prediction):
        return None
    learned_top = learned_top_action(prediction)
    preferred = str(record.get("preferred_action", ""))
    return 1 if learned_top == preferred else 0


def proposes_intervention(record: Mapping[str, Any], prediction: Mapping[str, Any]) -> bool:
    candidates = tuple(str(candidate) for candidate in record.get("candidates", ()))
    if not candidates:
        return False
    learned_top = learned_top_action(prediction)
    return learned_top is not None and learned_top != candidates[0]


def learned_top_action(prediction: Mapping[str, Any]) -> str | None:
    ranked = prediction.get("ranked_actions", ())
    if ranked:
        return str(ranked[0])
    scores = prediction_scores(prediction)
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def gate_feature_vector(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> list[float]:
    features = gate_features(record, prediction)
    return [float(features.get(name, 0.0)) for name in GATE_FEATURE_NAMES]


def gate_features(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, float]:
    candidates = tuple(str(candidate) for candidate in record.get("candidates", ()))
    history = tuple(str(action) for action in record.get("history", ()))
    scores = prediction_scores(prediction)
    raw_top = candidates[0] if candidates else ""
    learned_top = learned_top_action(prediction) or raw_top
    ranked_scores = sorted(scores.values(), reverse=True)
    top2_gap = ranked_scores[0] - ranked_scores[1] if len(ranked_scores) >= 2 else 0.0
    raw_rank = candidates.index(learned_top) + 1 if learned_top in candidates else len(candidates) + 1
    raw_verb = action_verb(raw_top)
    learned_verb = action_verb(learned_top)
    learned_recent = learned_top in history[-4:]
    raw_recent = raw_top in history[-4:]
    raw_static = raw_verb in {"look", "inventory", "examine"}
    learned_static = learned_verb in {"look", "inventory", "examine"}
    raw_manip = raw_verb in {"take", "drop", "put", "open", "close", "unlock", "lock", "insert"}
    learned_manip = learned_verb in {"take", "drop", "put", "open", "close", "unlock", "lock", "insert"}
    raw_nav = raw_verb in {"go", "north", "south", "east", "west"}
    learned_nav = learned_verb in {"go", "north", "south", "east", "west"}
    margin = float(
        prediction.get(
            "margin_over_raw_top1",
            scores.get(learned_top, 0.0) - scores.get(raw_top, 0.0),
        )
    )
    return {
        "margin_over_raw_top1": margin,
        "learned_score": float(scores.get(learned_top, 0.0)),
        "raw_score": float(scores.get(raw_top, 0.0)),
        "score_gap_top2": float(top2_gap),
        "raw_rank_of_learned_top": float(raw_rank),
        "candidate_count": float(len(candidates)),
        "step_index_log1p": math.log1p(float(record.get("step_index", len(history)))),
        "history_len_log1p": math.log1p(float(len(history))),
        "learned_is_navigation": float(learned_nav),
        "learned_is_manipulation": float(learned_manip),
        "learned_is_static": float(learned_static),
        "raw_is_navigation": float(raw_nav),
        "raw_is_manipulation": float(raw_manip),
        "raw_is_static": float(raw_static),
        "learned_recent_repeat": float(learned_recent),
        "raw_recent_repeat": float(raw_recent),
        "raw_suspicious": float(raw_recent or raw_static),
        "demotes_manipulation_to_navigation": float(raw_manip and learned_nav),
    }


def prediction_scores(prediction: Mapping[str, Any]) -> dict[str, float]:
    raw_scores = prediction.get("scores", prediction.get("candidate_scores", {}))
    if isinstance(raw_scores, Mapping):
        return {str(action): float(score) for action, score in raw_scores.items()}
    scores: dict[str, float] = {}
    if isinstance(raw_scores, list):
        for item in raw_scores:
            if isinstance(item, Mapping) and "action" in item and "score" in item:
                scores[str(item["action"])] = float(item["score"])
    return scores


def audit_row(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    label: int,
    source: str,
) -> dict[str, Any]:
    return {
        "task_id": str(record["task_id"]),
        "seed": int(record.get("seed", 0)),
        "step_index": int(record.get("step_index", len(record.get("history", ())))),
        "source": source,
        "label": int(label),
        "learned_top": learned_top_action(prediction),
        "raw_top": tuple(str(candidate) for candidate in record.get("candidates", ("",)))[0],
        "preferred_action": record.get("preferred_action"),
        "margin_over_raw_top1": prediction.get("margin_over_raw_top1"),
    }


if __name__ == "__main__":
    main()
