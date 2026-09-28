from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recap.models.reranker_dataset import FEATURE_NAMES, featurize_candidate


MODEL_TYPES = ("mlp", "gradient-boosting", "random-forest")


def candidate_rows(
    preferences: tuple[Mapping[str, Any], ...],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for preference in preferences:
        preferred = str(preference["preferred_action"])
        for action in tuple(str(candidate) for candidate in preference.get("candidates", ())):
            features = featurize_candidate(preference, action)
            rows.append([float(features.get(name, 0.0)) for name in FEATURE_NAMES])
            labels.append(1 if action == preferred else 0)
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,), dtype=int)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def build_classifier(model_type: str, seed: int = 0) -> Pipeline | GradientBoostingClassifier | RandomForestClassifier:
    if model_type == "mlp":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        solver="lbfgs",
                        alpha=0.01,
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_type == "gradient-boosting":
        return GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        )
    if model_type == "random-forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            random_state=seed,
            class_weight="balanced",
        )
    raise ValueError(f"unknown sklearn reranker model: {model_type}")


def fit_sklearn_reranker(
    preferences: tuple[Mapping[str, Any], ...],
    model_type: str = "mlp",
    seed: int = 0,
) -> dict[str, Any]:
    x, y = candidate_rows(preferences)
    if len(set(y.tolist())) < 2:
        raise ValueError("sklearn reranker requires at least one positive and one negative candidate")
    estimator = build_classifier(model_type, seed=seed)
    estimator.fit(x, y)
    return {
        "model_type": f"recap_sklearn_{model_type}",
        "estimator": estimator,
        "feature_names": tuple(FEATURE_NAMES),
        "hyperparameters": {
            "model_type": model_type,
            "seed": seed,
            "training_objective": "pointwise_candidate_classification",
        },
        "summary": {
            "train_preferences": len(preferences),
            "train_candidate_rows": int(x.shape[0]),
            "positive_rows": int(y.sum()),
            "negative_rows": int((1 - y).sum()),
        },
    }


def score_candidate(
    preference: Mapping[str, Any],
    action: str,
    model: Mapping[str, Any],
) -> float:
    estimator = model["estimator"]
    features = featurize_candidate(preference, action)
    x = np.asarray([[float(features.get(name, 0.0)) for name in FEATURE_NAMES]], dtype=float)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)[0]
        classes = list(getattr(estimator, "classes_", (0, 1)))
        if hasattr(estimator, "named_steps"):
            classes = list(estimator.named_steps["mlp"].classes_)
        try:
            return float(proba[classes.index(1)])
        except ValueError:
            return 0.0
    if hasattr(estimator, "decision_function"):
        return float(estimator.decision_function(x)[0])
    return float(estimator.predict(x)[0])


def rank_candidates(
    preference: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[tuple[str, float], ...]:
    candidates = tuple(str(candidate) for candidate in preference.get("candidates", ()))
    scored = [
        (index, action, score_candidate(preference, action, model))
        for index, action in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item[2], item[0]))
    return tuple((action, score) for _index, action, score in scored)
