from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from recap.models.cross_encoder_reranker import candidate_text, context_text


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def joint_text(preference: Mapping[str, Any], action: str, max_history: int = 8) -> str:
    return f"{context_text(preference, max_history=max_history)}\n{candidate_text(action)}"


def candidate_text_rows(
    preferences: Sequence[Mapping[str, Any]],
    max_history: int = 8,
) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []
    for preference in preferences:
        preferred = str(preference["preferred_action"])
        for action in preference.get("candidates", ()):
            action_str = str(action)
            texts.append(joint_text(preference, action_str, max_history=max_history))
            labels.append(1 if action_str == preferred else 0)
    return texts, np.asarray(labels, dtype=int)


def encode_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    local_files_only: bool = False,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name, local_files_only=local_files_only)
    return np.asarray(
        encoder.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=float,
    )


def fit_embedding_reranker(
    preferences: Sequence[Mapping[str, Any]],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    max_history: int = 8,
    local_files_only: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    texts, labels = candidate_text_rows(preferences, max_history=max_history)
    if len(set(labels.tolist())) < 2:
        raise ValueError("embedding reranker requires at least one positive and one negative candidate")
    embeddings = encode_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    estimator = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    estimator.fit(embeddings, labels)
    return {
        "model_type": "recap_embedding_logistic_reranker",
        "embedding_model": model_name,
        "estimator": estimator,
        "hyperparameters": {
            "batch_size": batch_size,
            "max_history": max_history,
            "local_files_only": local_files_only,
            "seed": seed,
            "training_objective": "pointwise_candidate_classification_on_sentence_embeddings",
        },
        "summary": {
            "train_preferences": len(preferences),
            "train_candidate_rows": len(texts),
            "positive_rows": int(labels.sum()),
            "negative_rows": int((1 - labels).sum()),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        },
    }


def score_candidate(
    preference: Mapping[str, Any],
    action: str,
    model: Mapping[str, Any],
    batch_size: int = 32,
) -> float:
    scores = score_candidates(preference, tuple([action]), model, batch_size=batch_size)
    return float(scores[action])


def score_candidates(
    preference: Mapping[str, Any],
    candidates: Sequence[str],
    model: Mapping[str, Any],
    batch_size: int = 32,
) -> dict[str, float]:
    texts = [
        joint_text(
            preference,
            str(action),
            max_history=int(model.get("hyperparameters", {}).get("max_history", 8)),
        )
        for action in candidates
    ]
    embeddings = encode_texts(
        texts,
        model_name=str(model["embedding_model"]),
        batch_size=batch_size,
        local_files_only=bool(model.get("hyperparameters", {}).get("local_files_only", False)),
    )
    estimator = model["estimator"]
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(embeddings)
        classes = list(getattr(estimator, "classes_", (0, 1)))
        positive_index = classes.index(1) if 1 in classes else -1
        raw_scores = proba[:, positive_index]
    else:
        raw_scores = estimator.decision_function(embeddings)
    return {str(action): float(score) for action, score in zip(candidates, raw_scores)}


def rank_candidates(
    preference: Mapping[str, Any],
    model: Mapping[str, Any],
    batch_size: int = 32,
) -> tuple[tuple[str, float], ...]:
    candidates = tuple(str(action) for action in preference.get("candidates", ()))
    scores = score_candidates(preference, candidates, model, batch_size=batch_size)
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda item: (-scores.get(item[1], 0.0), item[0]))
    return tuple((action, scores.get(action, 0.0)) for _index, action in indexed)
