from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from recap.models.reranker_dataset import normalize_action, tokenize


def build_memory_model(
    train_records: tuple[Mapping[str, Any], ...],
    negative_scope: str = "rejected",
    recent_history: int = 8,
    max_exemplars: int | None = None,
    neighbors: int = 3,
) -> dict[str, Any]:
    exemplars: list[dict[str, Any]] = []
    for preference in train_records:
        preferred = str(preference["preferred_action"])
        negatives = negative_actions(preference, negative_scope=negative_scope)
        exemplars.append(exemplar(preference, preferred, label="positive", recent_history=recent_history))
        for action in negatives:
            exemplars.append(exemplar(preference, action, label="negative", recent_history=recent_history))

    if max_exemplars is not None and len(exemplars) > max_exemplars:
        exemplars = exemplars[-max_exemplars:]

    model = {
        "model_type": "recap_nn_memory",
        "exemplars": exemplars,
        "hyperparameters": {
            "negative_scope": negative_scope,
            "recent_history": recent_history,
            "max_exemplars": max_exemplars,
            "neighbors": neighbors,
        },
        "summary": {
            "train_preferences": len(train_records),
            "exemplars": len(exemplars),
            "positive_exemplars": sum(1 for item in exemplars if item["label"] == "positive"),
            "negative_exemplars": sum(1 for item in exemplars if item["label"] == "negative"),
            "train_pairwise_accuracy": pairwise_accuracy(
                train_records,
                {"exemplars": exemplars},
                neighbors=neighbors,
                recent_history=recent_history,
            ),
        },
    }
    return model


def negative_actions(
    preference: Mapping[str, Any],
    negative_scope: str = "rejected",
) -> tuple[str, ...]:
    preferred = str(preference["preferred_action"])
    if negative_scope == "rejected":
        return (str(preference["rejected_action"]),)
    if negative_scope != "all-candidates":
        raise ValueError(f"unknown negative scope: {negative_scope}")
    negatives = tuple(
        str(candidate)
        for candidate in preference.get("candidates", ())
        if str(candidate) != preferred
    )
    return negatives or (str(preference["rejected_action"]),)


def exemplar(
    preference: Mapping[str, Any],
    action: str,
    label: str,
    recent_history: int = 8,
) -> dict[str, Any]:
    return {
        "label": label,
        "action": action,
        "task_id": str(preference.get("task_id", "")),
        "seed": int(preference.get("seed", 0)),
        "step_index": int(preference.get("step_index", len(preference.get("history", ())))),
        "token_counts": dict(candidate_counts(preference, action, recent_history=recent_history)),
    }


def score_candidate(
    preference: Mapping[str, Any],
    action: str,
    model: Mapping[str, Any],
    neighbors: int | None = None,
    recent_history: int | None = None,
) -> float:
    hyperparameters = dict(model.get("hyperparameters", {}))
    k = int(neighbors or hyperparameters.get("neighbors", 3))
    history_window = int(recent_history or hyperparameters.get("recent_history", 8))
    query = candidate_counts(preference, action, recent_history=history_window)
    positive: list[float] = []
    negative: list[float] = []
    for item in model.get("exemplars", ()):
        similarity = cosine(query, Counter({str(k): float(v) for k, v in item["token_counts"].items()}))
        if item.get("label") == "positive":
            positive.append(similarity)
        elif item.get("label") == "negative":
            negative.append(similarity)

    return topk_mean(positive, k) - topk_mean(negative, k)


def rank_candidates(
    preference: Mapping[str, Any],
    model: Mapping[str, Any],
    neighbors: int | None = None,
    recent_history: int | None = None,
) -> tuple[tuple[str, float], ...]:
    candidates = tuple(str(candidate) for candidate in preference.get("candidates", ()))
    scored = [
        (
            index,
            action,
            score_candidate(
                preference,
                action,
                model,
                neighbors=neighbors,
                recent_history=recent_history,
            ),
        )
        for index, action in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item[2], item[0]))
    return tuple((action, score) for _index, action, score in scored)


def pairwise_accuracy(
    records: tuple[Mapping[str, Any], ...],
    model: Mapping[str, Any],
    neighbors: int | None = None,
    recent_history: int | None = None,
) -> float:
    if not records:
        return 0.0
    correct = 0
    total = 0
    for preference in records:
        preferred = str(preference["preferred_action"])
        rejected = str(preference["rejected_action"])
        if score_candidate(
            preference,
            preferred,
            model,
            neighbors=neighbors,
            recent_history=recent_history,
        ) > score_candidate(
            preference,
            rejected,
            model,
            neighbors=neighbors,
            recent_history=recent_history,
        ):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def candidate_counts(
    preference: Mapping[str, Any],
    action: str,
    recent_history: int = 8,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    history = tuple(str(item) for item in preference.get("history", ()))
    for item in history[-recent_history:]:
        for token in tokenize(item):
            counts[f"h:{token}"] += 1.0

    action_norm = normalize_action(action)
    action_tokens = tokenize(action_norm)
    for token in action_tokens:
        counts[f"a:{token}"] += 2.0
    if action_tokens:
        counts[f"verb:{action_tokens[0]}"] += 2.0

    candidates = tuple(str(candidate) for candidate in preference.get("candidates", ()))
    try:
        rank = candidates.index(action) + 1
    except ValueError:
        rank = 0
    counts[f"rank:{rank}"] += 0.25
    counts[f"raw_top:{rank == 1}"] += 0.25
    return counts


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(float(value) * float(right.get(token, 0.0)) for token, value in left.items())
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def topk_mean(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    top = sorted(values, reverse=True)[: max(k, 1)]
    return sum(top) / len(top)
