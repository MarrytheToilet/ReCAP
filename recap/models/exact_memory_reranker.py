from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from recap.models.nn_memory_reranker import negative_actions
from recap.models.reranker_dataset import normalize_action


def build_exact_memory_model(
    train_records: tuple[Mapping[str, Any], ...],
    negative_scope: str = "rejected",
    recent_history: int = 8,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for preference in train_records:
        preferred = str(preference["preferred_action"])
        counts[memory_key(preference, preferred, recent_history=recent_history, label="positive")] += 1
        for action in negative_actions(preference, negative_scope=negative_scope):
            counts[memory_key(preference, action, recent_history=recent_history, label="negative")] += 1

    entries = dict(sorted(counts.items()))
    return {
        "model_type": "recap_exact_memory",
        "entries": entries,
        "hyperparameters": {
            "negative_scope": negative_scope,
            "recent_history": recent_history,
        },
        "summary": {
            "train_preferences": len(train_records),
            "entries": len(entries),
            "positive_entries": sum(1 for key in entries if key.startswith("positive|")),
            "negative_entries": sum(1 for key in entries if key.startswith("negative|")),
        },
    }


def score_candidate(
    preference: Mapping[str, Any],
    action: str,
    model: Mapping[str, Any],
    recent_history: int | None = None,
) -> float:
    hyperparameters = dict(model.get("hyperparameters", {}))
    history_window = int(recent_history or hyperparameters.get("recent_history", 8))
    entries = dict(model.get("entries", {}))
    positive = entries.get(memory_key(preference, action, history_window, "positive"), 0)
    negative = entries.get(memory_key(preference, action, history_window, "negative"), 0)
    return float(positive) - float(negative)


def rank_candidates(
    preference: Mapping[str, Any],
    model: Mapping[str, Any],
    recent_history: int | None = None,
) -> tuple[tuple[str, float], ...]:
    candidates = tuple(str(candidate) for candidate in preference.get("candidates", ()))
    scored = [
        (index, action, score_candidate(preference, action, model, recent_history=recent_history))
        for index, action in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item[2], item[0]))
    return tuple((action, score) for _index, action, score in scored)


def memory_key(
    preference: Mapping[str, Any],
    action: str,
    recent_history: int = 8,
    label: str = "positive",
) -> str:
    history = tuple(normalize_action(str(item)) for item in preference.get("history", ())[-recent_history:])
    return f"{label}|{' <A> '.join(history)}|{normalize_action(action)}"
