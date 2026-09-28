from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def compact_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    stripped = str(text).strip()
    if not stripped:
        return ""
    for marker in ("Here is your task", "Your task", "First off"):
        index = stripped.find(marker)
        if index >= 0:
            stripped = stripped[index:]
            break
    lines = []
    for line in stripped.splitlines():
        clean = line.strip()
        if not clean:
            continue
        symbol_chars = sum(1 for char in clean if char in "_$\\|/=-")
        if len(clean) >= 20 and symbol_chars / max(1, len(clean)) > 0.55:
            continue
        lines.append(clean)
    compact = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def context_text(
    preference: Mapping[str, Any],
    max_history: int = 8,
    max_observation_chars: int = 0,
) -> str:
    history = [str(item) for item in preference.get("history", ())][-max_history:]
    candidates = [str(item) for item in preference.get("candidates", ())]
    lines = ["Text-game decision state."]
    initial_observation = compact_text(
        str(preference.get("initial_observation", "")),
        max_observation_chars,
    )
    if initial_observation:
        lines.append("Objective/state excerpt:")
        lines.append(initial_observation)
    observation = compact_text(
        str(preference.get("observation", "")),
        max_observation_chars,
    )
    if observation:
        lines.append("Current observation excerpt:")
        lines.append(observation)
    if history:
        lines.append("Recent executed actions:")
        lines.extend(f"{index + 1}. {action}" for index, action in enumerate(history))
    else:
        lines.append("Recent executed actions: none.")
    if candidates:
        lines.append("Logged candidate actions in original rank order:")
        lines.extend(f"{index + 1}. {action}" for index, action in enumerate(candidates))
    raw_top = candidates[0] if candidates else ""
    if raw_top:
        lines.append(f"Raw top-1 action: {raw_top}")
    lines.append("Question: should the proposed action be ranked first now?")
    return "\n".join(lines)


def candidate_text(action: str) -> str:
    return f"Proposed action: {action}"


def candidate_pairs(
    preference: Mapping[str, Any],
    max_history: int = 8,
    max_observation_chars: int = 0,
) -> list[tuple[str, str]]:
    context = context_text(
        preference,
        max_history=max_history,
        max_observation_chars=max_observation_chars,
    )
    return [
        (context, candidate_text(str(action)))
        for action in preference.get("candidates", ())
    ]


def pointwise_training_rows(
    preferences: Sequence[Mapping[str, Any]],
    max_history: int = 8,
    max_observation_chars: int = 0,
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for preference in preferences:
        preferred = str(preference["preferred_action"])
        context = context_text(
            preference,
            max_history=max_history,
            max_observation_chars=max_observation_chars,
        )
        for action in preference.get("candidates", ()):
            action_str = str(action)
            rows.append(
                (
                    context,
                    candidate_text(action_str),
                    1.0 if action_str == preferred else 0.0,
                )
            )
    return rows


def rank_candidates(
    preference: Mapping[str, Any],
    model: Any,
    batch_size: int = 16,
    max_history: int = 8,
    max_observation_chars: int = 0,
) -> tuple[tuple[str, float], ...]:
    candidates = tuple(str(action) for action in preference.get("candidates", ()))
    if not candidates:
        return ()
    pairs = candidate_pairs(
        preference,
        max_history=max_history,
        max_observation_chars=max_observation_chars,
    )
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scores = [float(score) for score in raw_scores]
    indexed = list(enumerate(zip(candidates, scores)))
    indexed.sort(key=lambda item: (-item[1][1], item[0]))
    return tuple((action, score) for _index, (action, score) in indexed)
