from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from recap.agents.base import Agent, AgentContext, CandidateAction


@dataclass(frozen=True)
class ActionPreference:
    task_id: str
    seed: int
    history: tuple[str, ...]
    preferred_action: str
    rejected_action: str
    source: str = "preference"


class PreferenceRerankAgent:
    """Rerank candidate actions with compiled action preferences."""

    def __init__(
        self,
        base_agent: Agent,
        preferences: tuple[ActionPreference, ...],
        bonus: float = 100.0,
    ) -> None:
        self.base_agent = base_agent
        self.preferences = preferences
        self.bonus = bonus
        self._by_key = index_preferences(preferences)

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        candidates = list(self.base_agent.candidates(context))
        if not candidates:
            return ()

        preferences = self._preferences_for_context(context)
        if not preferences:
            return tuple(candidates)

        preferred_sources: dict[str, list[str]] = {}
        rejected_actions: set[str] = set()
        for preference in preferences:
            preferred_sources.setdefault(preference.preferred_action, []).append(preference.source)
            rejected_actions.add(preference.rejected_action)

        reranked: list[CandidateAction] = []
        for candidate in candidates:
            metadata = dict(candidate.metadata)
            score = candidate.score
            if candidate.action in preferred_sources:
                score += self.bonus
                metadata["preference_sources"] = tuple(preferred_sources[candidate.action])
            if candidate.action in rejected_actions:
                score -= self.bonus
                metadata["preference_rejected"] = True
            reranked.append(
                CandidateAction(
                    action=candidate.action,
                    score=score,
                    source=candidate.source,
                    metadata=metadata,
                )
            )

        reranked.sort(key=lambda candidate: (-candidate.score, candidate.action))
        return tuple(reranked)

    def _preferences_for_context(self, context: AgentContext) -> tuple[ActionPreference, ...]:
        history = tuple(context.history)
        keys = [
            (context.task_id, context.seed, history),
            (Path(context.task_id).stem, context.seed, history),
        ]
        preferences: list[ActionPreference] = []
        for key in keys:
            preferences.extend(self._by_key.get(key, ()))
        return tuple(preferences)


def load_action_preferences(path: Path) -> tuple[ActionPreference, ...]:
    preferences: list[ActionPreference] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        preferences.append(preference_from_mapping(payload))
    return tuple(preferences)


def preference_from_mapping(payload: Mapping[str, object]) -> ActionPreference:
    return ActionPreference(
        task_id=str(payload["task_id"]),
        seed=int(payload.get("seed", 0)),
        history=tuple(str(action) for action in payload.get("history", ())),
        preferred_action=str(payload["preferred_action"]),
        rejected_action=str(payload["rejected_action"]),
        source=str(payload.get("source", "preference")),
    )


def index_preferences(
    preferences: tuple[ActionPreference, ...],
) -> dict[tuple[str, int, tuple[str, ...]], tuple[ActionPreference, ...]]:
    indexed: dict[tuple[str, int, tuple[str, ...]], list[ActionPreference]] = {}
    for preference in preferences:
        keys = [
            (preference.task_id, preference.seed, preference.history),
            (Path(preference.task_id).stem, preference.seed, preference.history),
        ]
        for key in keys:
            indexed.setdefault(key, []).append(preference)
    return {key: tuple(value) for key, value in indexed.items()}
