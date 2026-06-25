from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class CandidateAction:
    """One proposed action with an optional model score."""

    action: str
    score: float = 0.0
    source: str = "agent"
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentContext:
    """Information available to an agent at one decision point."""

    task_id: str
    seed: int
    step_index: int
    observation: str
    admissible_actions: tuple[str, ...]
    history: tuple[str, ...]
    state_signature: Hashable
    seen_signatures: tuple[Hashable, ...]
    initial_observation: str = ""


class Agent(Protocol):
    """Agent protocol. Agents propose candidates; controllers may rerank them."""

    def candidates(self, context: AgentContext) -> Sequence[CandidateAction]:
        """Return candidate actions for the current state."""
