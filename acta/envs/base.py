from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence


@dataclass(frozen=True)
class StepResult:
    """Result of one environment action."""

    state: Any
    observation: str
    reward: float = 0.0
    done: bool = False
    valid: bool = True
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayResult:
    """Result of replaying a prefix from a deterministic reset."""

    state: Any
    observation: str
    steps: tuple[StepResult, ...]
    done: bool = False
    valid: bool = True
    info: Mapping[str, Any] = field(default_factory=dict)


class EnvAdapter(ABC):
    """Minimal interface needed for black-box action relation probing."""

    @abstractmethod
    def reset(self, task_id: str, seed: int = 0) -> StepResult:
        """Reset a task instance."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Apply one action to the current environment state."""

    def replay(
        self,
        task_id: str,
        prefix_actions: Sequence[str],
        seed: int = 0,
    ) -> ReplayResult:
        """Reset and replay an action prefix.

        Adapters can override this for faster checkpointed replay, but the
        default implementation is intentionally simple and deterministic.
        """

        first = self.reset(task_id=task_id, seed=seed)
        steps: list[StepResult] = []
        current = first
        valid = first.valid
        for action in prefix_actions:
            current = self.step(action)
            steps.append(current)
            valid = valid and current.valid
            if current.done:
                break
        return ReplayResult(
            state=current.state,
            observation=current.observation,
            steps=tuple(steps),
            done=current.done,
            valid=valid,
            info={"reset": first.info},
        )

    @abstractmethod
    def admissible_actions(self, state: Any) -> Sequence[str]:
        """Return actions known to be valid in a state when available."""

    @abstractmethod
    def signature(self, state: Any, mode: str = "full") -> Hashable:
        """Return a comparable state signature."""

    def is_equivalent(self, sig1: Hashable, sig2: Hashable, mode: str = "full") -> bool:
        """Compare two signatures under a selected equivalence mode."""

        return sig1 == sig2

    def policy_commands(self, state: Any) -> Sequence[str]:
        """Return oracle remaining commands when the environment exposes them."""

        return tuple(getattr(state, "policy_commands", ()) or ())

    def objective(self, state: Any) -> str | None:
        """Return task objective text when available."""

        objective = getattr(state, "objective", None)
        return str(objective) if objective is not None else None
