from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Hashable, Sequence

from recap.envs.base import EnvAdapter, StepResult


@dataclass(frozen=True)
class ToyState:
    """Small deterministic text environment for probing tests."""

    location: str = "foyer"
    door_open: bool = False
    lamp_on: bool = False
    note_read: bool = False
    inventory: frozenset[str] = frozenset()
    apple_location: str = "kitchen"


class ToyAdapter(EnvAdapter):
    """Deterministic adapter with enough structure to test relation probing.

    The adapter is deliberately small: it is not a benchmark, only a local
    oracle for replay, signatures, idempotence, commutation, and dependency.
    """

    def __init__(self) -> None:
        self._state = ToyState()

    def reset(self, task_id: str, seed: int = 0) -> StepResult:
        self._state = ToyState()
        return StepResult(
            state=self._state,
            observation=self._describe(self._state),
            info={"task_id": task_id, "seed": seed},
        )

    def step(self, action: str) -> StepResult:
        state = self._state
        new_state = state
        valid = True

        if action == "look":
            new_state = state
        elif action == "open door":
            new_state = replace(state, door_open=True)
        elif action == "close door":
            new_state = replace(state, door_open=False)
        elif action == "toggle lamp":
            new_state = replace(state, lamp_on=not state.lamp_on)
        elif action == "read note":
            new_state = replace(state, note_read=True)
        elif action == "go kitchen":
            if state.location == "foyer" and state.door_open:
                new_state = replace(state, location="kitchen")
            else:
                valid = False
        elif action == "go foyer":
            if state.location == "kitchen":
                new_state = replace(state, location="foyer")
            else:
                valid = False
        elif action == "take apple":
            if state.location == "kitchen" and state.apple_location == "kitchen":
                new_state = replace(
                    state,
                    apple_location="inventory",
                    inventory=state.inventory | frozenset({"apple"}),
                )
            else:
                valid = False
        elif action == "put apple in fridge":
            if "apple" in state.inventory:
                new_state = replace(
                    state,
                    apple_location="fridge",
                    inventory=state.inventory - frozenset({"apple"}),
                )
            else:
                valid = False
        elif action == "put apple in sink":
            if "apple" in state.inventory:
                new_state = replace(
                    state,
                    apple_location="sink",
                    inventory=state.inventory - frozenset({"apple"}),
                )
            else:
                valid = False
        else:
            valid = False

        self._state = new_state
        return StepResult(
            state=new_state,
            observation=self._describe(new_state),
            valid=valid,
            info={"action": action},
        )

    def admissible_actions(self, state: Any) -> Sequence[str]:
        state = self._coerce_state(state)
        actions = ["look", "open door", "close door", "toggle lamp", "read note"]
        if state.location == "foyer" and state.door_open:
            actions.append("go kitchen")
        if state.location == "kitchen":
            actions.append("go foyer")
        if state.location == "kitchen" and state.apple_location == "kitchen":
            actions.append("take apple")
        if "apple" in state.inventory:
            actions.extend(["put apple in fridge", "put apple in sink"])
        return actions

    def signature(self, state: Any, mode: str = "full") -> Hashable:
        state = self._coerce_state(state)
        if mode == "goal":
            return (
                ("location", state.location),
                ("inventory", tuple(sorted(state.inventory))),
                ("apple_location", state.apple_location),
            )
        if mode == "observable":
            return (
                ("location", state.location),
                ("door_open", state.door_open),
                ("inventory", tuple(sorted(state.inventory))),
                ("apple_location", state.apple_location),
            )
        if mode != "full":
            raise ValueError(f"unknown signature mode: {mode}")
        return (
            ("location", state.location),
            ("door_open", state.door_open),
            ("lamp_on", state.lamp_on),
            ("note_read", state.note_read),
            ("inventory", tuple(sorted(state.inventory))),
            ("apple_location", state.apple_location),
        )

    def _coerce_state(self, state: Any) -> ToyState:
        if not isinstance(state, ToyState):
            raise TypeError(f"expected ToyState, got {type(state)!r}")
        return state

    def _describe(self, state: ToyState) -> str:
        inventory = ", ".join(sorted(state.inventory)) or "empty"
        return (
            f"You are in the {state.location}. "
            f"The door is {'open' if state.door_open else 'closed'}. "
            f"The lamp is {'on' if state.lamp_on else 'off'}. "
            f"Inventory: {inventory}. "
            f"Apple: {state.apple_location}."
        )

