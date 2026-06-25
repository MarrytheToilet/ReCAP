from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Hashable, Mapping, Sequence

from acta.envs.base import EnvAdapter, StepResult


@dataclass(frozen=True)
class ScienceWorldState:
    observation: str
    info: Mapping[str, Any]
    task_name: str
    variation: int
    simplification: str
    score: float
    won: bool
    history: tuple[str, ...]
    gold_plan: tuple[str, ...]
    admissible_commands: tuple[str, ...]
    policy_commands: tuple[str, ...]


class ScienceWorldAdapter(EnvAdapter):
    """ScienceWorld adapter with recorded-success suffix certificates.

    Task ids use `scienceworld://<task>/<variation>/<simplification>`, for
    example `scienceworld://boil/0/easy`. The adapter loads ScienceWorld with
    `generateGoldPath=True` and exposes the remaining recorded gold path as a
    weak suffix provider. ReCAP still verifies any emitted repair by replay.
    """

    def __init__(self, env_step_limit: int = 100) -> None:
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError as exc:
            raise ImportError(
                "ScienceWorldAdapter requires scienceworld. Install with "
                "`python -m pip install scienceworld`."
            ) from exc

        self._env_cls = ScienceWorldEnv
        self._env_step_limit = env_step_limit
        self._env = ScienceWorldEnv(envStepLimit=env_step_limit)
        self._loaded_task: tuple[str, int, str] | None = None
        self._last_state: ScienceWorldState | None = None

    def reset(self, task_id: str, seed: int = 0) -> StepResult:
        task_name, variation, simplification = parse_scienceworld_task_id(task_id)
        loaded = (task_name, variation, simplification)
        if self._loaded_task != loaded:
            self._env.load(
                task_name,
                variation,
                simplification,
                generateGoldPath=True,
            )
            self._loaded_task = loaded
        obs, info = self._env.reset()
        gold_plan = tuple(
            action
            for action in self._env.get_gold_action_sequence()
            if not str(action).startswith("ERROR:")
        )
        state = self._state(
            observation=obs,
            info=info,
            history=(),
            gold_plan=gold_plan,
        )
        self._last_state = state
        return StepResult(
            state=state,
            observation=state.observation,
            valid=True,
            done=state.won,
            info={"task_id": task_id, "seed": seed, **self._state_info(state)},
        )

    def step(self, action: str) -> StepResult:
        if self._last_state is None:
            raise RuntimeError("reset must be called before step")
        before = self._last_state
        admissible = set(before.admissible_commands)
        obs, reward, done, info = self._env.step(action)
        state = self._state(
            observation=obs,
            info=info,
            history=before.history + (action,),
            gold_plan=before.gold_plan,
        )
        self._last_state = state
        valid = not admissible or action in admissible
        return StepResult(
            state=state,
            observation=state.observation,
            reward=float(reward),
            done=bool(done and state.won),
            valid=valid,
            info={"action": action, **self._state_info(state)},
        )

    def admissible_actions(self, state: Any) -> Sequence[str]:
        return tuple(getattr(state, "admissible_commands", ()) or ())

    def policy_commands(self, state: Any) -> Sequence[str]:
        return tuple(getattr(state, "policy_commands", ()) or ())

    def objective(self, state: Any) -> str | None:
        info = getattr(state, "info", {}) or {}
        task_desc = info.get("taskDesc")
        return str(task_desc) if task_desc else None

    def signature(self, state: Any, mode: str = "full") -> Hashable:
        info = getattr(state, "info", {}) or {}
        look = str(info.get("look", getattr(state, "observation", "")))
        inv = str(info.get("inv", ""))
        score = getattr(state, "score", 0.0)
        if mode == "observable":
            return (("look", look), ("inv", inv))
        if mode == "goal":
            return (("look", look), ("inv", inv), ("score", score))
        if mode == "full":
            return (
                ("look", look),
                ("inv", inv),
                ("score", score),
                ("valid", tuple(getattr(state, "admissible_commands", ()) or ())),
            )
        raise ValueError(f"unknown signature mode: {mode}")

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._last_state = None

    def _state(
        self,
        observation: Any,
        info: Mapping[str, Any],
        history: tuple[str, ...],
        gold_plan: tuple[str, ...],
    ) -> ScienceWorldState:
        score = float(info.get("score", 0.0) or 0.0)
        won = score >= 100.0
        task_name = str(info.get("taskName", self._loaded_task[0] if self._loaded_task else ""))
        variation = int(info.get("variationIdx", self._loaded_task[1] if self._loaded_task else 0))
        simplification = str(
            info.get("simplificationStr", self._loaded_task[2] if self._loaded_task else "")
        )
        admissible = tuple(str(action) for action in info.get("valid", ()) or ())
        remaining = adapt_gold_suffix_to_admissible(
            remaining_gold_suffix(gold_plan, history),
            admissible,
        )
        return ScienceWorldState(
            observation=str(observation),
            info=dict(info),
            task_name=task_name,
            variation=variation,
            simplification=simplification,
            score=score,
            won=won,
            history=history,
            gold_plan=gold_plan,
            admissible_commands=admissible,
            policy_commands=remaining,
        )

    def _state_info(self, state: ScienceWorldState) -> dict[str, Any]:
        return {
            "policy_commands": state.policy_commands,
            "objective": self.objective(state),
            "score": state.score,
            "won": state.won,
            "task_name": state.task_name,
            "variation": state.variation,
            "simplification": state.simplification,
            "admissible_commands": state.admissible_commands,
        }


def parse_scienceworld_task_id(task_id: str) -> tuple[str, int, str]:
    value = task_id.strip()
    if value.startswith("scienceworld://"):
        value = value.removeprefix("scienceworld://")
        parts = value.split("/")
    else:
        parts = value.split(":")
    if len(parts) == 1:
        return parts[0], 0, "easy"
    if len(parts) == 2:
        return parts[0], int(parts[1]), "easy"
    return parts[0], int(parts[1]), parts[2]


def remaining_gold_suffix(
    gold_plan: tuple[str, ...],
    history: tuple[str, ...],
) -> tuple[str, ...]:
    """Return gold suffix after matching executed history as an ordered subsequence."""

    index = 0
    for action in history:
        if index < len(gold_plan) and action == gold_plan[index]:
            index += 1
    return gold_plan[index:]


def adapt_gold_suffix_to_admissible(
    suffix: tuple[str, ...],
    admissible: tuple[str, ...],
) -> tuple[str, ...]:
    if not suffix:
        return suffix
    admissible_set = set(admissible)
    adapted = list(suffix)

    # ScienceWorld's easy simplification exposes direct teleport actions while
    # gold paths may still use open-door/go pairs.
    first = adapted[0]
    second = adapted[1] if len(adapted) > 1 else ""
    room_from_open = _target(first, r"^open door to (.+)$")
    room_from_go = _target(first, r"^go to (.+)$")
    if room_from_open and second == f"go to {room_from_open}":
        teleport = f"teleport to {room_from_open}"
        if teleport in admissible_set:
            return (teleport, *tuple(adapted[2:]))
    if room_from_go:
        teleport = f"teleport to {room_from_go}"
        if teleport in admissible_set:
            return (teleport, *tuple(adapted[1:]))

    replacement = equivalent_admissible_action(first, admissible_set)
    if replacement is not None:
        adapted[0] = replacement
    return tuple(adapted)


def equivalent_admissible_action(action: str, admissible: set[str]) -> str | None:
    if action in admissible:
        return action
    thermometer_target = _target(action, r"^use thermometer in inventory on (.+)$")
    if thermometer_target:
        simplified = f"use thermometer on {thermometer_target}"
        if simplified in admissible:
            return simplified
    focus_target = _target(action, r"^focus on (.+) in inventory$")
    if focus_target:
        simplified = f"focus on {focus_target}"
        if simplified in admissible:
            return simplified
    examine_target = _target(action, r"^examine (.+)$")
    if examine_target:
        for prefix in ("look at", "look in"):
            simplified = f"{prefix} {examine_target}"
            if simplified in admissible:
                return simplified
    return None


def _target(action: str, pattern: str) -> str | None:
    match = re.match(pattern, action)
    return match.group(1) if match else None
