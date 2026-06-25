from __future__ import annotations

from typing import Any, Hashable, Sequence

from recap.envs.base import EnvAdapter, StepResult


class TextWorldAdapter(EnvAdapter):
    """Best-effort TextWorld adapter.

    `task_id` is expected to be a TextWorld game file path. TextWorld is an
    optional dependency; install `recap[textworld]` before using this adapter.
    """

    def __init__(self, request_admissible: bool = True) -> None:
        try:
            import textworld
            from textworld import EnvInfos
        except ImportError as exc:
            raise ImportError(
                "TextWorldAdapter requires TextWorld. Install with "
                "`python -m pip install '.[textworld]'` or provide another adapter."
            ) from exc

        self._textworld = textworld
        self._infos = EnvInfos(
            admissible_commands=request_admissible,
            description=True,
            inventory=True,
            facts=True,
            intermediate_reward=True,
            lost=True,
            max_score=True,
            objective=True,
            policy_commands=True,
            score=True,
            won=True,
        )
        self._env = None
        self._game_file: str | None = None
        self._last_state = None

    def reset(self, task_id: str, seed: int = 0) -> StepResult:
        if self._env is None or self._game_file != task_id:
            self.close()
            self._env = self._textworld.start(task_id, request_infos=self._infos)
            self._game_file = task_id
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        state = self._env.reset()
        self._last_state = state
        return StepResult(
            state=state,
            observation=getattr(state, "feedback", str(state)),
            valid=True,
            info={"task_id": task_id, "seed": seed, **self._state_info(state)},
        )

    def step(self, action: str) -> StepResult:
        if self._env is None or self._last_state is None:
            raise RuntimeError("reset must be called before step")

        before = self._last_state
        admissible = set(self.admissible_actions(before))
        state, reward, done = self._env.step(action)
        self._last_state = state
        valid = not admissible or action in admissible
        return StepResult(
            state=state,
            observation=getattr(state, "feedback", str(state)),
            reward=float(reward),
            done=bool(done),
            valid=valid,
            info={"action": action, **self._state_info(state)},
        )

    def admissible_actions(self, state: Any) -> Sequence[str]:
        return tuple(getattr(state, "admissible_commands", ()) or ())

    def signature(self, state: Any, mode: str = "full") -> Hashable:
        if mode == "full":
            facts = getattr(state, "facts", None)
            if facts is not None:
                return tuple(sorted(map(str, facts)))

        feedback = getattr(state, "feedback", "")
        inventory = getattr(state, "inventory", "")
        description = getattr(state, "description", "")

        if mode == "observable":
            return (("description", description), ("inventory", inventory), ("feedback", feedback))
        if mode == "goal":
            return (("inventory", inventory), ("description", description))
        if mode == "full":
            return (("description", description), ("inventory", inventory), ("feedback", feedback))
        raise ValueError(f"unknown signature mode: {mode}")

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None
        self._game_file = None
        self._last_state = None

    def _state_info(self, state: Any) -> dict[str, Any]:
        return {
            "policy_commands": tuple(getattr(state, "policy_commands", ()) or ()),
            "objective": getattr(state, "objective", None),
            "score": getattr(state, "score", None),
            "max_score": getattr(state, "max_score", None),
            "won": getattr(state, "won", None),
            "lost": getattr(state, "lost", None),
            "intermediate_reward": getattr(state, "intermediate_reward", None),
        }
