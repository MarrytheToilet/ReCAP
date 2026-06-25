from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

from recap.envs.base import EnvAdapter, StepResult


@dataclass(frozen=True)
class ALFWorldState:
    """Small state wrapper for ALFWorld text-mode gym observations."""

    observation: str
    infos: Mapping[str, Any]
    gamefile: str
    won: bool = False
    score: float | None = None
    max_score: float | None = None
    policy_commands: tuple[str, ...] = ()
    admissible_commands: tuple[str, ...] = ()


class ALFWorldTextAdapter(EnvAdapter):
    """ALFWorld text-mode adapter backed by TextWorld gym games.

    `task_id` should point to an ALFWorld `game.tw-pddl` file. This adapter
    keeps the same reset/replay contract as TextWorldAdapter while exposing the
    ALFWorld/ALFRED text tasks as a separate benchmark.
    """

    def __init__(self, max_episode_steps: int = 150) -> None:
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        except ImportError as exc:
            raise ImportError(
                "ALFWorldTextAdapter requires alfworld and textworld. Install with "
                "`python -m pip install alfworld`."
            ) from exc

        self._textworld = textworld
        self._tw_gym = textworld.gym
        self._demangler_cls = AlfredDemangler
        self._max_episode_steps = max_episode_steps
        self._env = None
        self._env_id: str | None = None
        self._game_file: str | None = None
        self._last_state: ALFWorldState | None = None

    def reset(self, task_id: str, seed: int = 0) -> StepResult:
        game_file = str(Path(task_id).expanduser())
        if self._env is None or self._game_file != game_file:
            self.close()
            infos = self._textworld.EnvInfos(
                won=True,
                admissible_commands=True,
                facts=True,
                policy_commands=True,
                score=True,
                max_score=True,
                intermediate_reward=True,
            )
            self._env_id = self._tw_gym.register_game(
                game_file,
                infos,
                max_episode_steps=self._max_episode_steps,
                wrappers=[self._demangler_cls()],
            )
            self._env = self._tw_gym.make(self._env_id)
            self._game_file = game_file
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        obs, infos = self._env.reset()
        state = self._state(obs, infos, game_file)
        self._last_state = state
        return StepResult(
            state=state,
            observation=state.observation,
            valid=True,
            done=state.won,
            info={"task_id": game_file, "seed": seed, **self._state_info(state)},
        )

    def step(self, action: str) -> StepResult:
        if self._env is None or self._last_state is None:
            raise RuntimeError("reset must be called before step")

        before = self._last_state
        admissible = set(before.admissible_commands)
        obs, reward, done, infos = self._env.step(action)
        state = self._state(obs, infos, before.gamefile)
        self._last_state = state
        valid = not admissible or action in admissible
        return StepResult(
            state=state,
            observation=state.observation,
            reward=float(reward),
            done=bool(done or state.won),
            valid=valid,
            info={"action": action, **self._state_info(state)},
        )

    def admissible_actions(self, state: Any) -> Sequence[str]:
        return tuple(getattr(state, "admissible_commands", ()) or ())

    def policy_commands(self, state: Any) -> Sequence[str]:
        return tuple(getattr(state, "policy_commands", ()) or ())

    def objective(self, state: Any) -> str | None:
        infos = getattr(state, "infos", {}) or {}
        for key in ("objective", "goal", "extra.goal", "extra.objective"):
            if key in infos and infos[key] is not None:
                return str(infos[key])
        return None

    def signature(self, state: Any, mode: str = "full") -> Hashable:
        infos = getattr(state, "infos", {}) or {}
        facts = infos.get("facts")
        if mode == "full" and facts:
            return tuple(sorted(map(str, facts)))
        observation = getattr(state, "observation", str(state))
        admissible = tuple(getattr(state, "admissible_commands", ()) or ())
        score = getattr(state, "score", None)
        if mode == "observable":
            return (("observation", observation),)
        if mode == "goal":
            return (("observation", observation), ("score", score))
        if mode == "full":
            return (("observation", observation), ("admissible", admissible), ("score", score))
        raise ValueError(f"unknown signature mode: {mode}")

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None
        self._env_id = None
        self._game_file = None
        self._last_state = None

    def _state(self, obs: Any, infos: Mapping[str, Any], gamefile: str) -> ALFWorldState:
        return ALFWorldState(
            observation=str(obs),
            infos=dict(infos),
            gamefile=gamefile,
            won=bool(infos.get("won", False)),
            score=_maybe_float(infos.get("score")),
            max_score=_maybe_float(infos.get("max_score")),
            policy_commands=tuple(str(a) for a in infos.get("policy_commands", ()) or ()),
            admissible_commands=tuple(str(a) for a in infos.get("admissible_commands", ()) or ()),
        )

    def _state_info(self, state: ALFWorldState) -> dict[str, Any]:
        return {
            "policy_commands": state.policy_commands,
            "objective": self.objective(state),
            "score": state.score,
            "max_score": state.max_score,
            "won": state.won,
            "admissible_commands": state.admissible_commands,
        }


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
