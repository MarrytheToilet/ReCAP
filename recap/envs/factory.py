from __future__ import annotations

from pathlib import Path

from recap.envs.base import EnvAdapter
from recap.envs.textworld_adapter import TextWorldAdapter


def build_adapter(env_name: str) -> EnvAdapter:
    if env_name == "textworld":
        return TextWorldAdapter()
    if env_name == "alfworld":
        from recap.envs.alfworld_adapter import ALFWorldTextAdapter

        return ALFWorldTextAdapter()
    if env_name == "scienceworld":
        from recap.envs.scienceworld_adapter import ScienceWorldAdapter

        return ScienceWorldAdapter()
    raise ValueError(f"unknown env: {env_name}")


def default_task_dir(env_name: str, difficulty: str) -> Path:
    if env_name == "textworld":
        if difficulty in {"easy", "default"}:
            return Path("data/textworld_games")
        return Path(f"data/textworld_{difficulty}_games")
    if env_name == "alfworld":
        return Path.home() / ".cache" / "alfworld"
    if env_name == "scienceworld":
        return Path("scienceworld://")
    raise ValueError(f"unknown env: {env_name}")


def default_task_glob(env_name: str) -> str:
    if env_name == "textworld":
        return "*.z8"
    if env_name == "alfworld":
        return "**/game.tw-pddl"
    if env_name == "scienceworld":
        return "*"
    raise ValueError(f"unknown env: {env_name}")
