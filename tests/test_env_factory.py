from pathlib import Path

from recap.envs.factory import default_task_dir, default_task_glob


def test_textworld_defaults_match_existing_layout() -> None:
    assert default_task_dir("textworld", "xhard").as_posix() == "data/textworld_xhard_games"
    assert default_task_glob("textworld") == "*.z8"


def test_alfworld_defaults_use_cache_and_recursive_game_files() -> None:
    assert default_task_dir("alfworld", "eval").name == "alfworld"
    assert default_task_dir("alfworld", "eval").parent == Path.home() / ".cache"
    assert default_task_glob("alfworld") == "**/game.tw-pddl"
