from __future__ import annotations

import json

from acta.eval.merge_trajectory_jsonl import merge_trajectories


def test_merge_trajectories_deduplicates_and_skips_errors(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    trajectory = {
        "trajectory_id": "game1:seed0:rollout0",
        "task_id": "game1.z8",
        "game_seed": 1,
        "seed": 0,
        "success": False,
        "steps": [{"action": "look"}],
    }
    duplicate = dict(trajectory)
    error = {
        "trajectory_id": "game2:seed0:rollout0",
        "task_id": "game2.z8",
        "game_seed": 2,
        "seed": 0,
        "success": False,
        "steps": [],
        "error": "RateLimitError",
    }
    first.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    second.write_text(json.dumps(duplicate) + "\n" + json.dumps(error) + "\n", encoding="utf-8")

    records, summary = merge_trajectories((first, second))

    assert records == [trajectory]
    assert summary["written_trajectories"] == 1
    assert summary["failed_trajectories"] == 1
    assert summary["duplicate_records"] == 1
    assert summary["skipped_error_records"] == 1
