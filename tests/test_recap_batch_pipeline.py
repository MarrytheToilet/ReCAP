from __future__ import annotations

import json

from acta.eval.build_action_preferences import build_recap_dataset
from acta.eval.collect_failure_trajectories import default_game_dir, infer_seed, select_games
from acta.eval.compile_recap_batch import build_report_payload, load_episodes, load_reports
from acta.eval.eval_candidate_ranking import (
    evaluate_candidate_ranking,
    index_predictions,
    summarize_ranking,
)
from acta.eval.make_recap_splits import group_key, make_splits, split_summary


def test_compile_batch_loads_jsonl_episodes_and_reports(tmp_path) -> None:
    trajectories = tmp_path / "trajectories.jsonl"
    reports = tmp_path / "reports.jsonl"
    episode = {
        "task_id": "data/textworld_xhard_games/acta_xhard_seed203.z8",
        "seed": 0,
        "success": False,
        "steps": [
            {
                "step_index": 0,
                "action": "look",
                "candidates_before": ["look", "open door"],
            }
        ],
    }
    report = {
        "task_id": "data/textworld_xhard_games/acta_xhard_seed203.z8",
        "seed": 0,
        "original_actions": ["look"],
        "original_success": False,
        "edits": [
            {
                "edit_type": "policy_suffix",
                "index": -1,
                "labels": ["policy_repair_candidate"],
                "repair_suffix": ["open door"],
                "edited_actions": ["open door"],
                "success": True,
            }
        ],
    }
    trajectories.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    reports.write_text(json.dumps(report) + "\n", encoding="utf-8")

    preferences, step_ledger, trajectory_ledger = build_recap_dataset(
        run_payload={"episodes": load_episodes(trajectories)},
        report_payload={"reports": load_reports(reports)},
        source="policy-repair",
    )

    assert len(preferences) == 1
    assert preferences[0]["preferred_action"] == "open door"
    assert step_ledger[0]["status"] == "certified_preference"
    assert trajectory_ledger[0]["num_certified_preferences"] == 1


def test_compile_batch_does_not_probe_when_using_gold_source() -> None:
    assert (
        build_report_payload(
            episodes=[],
            trace_report=None,
            out_trace_report=None,
            source="gold",
        )
        is None
    )


def test_candidate_ranking_keeps_learned_metrics_null_without_predictions() -> None:
    preferences = (
        {
            "task_id": "game.z8",
            "seed": 0,
            "step_index": 0,
            "candidates": ["look", "open door"],
            "preferred_action": "open door",
            "rejected_action": "look",
        },
    )

    summary = summarize_ranking(evaluate_candidate_ranking(preferences))

    assert summary["raw_mrr"] == 0.5
    assert summary["oracle_recap_mrr"] == 1.0
    assert summary["learned_mrr"] is None
    assert summary["learned_top1_correction_rate"] is None
    assert summary["learned_abstain_rate"] is None


def test_candidate_ranking_uses_learned_predictions_when_provided() -> None:
    preferences = (
        {
            "task_id": "game.z8",
            "seed": 0,
            "step_index": 0,
            "candidates": ["look", "open door"],
            "preferred_action": "open door",
            "rejected_action": "look",
        },
    )
    predictions = index_predictions(
        (
            {
                "task_id": "game.z8",
                "seed": 0,
                "step_index": 0,
                "scores": {"look": 0.0, "open door": 2.0},
            },
        )
    )

    summary = summarize_ranking(evaluate_candidate_ranking(preferences, predictions))

    assert summary["learned_predictions"] == 1
    assert summary["learned_mrr"] == 1.0
    assert summary["learned_ndcg"] == 1.0
    assert summary["learned_top1_correction_rate"] == 1.0
    assert summary["learned_abstain_rate"] == 0.0


def test_make_recap_splits_prevents_seed_leakage() -> None:
    records = tuple(
        {
            "task_id": f"data/textworld_xhard_games/acta_xhard_seed{seed}.z8",
            "seed": 0,
            "step_index": 0,
            "preferred_action": "go west",
            "rejected_action": "look",
        }
        for seed in range(201, 207)
    )

    splits = make_splits(
        records=records,
        split_by="seed",
        train_frac=0.5,
        valid_frac=0.25,
        test_frac=0.25,
        seed=0,
    )
    summary = split_summary(splits, split_by="seed")

    assert summary["leakage"] is False
    assert sum(summary["records"].values()) == 6
    assert group_key(records[0], "seed") == "201"


def test_make_recap_splits_prevents_game_id_leakage() -> None:
    records = (
        {"task_id": "games/game_a.z8", "seed": 0, "step_index": 0},
        {"task_id": "games/game_a.z8", "seed": 1, "step_index": 0},
        {"task_id": "games/game_b.z8", "seed": 0, "step_index": 0},
        {"task_id": "games/game_c.z8", "seed": 0, "step_index": 0},
    )

    splits = make_splits(
        records=records,
        split_by="game_id",
        train_frac=0.5,
        valid_frac=0.25,
        test_frac=0.25,
        seed=1,
    )
    summary = split_summary(splits, split_by="game_id")

    assert summary["leakage"] is False
    assert summary["groups"]["train"] + summary["groups"]["valid"] + summary["groups"]["test"] == 3


def test_make_recap_splits_supports_task_id_groups() -> None:
    records = (
        {"task_id": "scienceworld://boil/0/easy", "seed": 0, "step_index": 0},
        {"task_id": "scienceworld://freeze/0/easy", "seed": 0, "step_index": 0},
    )

    splits = make_splits(
        records=records,
        split_by="task_id",
        train_frac=0.5,
        valid_frac=0.0,
        test_frac=0.5,
        seed=0,
    )
    summary = split_summary(splits, split_by="task_id")

    assert summary["leakage"] is False
    assert summary["groups"]["train"] == 1
    assert summary["groups"]["test"] == 1


def test_collect_failure_helpers_select_games_and_infer_seed(tmp_path) -> None:
    game_dir = tmp_path / "games"
    game_dir.mkdir()
    first = game_dir / "acta_xhard_seed203.z8"
    second = game_dir / "acta_xhard_seed204.z8"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    selected = select_games(game_dir=game_dir, game_glob="*.z8", num_games=1)

    assert selected == (first,)
    assert infer_seed(first) == 203
    assert default_game_dir("xhard").as_posix() == "data/textworld_xhard_games"
