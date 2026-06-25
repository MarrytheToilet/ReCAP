from __future__ import annotations

from acta.eval.eval_recap_failure_decomposition import summarize_decomposition


def test_summarizes_failed_trajectory_decomposition() -> None:
    ledger = (
        {
            "trajectory_id": "ok",
            "original_outcome": "success",
            "status": "no_repair_found",
            "candidate_count": 1,
            "has_logged_candidates": True,
        },
        {
            "trajectory_id": "ok",
            "original_outcome": "success",
            "status": "no_repair_found",
            "candidate_count": 1,
            "has_logged_candidates": True,
        },
        {
            "trajectory_id": "fail",
            "original_outcome": "fail",
            "status": "certified_preference",
            "candidate_count": 1,
            "has_logged_candidates": True,
        },
        {
            "trajectory_id": "fail",
            "original_outcome": "fail",
            "status": "certified_preference",
            "candidate_count": 1,
            "has_logged_candidates": True,
        },
        {
            "trajectory_id": "fail",
            "original_outcome": "fail",
            "status": "repair_not_in_candidates",
            "candidate_count": 1,
            "has_logged_candidates": True,
        },
        {
            "trajectory_id": "fail",
            "original_outcome": "fail",
            "status": "no_repair_found",
            "candidate_count": 1,
            "has_logged_candidates": True,
        },
    )
    preferences = (
        {"preferred_rank_before": 2},
        {"preferred_rank_before": 4},
    )

    summary = summarize_decomposition(ledger, preferences)

    assert summary["trajectories"] == 2
    assert summary["failed_trajectories"] == 1
    assert summary["summarized_trajectories"] == 1
    assert summary["denominators"] == {
        "trajectory_level": 1,
        "candidate_step_level": 4,
        "certified_preference_level": 2,
    }
    assert summary["certified_preferences"] == 2
    assert summary["candidate_absent"] == 1
    assert summary["certified_misranking_step_rate"] == 0.5
    assert summary["candidate_absent_step_rate"] == 0.25
    assert summary["avg_preferred_rank_on_certified_preferences"] == 3.0


def test_summarizes_legacy_trajectory_ledger() -> None:
    ledger = (
        {
            "trajectory_id": "fail",
            "outcome": "fail",
            "step_statuses": [
                {"status": "certified_preference", "candidate_count": 1},
                {"status": "repair_same_as_executed", "candidate_count": 1},
            ],
        },
    )

    summary = summarize_decomposition(ledger)

    assert summary["failed_trajectories"] == 1
    assert summary["certified_preferences"] == 1
    assert summary["repair_same_as_executed"] == 1
    assert summary["repair_same_as_executed_step_rate"] == 0.5
