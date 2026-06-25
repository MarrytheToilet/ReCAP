from __future__ import annotations

from recap.eval.bootstrap_recap_metrics import bootstrap_metrics


def test_bootstrap_metrics_reports_point_and_intervals() -> None:
    ledger = (
        {
            "trajectory_id": "t1",
            "task_id": "game1.z8",
            "seed": 0,
            "step_index": 0,
            "original_outcome": "fail",
            "status": "certified_preference",
            "has_logged_candidates": True,
            "candidate_count": 2,
            "repair_rank_before": 2,
        },
        {
            "trajectory_id": "t2",
            "task_id": "game2.z8",
            "seed": 0,
            "step_index": 0,
            "original_outcome": "fail",
            "status": "repair_not_in_candidates",
            "has_logged_candidates": True,
            "candidate_count": 2,
        },
    )
    preferences = (
        {
            "trajectory_id": "t1",
            "task_id": "game1.z8",
            "seed": 0,
            "step_index": 0,
            "candidates": ["look", "open door"],
            "preferred_action": "open door",
            "rejected_action": "look",
            "preferred_rank_before": 2,
        },
    )

    output = bootstrap_metrics(ledger=ledger, preferences=preferences, iterations=10, seed=0)

    assert output["summary"]["trajectory_groups"] == 2
    assert output["decomposition"]["certified_misranking_step_rate"]["point"] == 0.5
    assert output["ranking"]["raw_mrr"]["point"] == 0.5
    assert output["ranking"]["raw_mrr"]["ci_low"] is not None
    assert output["ranking"]["raw_mrr"]["ci_high"] is not None
