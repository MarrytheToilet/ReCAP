from acta.eval.eval_recap_sensitivity import evaluate_sensitivity


def test_top_k_truncation_converts_lost_certified_repairs_to_absent() -> None:
    ledger = (
        {
            "trajectory_id": "t0",
            "task_id": "game0",
            "seed": 0,
            "status": "certified_preference",
            "original_outcome": "fail",
            "has_logged_candidates": True,
            "repair_rank_before": 2,
            "repair_suffix_len": 3,
        },
        {
            "trajectory_id": "t0",
            "task_id": "game0",
            "seed": 0,
            "status": "certified_preference",
            "original_outcome": "fail",
            "has_logged_candidates": True,
            "repair_rank_before": 5,
            "repair_suffix_len": 8,
        },
        {
            "trajectory_id": "t1",
            "task_id": "game1",
            "seed": 0,
            "status": "repair_not_in_candidates",
            "original_outcome": "fail",
            "has_logged_candidates": True,
        },
    )

    summary = evaluate_sensitivity(ledger, top_k_values=(3, 5), suffix_budgets=(5,))
    top3, top5 = summary["top_k_truncation"]

    assert summary["summary"]["failed_candidate_steps"] == 3
    assert top3["certified_preferences_retained"] == 1
    assert top3["certified_preferences_lost_vs_top5"] == 1
    assert top3["candidate_absent_step_rate"] == 2 / 3
    assert top5["certified_preferences_retained"] == 2


def test_suffix_budget_reports_retained_repairs_and_too_long_rate() -> None:
    ledger = (
        {
            "trajectory_id": "t0",
            "task_id": "game0",
            "seed": 0,
            "status": "certified_preference",
            "original_outcome": "fail",
            "has_logged_candidates": True,
            "repair_rank_before": 2,
            "repair_suffix_len": 3,
        },
        {
            "trajectory_id": "t1",
            "task_id": "game1",
            "seed": 0,
            "status": "certified_preference",
            "original_outcome": "fail",
            "has_logged_candidates": True,
            "repair_rank_before": 3,
            "repair_suffix_len": 7,
        },
    )

    summary = evaluate_sensitivity(ledger, top_k_values=(5,), suffix_budgets=(5,))
    cap5, full = summary["suffix_budget"]

    assert cap5["certified_preferences_retained"] == 1
    assert cap5["certified_preferences_lost_vs_full"] == 1
    assert cap5["suffix_too_long_step_rate"] == 1 / 2
    assert full["certified_preferences_retained"] == 2
