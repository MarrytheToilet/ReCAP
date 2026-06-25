from __future__ import annotations

from recap.eval.eval_repair_multiplicity import summarize_rows


def test_repair_multiplicity_summary_separates_logged_and_alternative_ties() -> None:
    rows = [
        {
            "trajectory_id": "t1",
            "candidate_count": 3,
            "has_successful_logged_candidate": True,
            "has_successful_alternative": True,
            "has_multiple_successful_logged_candidates": True,
            "has_multiple_successful_alternatives": False,
            "tie_break_disagreement": True,
            "successful_logged_candidates": 2,
            "successful_alternatives": 1,
            "executed_action": "look",
            "successful_actions": ["look", "open door"],
            "successful_suffix_lens": [3, 3],
        },
        {
            "trajectory_id": "t1",
            "candidate_count": 3,
            "has_successful_logged_candidate": True,
            "has_successful_alternative": True,
            "has_multiple_successful_logged_candidates": True,
            "has_multiple_successful_alternatives": True,
            "tie_break_disagreement": False,
            "successful_logged_candidates": 3,
            "successful_alternatives": 2,
            "executed_action": "look",
            "successful_actions": ["look", "go west", "open door"],
            "successful_suffix_lens": [5, 4, 4],
        },
        {
            "trajectory_id": "t2",
            "candidate_count": 2,
            "has_successful_logged_candidate": False,
            "has_successful_alternative": False,
            "has_multiple_successful_logged_candidates": False,
            "has_multiple_successful_alternatives": False,
            "tie_break_disagreement": False,
            "successful_logged_candidates": 0,
            "successful_alternatives": 0,
            "executed_action": "look",
            "successful_actions": [],
            "successful_suffix_lens": [],
        },
    ]

    summary = summarize_rows(rows)

    assert summary["trajectories"] == 2
    assert summary["candidate_steps"] == 3
    assert summary["steps_with_successful_logged_candidate"] == 2
    assert summary["steps_with_multiple_successful_logged_candidates"] == 2
    assert summary["steps_with_multiple_successful_alternatives"] == 1
    assert summary["tie_break_disagreement_rate_over_multi_success_steps"] == 0.5
    assert summary["steps_with_multiple_min_suffix_logged_candidates"] == 2
    assert summary["steps_with_multiple_min_suffix_alternatives"] == 1
