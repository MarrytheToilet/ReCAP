from __future__ import annotations

from acta.eval.build_action_preferences import build_preferences, build_recap_dataset


def test_builds_policy_repair_preference_from_trace_report() -> None:
    run_payload = {
        "episodes": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "steps": [
                    {
                        "step_index": 0,
                        "action": "look",
                        "candidates_before": ["look", "open door"],
                        "gold_action": "open door",
                    },
                    {
                        "step_index": 1,
                        "action": "open door",
                        "candidates_before": ["open door", "inventory"],
                        "gold_action": "open door",
                    },
                ],
            }
        ]
    }
    report_payload = {
        "reports": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "edits": [
                    {
                        "edit_type": "policy_suffix",
                        "index": -1,
                        "labels": ["policy_repair_candidate"],
                        "repair_suffix": ["open door"],
                    }
                ],
            }
        ]
    }

    preferences = build_preferences(
        run_payload=run_payload,
        report_payload=report_payload,
        source="policy-repair",
    )

    assert len(preferences) == 1
    preference = preferences[0]
    assert preference["task_id"] == "game.z8"
    assert preference["seed"] == 0
    assert preference["step_index"] == 0
    assert preference["history"] == ()
    assert preference["candidates"] == ("look", "open door")
    assert preference["preferred_action"] == "open door"
    assert preference["rejected_action"] == "look"
    assert preference["executed_action"] == "look"
    assert preference["preferred_rank_before"] == 2
    assert preference["rejected_rank_before"] == 1
    assert preference["certificate_level"] == "C3_failure_repair"
    assert preference["repair_suffix_len"] == 1
    assert preference["original_outcome"] == "fail"
    assert preference["repaired_outcome"] == "success"
    assert preference["source"] == "policy_repair_suffix"


def test_gold_source_builds_preference_without_trace_report() -> None:
    run_payload = {
        "episodes": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "steps": [
                    {
                        "step_index": 0,
                        "action": "inventory",
                        "candidates_before": ["inventory", "take coin"],
                        "gold_action": "take coin",
                    }
                ],
            }
        ]
    }

    preferences = build_preferences(run_payload=run_payload, source="gold")

    assert len(preferences) == 1
    assert preferences[0]["preferred_action"] == "take coin"
    assert preferences[0]["rejected_action"] == "inventory"
    assert preferences[0]["certificate_level"] == "C0_candidate_only"


def test_skips_preference_when_preferred_action_is_not_a_candidate() -> None:
    run_payload = {
        "episodes": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "steps": [
                    {
                        "step_index": 0,
                        "action": "look",
                        "candidates_before": ["look", "inventory"],
                        "gold_action": "open door",
                    }
                ],
            }
        ]
    }

    assert build_preferences(run_payload=run_payload, source="gold") == []


def test_builds_recap_ledger_status_counts() -> None:
    run_payload = {
        "episodes": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "success": False,
                "steps": [
                    {
                        "step_index": 0,
                        "action": "look",
                        "candidates_before": ["look", "open door"],
                    },
                    {
                        "step_index": 1,
                        "action": "inventory",
                        "candidates_before": ["inventory", "look"],
                    },
                ],
            }
        ]
    }
    report_payload = {
        "reports": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "original_actions": ["look", "inventory"],
                "edits": [
                    {
                        "edit_type": "policy_suffix",
                        "index": -1,
                        "labels": ["policy_repair_candidate"],
                        "repair_suffix": ["open door"],
                        "edited_actions": ["open door"],
                        "success": True,
                    },
                    {
                        "edit_type": "policy_suffix",
                        "index": 0,
                        "labels": ["policy_repair_candidate"],
                        "repair_suffix": ["take coin"],
                        "edited_actions": ["look", "take coin"],
                        "success": True,
                    },
                ],
            }
        ]
    }

    preferences, step_ledger, trajectory_ledger = build_recap_dataset(
        run_payload=run_payload,
        report_payload=report_payload,
        source="policy-repair",
    )

    assert len(preferences) == 1
    assert len(step_ledger) == 2
    assert step_ledger[0]["status"] == "certified_preference"
    assert step_ledger[0]["repair_in_candidates"] is True
    assert step_ledger[0]["executed_action_norm"] == "look"
    assert step_ledger[0]["repair_action_norm"] == "open door"
    assert step_ledger[1]["status"] == "repair_not_in_candidates"
    assert step_ledger[1]["status_reason"] == "verified repair action absent from logged candidates"
    assert trajectory_ledger[0]["num_certified_preferences"] == 1
    assert trajectory_ledger[0]["num_candidate_absent"] == 1
    assert trajectory_ledger[0]["status_counts"] == {
        "certified_preference": 1,
        "repair_not_in_candidates": 1,
    }


def test_marks_steps_without_logged_candidates() -> None:
    run_payload = {
        "episodes": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "success": False,
                "steps": [
                    {
                        "step_index": 0,
                        "action": "  LOOK   ",
                        "candidates_before": [],
                    }
                ],
            }
        ]
    }
    report_payload = {
        "reports": [
            {
                "task_id": "game.z8",
                "seed": 0,
                "original_actions": ["look"],
                "edits": [
                    {
                        "edit_type": "policy_suffix",
                        "index": -1,
                        "labels": ["policy_repair_candidate"],
                        "repair_suffix": ["Open   Door"],
                        "edited_actions": ["Open Door"],
                        "success": True,
                    }
                ],
            }
        ]
    }

    preferences, step_ledger, trajectory_ledger = build_recap_dataset(
        run_payload=run_payload,
        report_payload=report_payload,
        source="policy-repair",
    )

    assert preferences == []
    assert step_ledger[0]["status"] == "no_candidates_logged"
    assert step_ledger[0]["executed_action_norm"] == "look"
    assert step_ledger[0]["repair_action_norm"] == "open door"
    assert trajectory_ledger[0]["num_no_candidates_logged"] == 1
