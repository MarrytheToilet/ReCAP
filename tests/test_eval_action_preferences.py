from __future__ import annotations

from recap.agents.preference_agent import ActionPreference
from recap.eval.eval_action_preferences import evaluate_preferences, summarize_records


def test_evaluates_preference_against_recorded_candidates() -> None:
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
                    }
                ],
            }
        ]
    }
    preferences = (
        ActionPreference(
            task_id="game.z8",
            seed=0,
            history=(),
            preferred_action="open door",
            rejected_action="look",
            source="policy_repair_suffix",
        ),
    )

    records = evaluate_preferences(run_payload, preferences)
    summary = summarize_records(records)

    assert records[0]["preferred_rank_before"] == 2
    assert records[0]["rejected_rank_before"] == 1
    assert records[0]["selected_was_rejected"] is True
    assert records[0]["preferred_misranked"] is True
    assert records[0]["raw_mrr"] == 0.5
    assert records[0]["oracle_recap_mrr"] == 1.0
    assert summary["would_make_preferred_top1"] == 1
    assert summary["preferred_top1_before"] == 0
    assert summary["candidate_coverage_rate"] == 1.0
    assert summary["misranking_rate"] == 1.0
    assert summary["oracle_top1_repairable_rate"] == 1.0
    assert summary["raw_mrr"] == 0.5
    assert summary["oracle_recap_mrr"] == 1.0
    assert summary["learned_mrr"] is None
    assert summary["learned_top1_correction_rate"] is None
