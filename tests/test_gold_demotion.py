from __future__ import annotations

from acta.eval.build_success_retention_dataset import build_retention_dataset
from acta.eval.eval_candidate_ranking import evaluate_candidate_ranking, index_predictions
from acta.eval.eval_gold_demotion import summarize_gold_demotion


def trajectory() -> dict[str, object]:
    return {
        "task_id": "game.z8",
        "trajectory_id": "game:seed0:rollout0",
        "success": True,
        "seed": 0,
        "rollout_seed": 0,
        "game_seed": 1,
        "steps": [
            {
                "step_index": 0,
                "action": "go east",
                "gold_action": "go east",
                "candidates_before": ["go east", "look"],
            },
            {
                "step_index": 1,
                "action": "look",
                "gold_action": "go west",
                "candidates_before": ["look", "go west"],
            },
            {
                "step_index": 2,
                "action": "take key",
                "gold_action": "take key",
                "candidates_before": ["take key"],
            },
        ],
    }


def test_build_success_retention_dataset_keeps_raw_top1_gold_steps() -> None:
    records, summary = build_retention_dataset((trajectory(),))

    assert summary["retention_records"] == 1
    assert summary["skipped"]["gold_not_raw_top1"] == 1
    assert summary["skipped"]["single_candidate"] == 1
    assert records[0]["preferred_action"] == "go east"
    assert records[0]["rejected_action"] == "look"
    assert records[0]["preferred_rank_before"] == 1
    assert records[0]["history"] == ()


def test_gold_demotion_summary_counts_abstention_as_retention() -> None:
    records, _summary = build_retention_dataset((trajectory(),))
    predicted_demote = {
        "task_id": "game.z8",
        "seed": 0,
        "step_index": 0,
        "ranked_actions": ["look", "go east"],
        "abstain": False,
    }
    ranked_records = evaluate_candidate_ranking(records, index_predictions((predicted_demote,)))
    summary = summarize_gold_demotion(ranked_records)

    assert summary["gold_demotions"] == 1
    assert summary["gold_demotion_rate_when_intervened"] == 1.0
    assert summary["gold_retention_rate_with_abstention"] == 0.0

    predicted_abstain = dict(predicted_demote)
    predicted_abstain["abstain"] = True
    ranked_records = evaluate_candidate_ranking(records, index_predictions((predicted_abstain,)))
    summary = summarize_gold_demotion(ranked_records)

    assert summary["gold_demotions"] == 0
    assert summary["abstain_rate"] == 1.0
    assert summary["gold_retention_rate_with_abstention"] == 1.0
