from __future__ import annotations

from acta.eval.compare_agent_runs import paired_summary


def episode(task_id: str, success: bool, repeat: float = 0.0, interventions: int = 0):
    return {
        "task_id": task_id,
        "success": success,
        "steps": [{}],
        "metrics": {
            "steps": 1.0,
            "repeated_action_rate": repeat,
            "reranker_intervention_rate": float(interventions),
            "reranker_gold_demotion_rate": 0.0,
            "reranker_interventions": float(interventions),
            "reranker_demoted_gold": 0.0,
        },
    }


def test_paired_summary_reports_rescue_and_harm_counts() -> None:
    baseline = {
        "episodes": [
            episode("a.z8", False),
            episode("b.z8", True),
            episode("c.z8", True),
        ]
    }
    treatment = {
        "episodes": [
            episode("a.z8", True, interventions=1),
            episode("b.z8", False, interventions=1),
            episode("c.z8", True),
        ]
    }

    output = paired_summary(baseline, treatment, iterations=10, seed=0)

    assert output["episodes"] == 3
    assert output["rescued_failures"] == 1
    assert output["harmed_successes"] == 1
    assert output["success_delta"] == 0.0
    assert output["treatment_intervention_rate"] == 2 / 3
