from __future__ import annotations

from recap.models.select_intervention_gate_threshold import selection_key, threshold_grid


def test_threshold_grid_includes_rounded_stop() -> None:
    assert threshold_grid(0.5, 0.53, 0.01) == [0.5, 0.51, 0.52, 0.53]


def test_selection_key_coverage_prefers_intervention_on_tie() -> None:
    safer = {
        "valid_overall_top1_correction": 0.25,
        "retention_intervention_rate": 0.10,
        "retention_demotion_rate": 0.001,
    }
    practical = {
        "valid_overall_top1_correction": 0.25,
        "retention_intervention_rate": 0.30,
        "retention_demotion_rate": 0.02,
    }

    assert selection_key(practical, "coverage") > selection_key(safer, "coverage")
    assert selection_key(safer, "safety") > selection_key(practical, "safety")
