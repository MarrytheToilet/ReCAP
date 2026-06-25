from __future__ import annotations

from dataclasses import replace

from recap.envs.base import StepResult
from recap.envs.toy_adapter import ToyAdapter
from recap.probe.trace_edit_probe import TraceEditConfig, TraceEditProbe


class KitchenGoalToyAdapter(ToyAdapter):
    def step(self, action: str) -> StepResult:
        step = super().step(action)
        done = getattr(step.state, "location", None) == "kitchen"
        if done:
            return replace(step, reward=1.0, done=True)
        return step

    def policy_commands(self, state: object) -> tuple[str, ...]:
        if getattr(state, "location", None) == "kitchen":
            return ()
        if getattr(state, "door_open", False):
            return ("go kitchen",)
        return ("open door", "go kitchen")


def make_step(action: str, candidates: list[str] | None = None) -> dict[str, object]:
    return {
        "action": action,
        "valid": True,
        "reward": 0.0,
        "done": False,
        "selected_score": 0.0,
        "candidates_before": candidates or [action],
        "candidates": candidates or [action],
    }


def test_trace_edit_probe_labels_redundant_necessary_and_order_critical() -> None:
    probe = TraceEditProbe(KitchenGoalToyAdapter())
    episode = {
        "task_id": "toy-default",
        "seed": 0,
        "success": True,
        "steps": [
            make_step("look", ["look", "open door"]),
            make_step("open door", ["open door", "look"]),
            make_step("go kitchen", ["go kitchen", "look"]),
        ],
    }

    report = probe.probe_episode(episode)
    labels_by_action = {
        summary.action: set(summary.labels)
        for summary in report.action_summaries
    }

    assert "redundant" in labels_by_action["look"]
    assert "necessary" in labels_by_action["open door"]
    assert "necessary" in labels_by_action["go kitchen"]
    assert any(
        edit.edit_type == "swap_adjacent"
        and edit.index == 1
        and "order_critical" in edit.labels
        for edit in report.edits
    )


def test_trace_edit_probe_finds_replacement_repair_for_failed_trace() -> None:
    probe = TraceEditProbe(
        KitchenGoalToyAdapter(),
        config=TraceEditConfig(probe_deletions=False, probe_swaps=False),
    )
    episode = {
        "task_id": "toy-default",
        "seed": 0,
        "success": False,
        "steps": [
            make_step("look", ["look", "open door"]),
            make_step("go kitchen", ["go kitchen"]),
        ],
    }

    report = probe.probe_episode(episode)
    repair_edits = [
        edit
        for edit in report.edits
        if edit.edit_type == "replace" and "repair_candidate" in edit.labels
    ]

    assert len(repair_edits) == 1
    assert repair_edits[0].index == 0
    assert repair_edits[0].replacement == "open door"
    assert "open door" in report.action_summaries[0].repair_replacements


def test_trace_edit_probe_finds_policy_suffix_repairs_for_failed_trace() -> None:
    probe = TraceEditProbe(KitchenGoalToyAdapter())
    episode = {
        "task_id": "toy-default",
        "seed": 0,
        "success": False,
        "steps": [
            make_step("look", ["look", "open door"]),
            make_step("go kitchen", ["go kitchen"]),
        ],
    }

    report = probe.probe_episode(episode)
    policy_repairs = [
        edit for edit in report.edits if "policy_repair_candidate" in edit.labels
    ]

    assert policy_repairs
    assert policy_repairs[0].repair_suffix == ("open door", "go kitchen")
    assert report.metrics["policy_repair_candidates"] >= 1.0
