from __future__ import annotations

from acta.agents import AgentContext, CandidateAction
from acta.agents.feedback_agent import TraceFeedbackAgent
from acta.probe.trace_feedback import feedback_from_report, include_report


class RecordingAgent:
    def __init__(self) -> None:
        self.initial_observation = ""

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        self.initial_observation = context.initial_observation
        return (CandidateAction("look"),)


def make_report() -> dict[str, object]:
    return {
        "task_id": "game.z8",
        "action_summaries": [
            {"index": 0, "action": "look", "labels": ["redundant"]},
        ],
        "edits": [
            {
                "edit_type": "swap_adjacent",
                "labels": ["order_critical"],
                "action": "open box",
                "second_action": "take coin",
            },
            {
                "edit_type": "replace",
                "labels": ["repair_candidate"],
                "action": "look",
                "replacement": "open box",
            },
            {
                "edit_type": "policy_suffix",
                "labels": ["policy_repair_candidate"],
                "index": -1,
                "action": "<start>",
                "repair_suffix": ["go north", "open box"],
            },
        ],
    }


def test_safe_feedback_omits_oracle_policy_suffix() -> None:
    feedback = feedback_from_report(make_report(), mode="safe")

    assert "look" in feedback
    assert "replace look with open box" in feedback
    assert "go north ; open box" not in feedback
    assert "refocus on the task objective" in feedback


def test_oracle_feedback_includes_policy_suffix() -> None:
    feedback = feedback_from_report(make_report(), mode="oracle")

    assert "go north ; open box" in feedback


def test_safe_feedback_includes_failed_trace_diagnostics() -> None:
    report = make_report()
    report["original_success"] = False
    report["original_actions"] = [
        "go north",
        "go south",
        "take coin",
        "go north",
        "go south",
        "go north",
    ]

    feedback = feedback_from_report(report, mode="safe", max_items=4)

    assert "Prior failed action sequence" in feedback
    assert "Task-changing actions attempted before failure: take coin" in feedback
    assert "Prior failure contained navigation reversals" in feedback
    assert "Prior failure repeated recent actions" in feedback
    assert "go north ; open box" not in feedback


def test_trace_feedback_agent_injects_feedback_by_stem() -> None:
    base = RecordingAgent()
    agent = TraceFeedbackAgent(base, {"game": "verified note"})
    context = AgentContext(
        task_id="/tmp/game.z8",
        seed=0,
        step_index=0,
        observation="obs",
        admissible_actions=("look",),
        history=(),
        state_signature=(),
        seen_signatures=(),
        initial_observation="initial",
    )

    candidates = agent.candidates(context)

    assert candidates[0].action == "look"
    assert "initial" in base.initial_observation
    assert "verified note" in base.initial_observation


def test_feedback_source_filter() -> None:
    assert include_report({"original_success": False}, "failures")
    assert not include_report({"original_success": True}, "failures")
    assert include_report({"original_success": True}, "successes")
    assert include_report({"original_success": True}, "all")
