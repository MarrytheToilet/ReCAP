from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_trace_feedback(
    path: Path,
    mode: str = "safe",
    max_items: int = 6,
    source: str = "all",
) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feedback: dict[str, str] = {}
    for report in payload.get("reports", ()):
        if not include_report(report, source):
            continue
        text = feedback_from_report(report, mode=mode, max_items=max_items)
        if not text:
            continue
        task_id = str(report["task_id"])
        feedback[task_id] = text
        feedback[Path(task_id).stem] = text
    return feedback


def include_report(report: Mapping[str, Any], source: str) -> bool:
    if source == "all":
        return True
    success = bool(report.get("original_success", False))
    if source == "failures":
        return not success
    if source == "successes":
        return success
    raise ValueError(f"unknown feedback source: {source}")


def feedback_from_report(
    report: Mapping[str, Any],
    mode: str = "safe",
    max_items: int = 6,
) -> str:
    if mode not in {"safe", "oracle"}:
        raise ValueError(f"unknown feedback mode: {mode}")

    lines: list[str] = []
    redundant = redundant_actions(report)
    if redundant:
        lines.append(
            "Replay-verified redundant actions from a prior run: "
            + "; ".join(redundant[:max_items])
            + "."
        )

    order_critical = order_critical_pairs(report)
    if order_critical:
        lines.append(
            "Replay-verified order-critical pairs: "
            + "; ".join(order_critical[:max_items])
            + "."
        )

    replacement_repairs = replacement_repairs_from_report(report)
    if replacement_repairs:
        lines.append(
            "Replay-verified replacement repairs: "
            + "; ".join(replacement_repairs[:max_items])
            + "."
        )

    lines.extend(failed_trace_diagnostics(report, max_items=max_items))

    policy_repairs = policy_repairs_from_report(report, include_suffix=mode == "oracle")
    if policy_repairs:
        lines.append(
            "Replay-verified prefix repair hints: "
            + "; ".join(policy_repairs[:max_items])
            + "."
        )

    if not lines:
        return ""

    return (
        "Verified feedback from counterfactual environment replay:\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\nUse this feedback only when it applies to the current state and admissible actions."
    )


def redundant_actions(report: Mapping[str, Any]) -> list[str]:
    actions: list[str] = []
    for summary in report.get("action_summaries", ()):
        if "redundant" in summary.get("labels", ()):
            actions.append(f"{summary['index']}: {summary['action']}")
    return actions


def order_critical_pairs(report: Mapping[str, Any]) -> list[str]:
    pairs: list[str] = []
    for edit in report.get("edits", ()):
        if edit.get("edit_type") != "swap_adjacent":
            continue
        if "order_critical" not in edit.get("labels", ()):
            continue
        second = edit.get("second_action")
        if second is None:
            continue
        pairs.append(f"{edit['action']} before {second}")
    return unique(pairs)


def replacement_repairs_from_report(report: Mapping[str, Any]) -> list[str]:
    repairs: list[str] = []
    for edit in report.get("edits", ()):
        if edit.get("edit_type") != "replace":
            continue
        if "repair_candidate" not in edit.get("labels", ()):
            continue
        replacement = edit.get("replacement")
        if replacement:
            repairs.append(f"replace {edit['action']} with {replacement}")
    return unique(repairs)


def failed_trace_diagnostics(report: Mapping[str, Any], max_items: int) -> list[str]:
    if bool(report.get("original_success", False)):
        return []

    actions = [str(action) for action in report.get("original_actions", ())]
    if not actions:
        return []

    diagnostics = [
        "Prior failed action sequence: "
        + "; ".join(format_indexed_actions(actions[:max_items]))
        + ("; ..." if len(actions) > max_items else ".")
    ]

    task_actions = task_changing_actions(actions)
    if task_actions:
        diagnostics.append(
            "Task-changing actions attempted before failure: "
            + "; ".join(task_actions[:max_items])
            + "; do not assume these completed the objective."
        )

    reversals = navigation_reversals(actions)
    if reversals:
        diagnostics.append(
            "Prior failure contained navigation reversals: "
            + "; ".join(reversals[:max_items])
            + "."
        )

    repeats = recent_repeats(actions)
    if repeats:
        diagnostics.append(
            "Prior failure repeated recent actions: "
            + "; ".join(repeats[:max_items])
            + "."
        )

    return diagnostics


def format_indexed_actions(actions: list[str]) -> list[str]:
    return [f"{index}: {action}" for index, action in enumerate(actions)]


def task_changing_actions(actions: list[str]) -> list[str]:
    ignored_prefixes = ("go ", "look", "inventory", "examine ")
    return unique(
        [
            action
            for action in actions
            if not any(action == prefix.strip() or action.startswith(prefix) for prefix in ignored_prefixes)
        ]
    )


def navigation_reversals(actions: list[str]) -> list[str]:
    inverses = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "up": "down",
        "down": "up",
    }
    reversals: list[str] = []
    previous_direction: str | None = None
    previous_action: str | None = None
    previous_index: int | None = None
    for index, action in enumerate(actions):
        direction = go_direction(action)
        if direction is None:
            previous_direction = None
            previous_action = None
            previous_index = None
            continue
        if (
            previous_direction is not None
            and previous_action is not None
            and previous_index is not None
            and inverses.get(previous_direction) == direction
        ):
            reversals.append(f"{previous_index}: {previous_action} -> {index}: {action}")
        previous_direction = direction
        previous_action = action
        previous_index = index
    return unique(reversals)


def go_direction(action: str) -> str | None:
    if not action.startswith("go "):
        return None
    return action[3:].strip()


def recent_repeats(actions: list[str], window: int = 4) -> list[str]:
    repeats: list[str] = []
    for index, action in enumerate(actions):
        start = max(0, index - window)
        for previous_index in range(start, index):
            if actions[previous_index] == action:
                repeats.append(f"{previous_index}: {action} repeated at {index}")
                break
    return unique(repeats)


def policy_repairs_from_report(
    report: Mapping[str, Any],
    include_suffix: bool,
) -> list[str]:
    repairs: list[str] = []
    for edit in report.get("edits", ()):
        if edit.get("edit_type") != "policy_suffix":
            continue
        if "policy_repair_candidate" not in edit.get("labels", ()):
            continue
        prefix = (
            "from the start"
            if int(edit.get("index", 0)) < 0
            else f"after action {edit.get('index')}: {edit.get('action')}"
        )
        if include_suffix:
            suffix = " ; ".join(str(item) for item in edit.get("repair_suffix", ()))
            repairs.append(f"{prefix}, verified completion is: {suffix}")
        else:
            repairs.append(
                f"{prefix}, the prior failure was still repairable; refocus on the task objective"
            )
    return unique(repairs)


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
