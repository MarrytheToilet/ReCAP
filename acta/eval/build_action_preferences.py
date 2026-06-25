from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RepairAction:
    action: str
    source: str
    certificate_level: str
    repair_suffix_len: int = 0
    edited_action_len: int | None = None
    original_action_len: int | None = None
    repaired_outcome: str = "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build action preference data from agent traces and trace-edit reports."
    )
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--run-key", default=None)
    parser.add_argument("--trace-report", type=Path, default=None)
    parser.add_argument("--source", choices=["policy-repair", "gold", "both"], default="policy-repair")
    parser.add_argument("--ledger-out", type=Path, default=None)
    parser.add_argument("--trajectory-ledger-out", type=Path, default=None)
    parser.add_argument("--max-repair-suffix-len", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("analysis/action_preferences.jsonl"))
    args = parser.parse_args()

    run_payload = json.loads(args.run_json.read_text(encoding="utf-8"))
    if args.run_key is not None:
        run_payload = run_payload[args.run_key]

    report_payload = (
        json.loads(args.trace_report.read_text(encoding="utf-8"))
        if args.trace_report is not None
        else None
    )
    preferences, step_ledger, trajectory_ledger = build_recap_dataset(
        run_payload=run_payload,
        report_payload=report_payload,
        source=args.source,
        max_repair_suffix_len=args.max_repair_suffix_len,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in preferences),
        encoding="utf-8",
    )
    if args.ledger_out is not None:
        args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
        args.ledger_out.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in step_ledger),
            encoding="utf-8",
        )
    if args.trajectory_ledger_out is not None:
        args.trajectory_ledger_out.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory_ledger_out.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in trajectory_ledger),
            encoding="utf-8",
        )
    print(json.dumps(summarize_preferences(preferences), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")
    if args.ledger_out is not None:
        print(f"ledger={args.ledger_out}")
    if args.trajectory_ledger_out is not None:
        print(f"trajectory_ledger={args.trajectory_ledger_out}")


def build_preferences(
    run_payload: Mapping[str, Any],
    report_payload: Mapping[str, Any] | None = None,
    source: str = "policy-repair",
) -> list[dict[str, Any]]:
    preferences, _step_ledger, _trajectory_ledger = build_recap_dataset(
        run_payload=run_payload,
        report_payload=report_payload,
        source=source,
    )
    return preferences


def build_recap_dataset(
    run_payload: Mapping[str, Any],
    report_payload: Mapping[str, Any] | None = None,
    source: str = "policy-repair",
    max_repair_suffix_len: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reports = reports_by_episode(report_payload) if report_payload is not None else {}
    records: list[dict[str, Any]] = []
    step_ledger: list[dict[str, Any]] = []
    trajectory_ledger: list[dict[str, Any]] = []
    for episode in run_payload.get("episodes", ()):
        task_id = str(episode["task_id"])
        seed = int(episode.get("seed", 0))
        trajectory_id = str(episode.get("trajectory_id") or f"{Path(task_id).stem}:seed{seed}")
        report = reports.get((task_id, seed))
        repair_actions = repair_first_actions(report) if report is not None else {}
        original_outcome = "success" if bool(episode.get("success", False)) else "fail"
        steps = tuple(episode.get("steps", ()))
        trajectory_steps: list[dict[str, Any]] = []
        history: list[str] = []
        for raw_step in steps:
            step = dict(raw_step)
            step_index = int(step.get("step_index", len(history)))
            selected = str(step.get("action", ""))
            candidates = tuple(str(action) for action in step_candidates(step))
            preferred, preference_source, repair = preferred_action_for_step(
                step=step,
                repair_actions=repair_actions,
                source=source,
            )
            preferred_rank = rank(candidates, preferred) if preferred is not None else None
            rejected_rank = rank(candidates, selected)
            status = preference_status(
                preferred=preferred,
                selected=selected,
                preferred_rank=preferred_rank,
                repair=repair,
                candidate_count=len(candidates),
                max_repair_suffix_len=max_repair_suffix_len,
            )
            if (
                status == "certified_preference"
                and preferred is not None
                and preferred != selected
                and preferred_rank is not None
            ):
                records.append(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "step_index": step_index,
                        "history": tuple(history),
                        "candidates": candidates,
                        "candidate_count": len(candidates),
                        "preferred_action": preferred,
                        "preferred_action_norm": normalize_action(preferred),
                        "rejected_action": selected,
                        "rejected_action_norm": normalize_action(selected),
                        "executed_action": selected,
                        "executed_action_norm": normalize_action(selected),
                        "preferred_rank_before": preferred_rank,
                        "rejected_rank_before": rejected_rank,
                        "original_outcome": original_outcome,
                        "repaired_outcome": (
                            repair.repaired_outcome if repair is not None else "unknown"
                        ),
                        "certificate_level": (
                            repair.certificate_level if repair is not None else "C0_candidate_only"
                        ),
                        "repair_suffix_len": (
                            repair.repair_suffix_len if repair is not None else 0
                        ),
                        "edited_action_len": (
                            repair.edited_action_len if repair is not None else None
                        ),
                        "original_action_len": (
                            repair.original_action_len if repair is not None else None
                        ),
                        "source": preference_source,
                    }
                )
            step_record = step_ledger_record(
                trajectory_id=trajectory_id,
                task_id=task_id,
                seed=seed,
                original_outcome=original_outcome,
                step_index=step_index,
                selected=selected,
                preferred=preferred,
                preferred_rank=preferred_rank,
                rejected_rank=rejected_rank,
                candidate_count=len(candidates),
                status=status,
                repair=repair,
            )
            step_ledger.append(step_record)
            trajectory_steps.append(step_record)
            if selected:
                history.append(selected)
        trajectory_ledger.append(
            trajectory_ledger_record(
                trajectory_id=trajectory_id,
                task_id=task_id,
                seed=seed,
                outcome=original_outcome,
                steps=steps,
                step_records=trajectory_steps,
            )
        )
    return records, step_ledger, trajectory_ledger


def preference_status(
    preferred: str | None,
    selected: str,
    preferred_rank: int | None,
    repair: RepairAction | None,
    candidate_count: int,
    max_repair_suffix_len: int | None,
) -> str:
    if candidate_count == 0:
        return "no_candidates_logged"
    if preferred is None:
        return "no_repair_found"
    if repair is not None and repair.repaired_outcome == "fail":
        return "invalid_replay"
    if (
        repair is not None
        and max_repair_suffix_len is not None
        and repair.repair_suffix_len > max_repair_suffix_len
    ):
        return "suffix_too_long"
    if preferred == selected:
        return "repair_same_as_executed"
    if preferred_rank is None:
        return "repair_not_in_candidates"
    return "certified_preference"


def step_ledger_record(
    trajectory_id: str,
    task_id: str,
    seed: int,
    original_outcome: str,
    step_index: int,
    selected: str,
    preferred: str | None,
    preferred_rank: int | None,
    rejected_rank: int | None,
    candidate_count: int,
    status: str,
    repair: RepairAction | None,
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "seed": seed,
        "step_index": step_index,
        "status": status,
        "executed_action": selected,
        "executed_action_norm": normalize_action(selected),
        "repair_action": preferred,
        "repair_action_norm": normalize_action(preferred),
        "repair_in_candidates": preferred_rank is not None,
        "repair_rank_before": preferred_rank,
        "rejected_rank_before": rejected_rank,
        "candidate_count": candidate_count,
        "has_logged_candidates": candidate_count > 0,
        "certificate_level": repair.certificate_level if repair is not None else None,
        "repair_suffix_len": repair.repair_suffix_len if repair is not None else None,
        "original_outcome": original_outcome,
        "repaired_outcome": repair.repaired_outcome if repair is not None else None,
        "status_reason": status_reason(status),
    }


def status_reason(status: str) -> str | None:
    reasons = {
        "repair_not_in_candidates": "verified repair action absent from logged candidates",
        "repair_same_as_executed": "verified repair action is the executed action",
        "no_repair_found": "no certified repair action found for this step",
        "suffix_too_long": "verified repair suffix exceeds configured maximum length",
        "invalid_replay": "candidate repair did not replay to success",
        "no_candidates_logged": "no candidate actions were logged for this step",
    }
    return reasons.get(status)


def trajectory_ledger_record(
    trajectory_id: str,
    task_id: str,
    seed: int,
    outcome: str,
    steps: tuple[Mapping[str, Any], ...],
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(step["status"]) for step in step_records)
    return {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "seed": seed,
        "outcome": outcome,
        "num_steps": len(steps),
        "num_candidate_steps": sum(1 for step in step_records if step["candidate_count"] > 0),
        "num_certified_preferences": status_counts.get("certified_preference", 0),
        "num_candidate_absent": status_counts.get("repair_not_in_candidates", 0),
        "num_no_repair_found": status_counts.get("no_repair_found", 0),
        "num_repair_same_as_executed": status_counts.get("repair_same_as_executed", 0),
        "num_suffix_too_long": status_counts.get("suffix_too_long", 0),
        "num_invalid_replay": status_counts.get("invalid_replay", 0),
        "num_no_candidates_logged": status_counts.get("no_candidates_logged", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "step_statuses": step_records,
    }


def reports_by_episode(
    report_payload: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    reports: dict[tuple[str, int], Mapping[str, Any]] = {}
    for report in report_payload.get("reports", ()):
        task_id = str(report["task_id"])
        seed = int(report.get("seed", 0))
        reports[(task_id, seed)] = report
    return reports


def repair_first_actions(report: Mapping[str, Any]) -> dict[int, RepairAction]:
    repairs: dict[int, RepairAction] = {}
    original_actions = tuple(str(action) for action in report.get("original_actions", ()))
    original_success = bool(report.get("original_success", False))
    for edit in report.get("edits", ()):
        repair: RepairAction | None = None
        step_index: int | None = None
        labels = tuple(str(label) for label in edit.get("labels", ()))
        edit_type = str(edit.get("edit_type", ""))
        suffix = tuple(str(action) for action in edit.get("repair_suffix", ()))
        edited_actions = tuple(str(action) for action in edit.get("edited_actions", ()))
        if edit_type == "policy_suffix" and "policy_repair_candidate" in labels and suffix:
            step_index = int(edit.get("index", -1)) + 1
            repair = RepairAction(
                action=suffix[0],
                source="policy_repair_suffix",
                certificate_level=certificate_level_for_policy_repair(
                    original_success=original_success,
                    original_len=len(original_actions),
                    edited_len=len(edited_actions),
                ),
                repair_suffix_len=len(suffix),
                edited_action_len=len(edited_actions) if edited_actions else None,
                original_action_len=len(original_actions) if original_actions else None,
                repaired_outcome="success" if bool(edit.get("success", True)) else "fail",
            )
        elif (
            edit_type == "candidate_policy_suffix"
            and "candidate_policy_repair" in labels
            and edit.get("replacement") is not None
        ):
            step_index = int(edit.get("index", -1))
            repair = RepairAction(
                action=str(edit["replacement"]),
                source="candidate_policy_suffix",
                certificate_level=certificate_level_for_policy_repair(
                    original_success=original_success,
                    original_len=len(original_actions),
                    edited_len=len(edited_actions),
                ),
                repair_suffix_len=1 + len(suffix),
                edited_action_len=len(edited_actions) if edited_actions else None,
                original_action_len=len(original_actions) if original_actions else None,
                repaired_outcome="success" if bool(edit.get("success", True)) else "fail",
            )
        elif (
            edit_type == "suffix_match_policy"
            and "suffix_match_repair" in labels
            and edit.get("replacement") is not None
        ):
            step_index = int(edit.get("index", -1))
            repair = RepairAction(
                action=str(edit["replacement"]),
                source="suffix_match_policy",
                certificate_level=certificate_level_for_policy_repair(
                    original_success=original_success,
                    original_len=len(original_actions),
                    edited_len=len(edited_actions),
                ),
                repair_suffix_len=1 + len(suffix),
                edited_action_len=len(edited_actions) if edited_actions else None,
                original_action_len=len(original_actions) if original_actions else None,
                repaired_outcome="success" if bool(edit.get("success", True)) else "fail",
            )
        if repair is None or step_index is None:
            continue
        current = repairs.get(step_index)
        if current is None or repair_sort_key(repair) < repair_sort_key(current):
            repairs[step_index] = repair
    return repairs


def repair_sort_key(repair: RepairAction) -> tuple[int, int, str]:
    source_order = {
        "candidate_policy_suffix": 0,
        "suffix_match_policy": 1,
        "policy_repair_suffix": 2,
    }
    source_priority = source_order.get(repair.source, 9)
    return (repair.repair_suffix_len, source_priority, normalize_action(repair.action) or "")


def preferred_action_for_step(
    step: Mapping[str, Any],
    repair_actions: Mapping[int, RepairAction],
    source: str,
) -> tuple[str | None, str | None, RepairAction | None]:
    step_index = int(step.get("step_index", 0))
    if source in {"policy-repair", "both"} and step_index in repair_actions:
        repair = repair_actions[step_index]
        return repair.action, repair.source, repair
    if source in {"gold", "both"} and step.get("gold_action") is not None:
        return (
            str(step["gold_action"]),
            "gold_action",
            RepairAction(
                action=str(step["gold_action"]),
                source="gold_action",
                certificate_level="C0_candidate_only",
            ),
        )
    return None, None, None


def step_candidates(step: Mapping[str, Any]) -> tuple[str, ...]:
    raw = step.get("candidates_before") or step.get("candidates") or ()
    return tuple(str(action) for action in raw)


def normalize_action(action: str | None) -> str | None:
    if action is None:
        return None
    return re.sub(r"\s+", " ", str(action).strip().lower())


def rank(candidates: tuple[str, ...], action: str | None) -> int | None:
    if action is None:
        return None
    try:
        return candidates.index(action) + 1
    except ValueError:
        return None


def certificate_level_for_policy_repair(
    original_success: bool,
    original_len: int,
    edited_len: int,
) -> str:
    if original_success and edited_len and edited_len < original_len:
        return "C3_shorter_success"
    if not original_success:
        return "C3_failure_repair"
    return "C1_oracle_suffix"


def summarize_preferences(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(str(record.get("source", "")) for record in records)
    certificate_counts = Counter(str(record.get("certificate_level", "")) for record in records)
    tasks = {str(record["task_id"]) for record in records}
    preferred_ranks = [
        int(record["preferred_rank_before"])
        for record in records
        if record.get("preferred_rank_before") is not None
    ]
    return {
        "preferences": len(records),
        "tasks": len(tasks),
        "avg_preferred_rank_before": (
            sum(preferred_ranks) / len(preferred_ranks) if preferred_ranks else 0.0
        ),
        "sources": dict(sorted(source_counts.items())),
        "certificates": dict(sorted(certificate_counts.items())),
    }


if __name__ == "__main__":
    main()
