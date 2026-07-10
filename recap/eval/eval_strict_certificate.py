"""Audit replay-certified preferences with a strict paired-cost certificate.

For each emitted preference (a+ > a_exec at step t), this script compares the
verifier-relative remaining cost of the certified branch with the cost of the
executed action, using the per-prefix policy suffixes already stored in trace
reports:

    l_star(s_t)   = len(policy suffix from s_t)          -> 1 + l(a+)
    l(a_exec)     = len(policy suffix from s_{t+1})

A preference passes the strict certificate at margin delta when

    l(a+) + delta <= l(a_exec)    <=>    l(a_exec) - l_star(s_t) + 1 >= delta.

Statuses per preference:
    strict_misrank            delta_steps >= 1 (executed action makes no
                              verified progress; strictly worse at delta=1)
    tied_repair               delta_steps == 0 (executed action lies on an
                              equally short verified path)
    verifier_shorter_after_exec  delta_steps < 0 (the verifier finds a shorter
                              path after the executed action than it found at
                              s_t; evidence of verifier suboptimality, the
                              label is downgraded to branch-relative)
    unmatched / missing_suffix   bookkeeping failures, reported explicitly

No environment replays are performed; everything is recomputed from logged
trace reports, so the audit is deterministic and reproducible from logs.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferences", required=True)
    parser.add_argument(
        "--trace-reports",
        nargs="+",
        required=True,
        help="trace report JSON paths or globs (policy_suffix edits required)",
    )
    parser.add_argument("--deltas", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-rows", default=None)
    parser.add_argument("--out-summary", default=None)
    args = parser.parse_args()

    preferences = load_jsonl(args.preferences)
    reports = load_reports(args.trace_reports)
    rows = [audit_preference(pref, reports) for pref in preferences]
    summary = summarize(rows, deltas=args.deltas)

    if args.out_rows:
        with open(args.out_rows, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.out_summary:
        with open(args.out_summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_reports(patterns: Sequence[str]) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    reports: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    seen: set[tuple[str, int, tuple[str, ...]]] = set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        for report in data.get("reports", []):
            key = (str(report.get("task_id")), int(report.get("seed", 0)))
            actions = tuple(str(action) for action in report.get("original_actions") or ())
            dedup_key = (key[0], key[1], actions)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            reports.setdefault(key, []).append(report)
    return reports


def audit_preference(
    pref: Mapping[str, Any],
    reports: Mapping[tuple[str, int], list[Mapping[str, Any]]],
) -> dict[str, Any]:
    key = (str(pref.get("task_id")), int(pref.get("seed", 0)))
    step_index = int(pref["step_index"])
    history = [str(action) for action in pref.get("history") or ()]
    row: dict[str, Any] = {
        "task_id": key[0],
        "seed": key[1],
        "step_index": step_index,
        "preferred_action": pref.get("preferred_action"),
        "executed_action": pref.get("executed_action"),
        "repair_suffix_len": pref.get("repair_suffix_len"),
        "source": pref.get("source"),
    }
    matches = [
        report
        for report in reports.get(key, [])
        if not report.get("original_success")
        and list((report.get("original_actions") or [])[: len(history)]) == history
    ]
    if not matches:
        row.update(status="unmatched", delta_steps=None)
        return row
    row["ambiguous_match"] = len(matches) > 1
    suffix_lens = policy_suffix_lengths(matches[0])
    l_star = suffix_lens.get(step_index - 1)
    l_exec = suffix_lens.get(step_index)
    row["l_star_state"] = l_star
    row["l_exec"] = l_exec
    if l_star is None or l_exec is None:
        row.update(status="missing_suffix", delta_steps=None)
        return row
    row["suffix_len_consistent"] = (
        pref.get("source") != "policy_repair_suffix"
        or l_star == pref.get("repair_suffix_len")
    )
    delta_steps = l_exec - l_star + 1  # l(a_exec) - l(a+)
    row["delta_steps"] = delta_steps
    if delta_steps >= 1:
        row["status"] = "strict_misrank"
    elif delta_steps == 0:
        row["status"] = "tied_repair"
    else:
        row["status"] = "verifier_shorter_after_exec"
    return row


def policy_suffix_lengths(report: Mapping[str, Any]) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for edit in report.get("edits") or ():
        if edit.get("edit_type") == "policy_suffix":
            lengths[int(edit["index"])] = len(edit.get("repair_suffix") or ())
    return lengths


def summarize(rows: Sequence[Mapping[str, Any]], deltas: Sequence[int]) -> dict[str, Any]:
    statuses = Counter(row["status"] for row in rows)
    audited = [row for row in rows if row.get("delta_steps") is not None]
    delta_hist = Counter(row["delta_steps"] for row in audited)
    strict_by_delta = {
        str(delta): sum(1 for row in audited if row["delta_steps"] >= delta)
        for delta in deltas
    }
    return {
        "num_preferences": len(rows),
        "num_audited": len(audited),
        "statuses": dict(statuses),
        "strict_pass_by_delta": strict_by_delta,
        "strict_rate_by_delta": {
            delta: (count / len(audited) if audited else None)
            for delta, count in strict_by_delta.items()
        },
        "delta_steps_histogram": {str(k): v for k, v in sorted(delta_hist.items())},
        "num_ambiguous_match": sum(1 for row in rows if row.get("ambiguous_match")),
        "num_suffix_len_inconsistent": sum(
            1 for row in audited if row.get("suffix_len_consistent") is False
        ),
    }


if __name__ == "__main__":
    main()
