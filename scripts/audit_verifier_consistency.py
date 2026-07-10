"""Global verifier-consistency audit over failed-trajectory trace reports.

For every failed trajectory, compares verifier suffix lengths at consecutive
prefix states. A single action can legally reduce the verifier's remaining
cost by at most one step, so a recorded drop of two or more proves the suffix
at the earlier state was not minimal (verifier suboptimality). The drop
histogram also shows how often the executed action makes verified progress,
which is the non-tautology argument for the strict audit.

Reproduces: 991 consecutive-step pairs, 8 violations (0.81%),
histogram {1: 479, 0: 274, -1: 229, 3: 8, -9: 1}.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-reports",
        nargs="+",
        default=["analysis/recap_xhard_*trace_report.json"],
    )
    args = parser.parse_args()

    reports: dict[tuple, dict] = {}
    for pattern in args.trace_reports:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            for report in data.get("reports", []):
                dedup = (
                    report.get("task_id"),
                    report.get("seed", 0),
                    tuple(report.get("original_actions") or ()),
                )
                reports.setdefault(dedup, report)

    pairs = 0
    violations = 0
    drops: Counter = Counter()
    for report in reports.values():
        if report.get("original_success"):
            continue
        suffix_lens = {
            int(edit["index"]): len(edit.get("repair_suffix") or ())
            for edit in report.get("edits") or ()
            if edit.get("edit_type") == "policy_suffix"
        }
        for index in sorted(suffix_lens):
            if index + 1 in suffix_lens:
                pairs += 1
                drop = suffix_lens[index] - suffix_lens[index + 1]
                drops[drop] += 1
                if drop > 1:
                    violations += 1

    print(f"consecutive-step pairs: {pairs}")
    print(f"violations (drop > 1): {violations} ({violations / pairs * 100:.2f}%)")
    print("drop histogram:", dict(sorted(drops.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
