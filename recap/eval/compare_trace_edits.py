from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare counterfactual trace-edit reports.")
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("analysis/trace_edit_comparison.md"))
    args = parser.parse_args()

    rows = [row_from_report(path) for path in args.reports]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows), encoding="utf-8")
    print(f"wrote={args.out} rows={len(rows)}")


def row_from_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    actions = summary.get("actions", 0.0)
    return {
        "report": path.stem,
        "episodes": summary.get("episodes", 0),
        "success": summary.get("successful_episodes", 0),
        "actions": actions,
        "necessary_rate": rate(summary.get("necessary_actions", 0.0), actions),
        "redundant_rate": rate(summary.get("redundant_actions", 0.0), actions),
        "harmful": summary.get("harmful_actions", 0.0),
        "order_critical": summary.get("order_critical_pairs", 0.0),
        "order_invariant": summary.get("order_invariant_pairs", 0.0),
        "replacement_repairs": summary.get("repair_candidates", 0.0),
        "policy_repairs": summary.get("policy_repair_candidates", 0.0),
        "shorter_completions": summary.get("shorter_policy_completions", 0.0),
    }


def render(rows: list[dict[str, Any]]) -> str:
    headers = [
        "report",
        "episodes",
        "success",
        "actions",
        "necessary_rate",
        "redundant_rate",
        "harmful",
        "order_critical",
        "order_invariant",
        "replacement_repairs",
        "policy_repairs",
        "shorter_completions",
    ]
    lines = [
        "# Trace Edit Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[header]) for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
