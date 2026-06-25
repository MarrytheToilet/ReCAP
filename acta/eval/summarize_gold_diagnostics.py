from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gold-action diagnostics from agent runs.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("analysis/gold_diagnostics.md"))
    args = parser.parse_args()

    rows = [row_from_run(path) for path in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows), encoding="utf-8")
    print(f"wrote={args.out} rows={len(rows)}")


def row_from_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = [step for episode in data["episodes"] for step in episode["steps"]]
    gold_steps = [step for step in steps if step.get("gold_action") is not None]
    gold_in = [step for step in gold_steps if step.get("gold_in_candidates")]
    selected_gold = [step for step in gold_steps if step.get("selected_is_gold")]
    recovery_opportunities = [
        step
        for step in gold_steps
        if step.get("gold_rank_before") is not None and step.get("gold_rank_before") != 1
    ]
    recovered = [step for step in gold_steps if step.get("acta_recovered_gold")]
    demotion_opportunities = [step for step in gold_steps if step.get("gold_rank_before") == 1]
    demoted = [step for step in gold_steps if step.get("acta_demoted_gold")]
    top1_bad = [step for step in steps if step.get("top1_structural_bad")]
    blocked_bad_top1 = [step for step in steps if step.get("acta_blocked_bad_top1")]

    return {
        "run": path.stem,
        "controller": data["summary"].get("controller", ""),
        "episodes": data["summary"].get("episodes", 0),
        "success_rate": data["summary"].get("success_rate", 0.0),
        "gold_steps": len(gold_steps),
        "gold_in_topk_rate": rate(len(gold_in), len(gold_steps)),
        "selected_gold_rate": rate(len(selected_gold), len(gold_steps)),
        "recovery_opps": len(recovery_opportunities),
        "recovery_rate": rate(len(recovered), len(recovery_opportunities)),
        "demotion_opps": len(demotion_opportunities),
        "demotion_rate": rate(len(demoted), len(demotion_opportunities)),
        "bad_top1_steps": len(top1_bad),
        "bad_top1_block_rate": rate(len(blocked_bad_top1), len(top1_bad)),
    }


def render(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "controller",
        "episodes",
        "success_rate",
        "gold_steps",
        "gold_in_topk_rate",
        "selected_gold_rate",
        "recovery_opps",
        "recovery_rate",
        "demotion_opps",
        "demotion_rate",
        "bad_top1_steps",
        "bad_top1_block_rate",
    ]
    lines = [
        "# Gold Action Diagnostics",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[header]) for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
