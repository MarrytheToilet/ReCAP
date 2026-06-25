from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RL evaluation JSON files.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("analysis/rl_comparison.md"))
    args = parser.parse_args()

    rows = [row_from_run(path) for path in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows), encoding="utf-8")
    print(f"wrote={args.out} rows={len(rows)}")


def row_from_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    return {
        "run": path.stem,
        "prior": summary["prior"],
        "runs": summary["runs"],
        "success_rate": summary["avg_success_rate"],
        "last20_success": summary["avg_last_window_success_rate"],
        "first_success": summary["avg_first_success_episode"],
        "avg_return": summary["avg_return"],
        "avg_steps": summary["avg_steps"],
        "noop_seen_rate": summary["avg_noop_or_seen_selected_rate"],
        "blocked_per_step": summary["avg_blocked_actions_per_step"],
    }


def render(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "prior",
        "runs",
        "success_rate",
        "last20_success",
        "first_success",
        "avg_return",
        "avg_steps",
        "noop_seen_rate",
        "blocked_per_step",
    ]
    lines = [
        "# RL Run Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[header]) for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
