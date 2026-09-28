from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a per-task relation existence table.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=Path("analysis/textworld_relation_existence.md"))
    args = parser.parse_args()

    records = load_records(args.jsonl)
    rows = [summarize_group("ALL", records)]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    for task_id in sorted(by_task):
        rows.append(summarize_group(Path(task_id).stem, by_task[task_id]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(rows), encoding="utf-8")
    print(f"wrote={args.out} rows={len(rows)}")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_group(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    commute = Counter(record["relations"]["commute"] for record in records)
    known = commute[True] + commute[False]
    return {
        "group": name,
        "records": len(records),
        "prefixes": len({tuple(record["prefix_actions"]) for record in records}),
        "actions": len({record["action_a"] for record in records} | {record["action_b"] for record in records}),
        "commute_true": commute[True],
        "commute_false": commute[False],
        "commute_unknown": commute[None],
        "known_ratio": known / len(records) if records else 0.0,
        "noncommute_ratio_known": commute[False] / known if known else 0.0,
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "group",
        "records",
        "prefixes",
        "actions",
        "commute_true",
        "commute_false",
        "commute_unknown",
        "known_ratio",
        "noncommute_ratio_known",
    ]
    lines = [
        "# TextWorld Relation Existence",
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
