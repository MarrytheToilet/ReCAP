from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge trajectory JSONL shards with de-duplication.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--include-errors", action="store_true")
    args = parser.parse_args()

    records, summary = merge_trajectories(
        inputs=tuple(args.inputs),
        include_errors=args.include_errors,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary["out"] = str(args.out)
    summary_out = args.summary_out or args.out.with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")
    print(f"summary={summary_out}")


def merge_trajectories(
    inputs: tuple[Path, ...],
    include_errors: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    skipped_error_records = 0
    duplicate_records = 0
    for path in inputs:
        source_counts[str(path)] = 0
        for record in read_jsonl(path):
            if record.get("error") and not include_errors:
                skipped_error_records += 1
                continue
            key = trajectory_key(record)
            if key in seen:
                duplicate_records += 1
                continue
            seen[key] = record
            source_counts[str(path)] += 1

    records = sorted(
        seen.values(),
        key=lambda record: (
            record.get("game_seed") if record.get("game_seed") is not None else 10**12,
            record.get("rollout_seed", record.get("seed", 0)),
            str(record.get("trajectory_id", "")),
        ),
    )
    summary = {
        "input_files": [str(path) for path in inputs],
        "written_trajectories": len(records),
        "successful_trajectories": sum(1 for record in records if record.get("success", False)),
        "failed_trajectories": sum(1 for record in records if not record.get("success", False)),
        "errored_trajectories": sum(1 for record in records if record.get("error")),
        "total_steps": sum(len(record.get("steps", ())) for record in records),
        "skipped_error_records": skipped_error_records,
        "duplicate_records": duplicate_records,
        "source_counts_unique": source_counts,
    }
    return records, summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def trajectory_key(record: Mapping[str, Any]) -> str:
    if record.get("trajectory_id"):
        return str(record["trajectory_id"])
    return (
        f"{record.get('task_id', '')}:"
        f"seed{record.get('seed', record.get('rollout_seed', 0))}:"
        f"rollout{record.get('rollout_seed', record.get('seed', 0))}"
    )


if __name__ == "__main__":
    main()
