from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from acta.envs.textworld_adapter import TextWorldAdapter
from acta.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add replayed observations and initial objectives to ReCAP records."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    records = tuple(read_jsonl(args.input))
    enriched, summary = enrich_records(records)
    write_jsonl(args.out, enriched)
    summary_path = args.summary_out or args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def enrich_records(records: tuple[Mapping[str, Any], ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter = TextWorldAdapter()
    enriched: list[dict[str, Any]] = []
    replay_errors = 0
    try:
        initial_cache: dict[tuple[str, int], str] = {}
        observation_cache: dict[tuple[str, int, tuple[str, ...]], str | None] = {}
        for record in records:
            task_id = str(record["task_id"])
            seed = int(record.get("seed", 0))
            history = tuple(str(action) for action in record.get("history", ()))
            initial_key = (task_id, seed)
            if initial_key not in initial_cache:
                reset = adapter.reset(task_id=task_id, seed=seed)
                initial_cache[initial_key] = reset.observation
            observation_key = (task_id, seed, history)
            if observation_key not in observation_cache:
                replay = adapter.replay(task_id=task_id, prefix_actions=history, seed=seed)
                observation_cache[observation_key] = replay.observation if replay.valid else None
            row = dict(record)
            row["initial_observation"] = initial_cache[initial_key]
            observation = observation_cache[observation_key]
            if observation is None:
                replay_errors += 1
                row["replay_context_valid"] = False
            else:
                row["observation"] = observation
                row["replay_context_valid"] = True
            enriched.append(row)
    finally:
        adapter.close()
    return enriched, {
        "records": len(records),
        "enriched": len(enriched),
        "replay_context_errors": replay_errors,
    }


if __name__ == "__main__":
    main()
