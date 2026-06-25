from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from recap.envs.textworld_adapter import TextWorldAdapter
from recap.rewrite import NormalizerConfig, ReplayNormalizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normal-form rewriting on JSONL prefixes.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", default="full", choices=["full", "observable", "goal"])
    parser.add_argument("--env", default="textworld", choices=["textworld"])
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.jsonl)
    traces = unique_traces(records)[: args.limit]

    adapter = TextWorldAdapter()
    normalizer = ReplayNormalizer(
        adapter=adapter,
        env_name=args.env,
        config=NormalizerConfig(equivalence_mode=args.mode),
    )

    rule_counts: Counter[str] = Counter()
    preserved = 0
    original_valid = 0
    normalized_valid = 0
    original_total_len = 0
    normalized_total_len = 0
    compressed = 0
    examples: list[dict[str, Any]] = []

    for task_id, seed, actions in traces:
        result = normalizer.normalize(task_id=task_id, seed=seed, actions=actions)
        preserved += int(result.state_preserved)
        original_valid += int(result.original_valid)
        normalized_valid += int(result.normalized_valid)
        original_total_len += len(result.original_actions)
        normalized_total_len += len(result.normalized_actions)
        compressed += int(len(result.normalized_actions) < len(result.original_actions))
        rule_counts.update(step.rule for step in result.steps)

        if len(examples) < args.examples and result.steps:
            examples.append(
                {
                    "task_id": task_id,
                    "original": result.original_actions,
                    "normalized": result.normalized_actions,
                    "rules": [step.rule for step in result.steps],
                    "state_preserved": result.state_preserved,
                }
            )

    total = len(traces)
    summary = {
        "traces": total,
        "state_preserved": preserved,
        "state_preservation_rate": preserved / total if total else 0.0,
        "original_valid": original_valid,
        "normalized_valid": normalized_valid,
        "avg_original_len": original_total_len / total if total else 0.0,
        "avg_normalized_len": normalized_total_len / total if total else 0.0,
        "compression_ratio": normalized_total_len / original_total_len if original_total_len else 0.0,
        "compressed_traces": compressed,
        "rule_counts": dict(sorted(rule_counts.items())),
        "examples": examples,
    }

    print(f"traces={summary['traces']}")
    print(
        f"state_preserved={summary['state_preserved']}/{total} "
        f"rate={summary['state_preservation_rate']:.3f}"
    )
    print(f"original_valid={summary['original_valid']}/{total}")
    print(f"normalized_valid={summary['normalized_valid']}/{total}")
    print(f"avg_original_len={summary['avg_original_len']:.3f}")
    print(f"avg_normalized_len={summary['avg_normalized_len']:.3f}")
    print(f"compression_ratio={summary['compression_ratio']:.3f}")
    print(f"compressed_traces={summary['compressed_traces']}/{total}")
    print("rules:")
    for rule, count in summary["rule_counts"].items():
        print(f"  {rule}: {count}")
    print("examples:")
    for example in summary["examples"]:
        print(json.dumps(to_jsonable(example), ensure_ascii=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote={args.out}")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_traces(records: list[dict[str, Any]]) -> list[tuple[str, int, tuple[str, ...]]]:
    seen: set[tuple[str, int, tuple[str, ...]]] = set()
    traces: list[tuple[str, int, tuple[str, ...]]] = []
    for record in records:
        trace = (
            record["task_id"],
            int(record["seed"]),
            tuple(record["prefix_actions"]),
        )
        if trace in seen or not trace[2]:
            continue
        seen.add(trace)
        traces.append(trace)
    return traces


def to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
