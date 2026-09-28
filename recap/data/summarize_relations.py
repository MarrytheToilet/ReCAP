from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ReCAP relation JSONL records.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    records = load_records(args.jsonl)
    if not records:
        raise SystemExit(f"no records found in {args.jsonl}")

    print(f"records: {len(records)}")
    print(f"envs: {format_counts(Counter(record['env'] for record in records))}")
    print(f"equivalence_modes: {format_counts(Counter(record['equivalence_mode'] for record in records))}")
    print(f"prefix_lengths: {format_counts(Counter(len(record['prefix_actions']) for record in records))}")
    print()

    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for relation, value in record["relations"].items():
            relation_counts[relation][str(value)] += 1

    print("relations:")
    for relation in sorted(relation_counts):
        print(f"  {relation}: {format_counts(relation_counts[relation])}")
    print()

    print("core ratios:")
    print_ratio("commute", relation_counts)
    print_ratio("idempotent_a", relation_counts)
    print_ratio("idempotent_b", relation_counts)
    print()

    print("effect buckets:")
    print_effect_buckets(records)
    print()

    print("sample non-commuting pairs:")
    print_samples(records, relation="commute", value=False, limit=args.samples)
    print()

    print("sample unknown commute pairs:")
    print_samples(records, relation="commute", value=None, limit=args.samples)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_counts(counter: Counter[Any]) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter, key=str))


def print_ratio(relation: str, relation_counts: dict[str, Counter[str]]) -> None:
    counts = relation_counts[relation]
    known = counts["True"] + counts["False"]
    if known == 0:
        print(f"  {relation}: known=0")
        return
    true_ratio = counts["True"] / known
    print(
        f"  {relation}: true={counts['True']} false={counts['False']} "
        f"unknown={counts['None']} true_ratio_known={true_ratio:.3f}"
    )


def print_effect_buckets(records: list[dict[str, Any]]) -> None:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sigs = record["signatures"]
        effect_a = sigs["s"] != sigs["a"]
        effect_b = sigs["s"] != sigs["b"]
        if effect_a and effect_b:
            bucket = "both_effectful"
        elif effect_a or effect_b:
            bucket = "one_effectful"
        else:
            bucket = "neither_effectful"
        buckets[bucket].append(record)

    for bucket in ["both_effectful", "one_effectful", "neither_effectful"]:
        bucket_records = buckets[bucket]
        counts = Counter(str(record["relations"]["commute"]) for record in bucket_records)
        known = counts["True"] + counts["False"]
        ratio = counts["True"] / known if known else 0.0
        print(
            f"  {bucket}: n={len(bucket_records)} "
            f"commute_true={counts['True']} commute_false={counts['False']} "
            f"commute_unknown={counts['None']} true_ratio_known={ratio:.3f}"
        )


def print_samples(
    records: list[dict[str, Any]],
    relation: str,
    value: bool | None,
    limit: int,
) -> None:
    printed = 0
    for record in records:
        if record["relations"].get(relation) != value:
            continue
        prefix = " ; ".join(record["prefix_actions"]) or "<empty>"
        print(
            f"  prefix=[{prefix}] | "
            f"a={record['action_a']!r} | b={record['action_b']!r} | "
            f"valid={record['validities']}"
        )
        printed += 1
        if printed >= limit:
            break
    if printed == 0:
        print("  <none>")


if __name__ == "__main__":
    main()
