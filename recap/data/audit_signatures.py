from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


COMPARISONS = {
    "commute": ("ab", "ba"),
    "idempotent_a": ("a", "aa"),
    "idempotent_b": ("b", "bb"),
    "inverse_ab": ("s", "ab"),
    "inverse_ba": ("s", "ba"),
    "absorb_ab": ("a", "ab"),
    "absorb_ba": ("b", "ba"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit signature diffs behind relation labels.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--relation", choices=sorted(COMPARISONS), default="commute")
    parser.add_argument("--value", choices=["true", "false", "none", "any"], default="any")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-diff", type=int, default=8)
    args = parser.parse_args()

    records = load_records(args.jsonl)
    selected = [
        record
        for record in records
        if args.relation in record["relations"]
        and (args.value == "any" or record["relations"][args.relation] is parse_value(args.value))
    ]

    print(f"file: {args.jsonl}")
    print(f"records: {len(records)}")
    print(f"selected: {len(selected)} relation={args.relation} value={args.value}")
    print()

    counts = Counter()
    suspicious: list[dict[str, Any]] = []
    for record in selected:
        audit = audit_record(record, args.relation)
        counts[audit["status"]] += 1
        if audit["suspicious"]:
            suspicious.append(record)

    print("audit_status:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"suspicious: {len(suspicious)}")
    print()

    print("samples:")
    for record in selected[: args.limit]:
        print_audit(record, args.relation, args.max_diff)

    if suspicious:
        print("suspicious_samples:")
        for record in suspicious[: args.limit]:
            print_audit(record, args.relation, args.max_diff)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_value(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "none":
        return None
    raise ValueError(f"cannot parse relation value: {value}")


def audit_record(record: dict[str, Any], relation: str) -> dict[str, Any]:
    left_key, right_key = COMPARISONS[relation]
    only_left, only_right = signature_diff(
        record["signatures"][left_key],
        record["signatures"][right_key],
    )
    has_diff = bool(only_left or only_right)
    label = record["relations"][relation]

    if label is True:
        status = "true_empty_diff" if not has_diff else "true_nonempty_diff"
        suspicious = has_diff
    elif label is False:
        status = "false_nonempty_diff" if has_diff else "false_empty_diff"
        suspicious = not has_diff
    else:
        status = "none_nonempty_diff" if has_diff else "none_empty_diff"
        suspicious = none_label_is_suspicious(record, relation)

    return {
        "status": status,
        "suspicious": suspicious,
        "only_left": only_left,
        "only_right": only_right,
    }


def none_label_is_suspicious(record: dict[str, Any], relation: str) -> bool:
    validities = record["validities"]
    if relation == "commute":
        return validities["ab"] and validities["ba"]
    if relation == "idempotent_a":
        return validities["a"] and validities["aa"]
    if relation == "idempotent_b":
        return validities["b"] and validities["bb"]
    if relation == "inverse_ab":
        return validities["a"] and validities["ab"]
    if relation == "inverse_ba":
        return validities["b"] and validities["ba"]
    if relation == "absorb_ab":
        return validities["a"] and validities["ab"]
    if relation == "absorb_ba":
        return validities["b"] and validities["ba"]
    return False


def signature_diff(left: Any, right: Any) -> tuple[list[str], list[str]]:
    left_items = set(canonical_items(left))
    right_items = set(canonical_items(right))
    return sorted(left_items - right_items), sorted(right_items - left_items)


def canonical_items(signature: Any) -> list[str]:
    if isinstance(signature, list):
        return [canonical_item(item) for item in signature]
    return [canonical_item(signature)]


def canonical_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def print_audit(record: dict[str, Any], relation: str, max_diff: int) -> None:
    left_key, right_key = COMPARISONS[relation]
    audit = audit_record(record, relation)
    prefix = " ; ".join(record["prefix_actions"]) or "<empty>"
    print(
        f"  prefix=[{prefix}] | a={record['action_a']!r} | b={record['action_b']!r} | "
        f"{relation}={record['relations'][relation]} | compare={left_key}/{right_key}"
    )
    print(f"    validities={record['validities']}")
    print_limited("only_left", audit["only_left"], max_diff)
    print_limited("only_right", audit["only_right"], max_diff)


def print_limited(label: str, items: list[str], limit: int) -> None:
    shown = items[:limit]
    print(f"    {label}({len(items)}): {shown}")


if __name__ == "__main__":
    main()
