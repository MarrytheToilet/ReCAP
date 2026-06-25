from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Create group-safe train/valid/test ReCAP splits.")
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument(
        "--split-by",
        choices=["seed", "game_id", "config_id", "trajectory_id", "task_id"],
        default="game_id",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records = tuple(read_jsonl(args.preferences))
    splits = make_splits(
        records=records,
        split_by=args.split_by,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        write_jsonl(args.out_dir / f"{split_name}.jsonl", split_records)

    summary = split_summary(splits, split_by=args.split_by)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"out_dir={args.out_dir}")


def make_splits(
    records: tuple[Mapping[str, Any], ...],
    split_by: str,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> dict[str, list[Mapping[str, Any]]]:
    validate_fracs(train_frac, valid_frac, test_frac)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(group_key(record, split_by), []).append(record)

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    train_groups, valid_groups, test_groups = assign_groups(
        group_names,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
    )
    split_group_names = {
        "train": train_groups,
        "valid": valid_groups,
        "test": test_groups,
    }
    assert_no_group_leakage(
        {name: set(split_groups) for name, split_groups in split_group_names.items()}
    )
    return {
        split_name: [
            record
            for group in split_group
            for record in groups[group]
        ]
        for split_name, split_group in split_group_names.items()
    }


def assign_groups(
    group_names: list[str],
    train_frac: float,
    valid_frac: float,
    test_frac: float,
) -> tuple[list[str], list[str], list[str]]:
    total = len(group_names)
    if total <= 1:
        return group_names, [], []

    test_count = rounded_count(total, test_frac)
    valid_count = rounded_count(total, valid_frac)
    if test_frac > 0 and test_count == 0:
        test_count = 1
    if valid_frac > 0 and valid_count == 0 and total - test_count > 1:
        valid_count = 1
    while total - test_count - valid_count < 1:
        if valid_count > 0:
            valid_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    test_groups = group_names[:test_count]
    valid_groups = group_names[test_count : test_count + valid_count]
    train_groups = group_names[test_count + valid_count :]
    return train_groups, valid_groups, test_groups


def rounded_count(total: int, fraction: float) -> int:
    return int(round(total * fraction))


def group_key(record: Mapping[str, Any], split_by: str) -> str:
    if split_by == "seed":
        return str(record.get("game_seed") or infer_seed(str(record.get("task_id", ""))) or record.get("seed", 0))
    if split_by == "game_id":
        return str(record.get("game_id") or Path(str(record.get("task_id", ""))).stem)
    if split_by == "config_id":
        return str(
            record.get("config_id")
            or record.get("difficulty")
            or Path(str(record.get("task_id", ""))).parent.name
            or "default"
        )
    if split_by == "trajectory_id":
        return str(
            record.get("trajectory_id")
            or f"{group_key(record, 'game_id')}:seed{record.get('seed', 0)}"
        )
    if split_by == "task_id":
        return str(record.get("task_id", ""))
    raise ValueError(f"unknown split key: {split_by}")


def infer_seed(task_id: str) -> int | None:
    match = re.search(r"seed(\d+)", Path(task_id).stem)
    return int(match.group(1)) if match else None


def assert_no_group_leakage(split_groups: Mapping[str, set[str]]) -> None:
    names = sorted(split_groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_groups[left] & split_groups[right]
            if overlap:
                raise ValueError(f"group leakage between {left} and {right}: {sorted(overlap)}")


def split_summary(
    splits: Mapping[str, list[Mapping[str, Any]]],
    split_by: str,
) -> dict[str, Any]:
    groups = {
        name: {group_key(record, split_by) for record in records}
        for name, records in splits.items()
    }
    assert_no_group_leakage(groups)
    return {
        "split_by": split_by,
        "records": {name: len(records) for name, records in splits.items()},
        "groups": {name: len(value) for name, value in groups.items()},
        "group_values": {name: sorted(value) for name, value in groups.items()},
        "leakage": False,
    }


def validate_fracs(train_frac: float, valid_frac: float, test_frac: float) -> None:
    if min(train_frac, valid_frac, test_frac) < 0:
        raise ValueError("split fractions must be non-negative")
    total = train_frac + valid_frac + test_frac
    if not 0.999 <= total <= 1.001:
        raise ValueError("split fractions must sum to 1.0")


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
