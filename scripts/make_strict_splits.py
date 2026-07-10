"""Filter preference splits down to strict-certificate preferences.

Reads the strict-audit rows produced by ``recap.eval.eval_strict_certificate``
and keeps only preferences whose status is ``strict_misrank``, writing
filtered train/valid/test splits next to the originals.

Reproduces: 212/38/105 -> 208/37/101 on the xhard-700 splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-rows",
        default="analysis/recap_xhard_700_mimo25_t1_top5_strict_certificate_rows.jsonl",
    )
    parser.add_argument(
        "--splits-dir",
        default="analysis/recap_xhard_700_mimo25_t1_top5_splits_t30",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/recap_xhard_700_mimo25_t1_top5_splits_t30_strict",
    )
    args = parser.parse_args()

    strict = set()
    with open(args.strict_rows, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["status"] == "strict_misrank":
                strict.add((row["task_id"], str(row["seed"]), str(row["step_index"])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        src = Path(args.splits_dir) / f"{split}.jsonl"
        records = [json.loads(line) for line in open(src, encoding="utf-8")]
        kept = [
            record
            for record in records
            if (record["task_id"], str(record.get("seed", 0)), str(record["step_index"])) in strict
        ]
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record) + "\n")
        print(f"{split}: {len(records)} -> {len(kept)}")


if __name__ == "__main__":
    main()
