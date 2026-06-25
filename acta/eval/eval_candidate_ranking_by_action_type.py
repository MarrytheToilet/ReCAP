from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from acta.eval.eval_candidate_ranking import (
    evaluate_candidate_ranking,
    index_predictions,
    summarize_ranking,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break candidate-ranking metrics down by preferred action type."
    )
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preferences = tuple(read_jsonl(args.preferences))
    predictions = (
        index_predictions(tuple(read_jsonl(args.predictions)))
        if args.predictions is not None
        else {}
    )
    output = evaluate_by_action_type(preferences, predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def evaluate_by_action_type(
    preferences: tuple[Mapping[str, Any], ...],
    predictions: Mapping[tuple[str, int, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    records = evaluate_candidate_ranking(preferences, predictions or {})
    enriched = []
    for preference, record in zip(preferences, records, strict=True):
        group = action_group(str(preference.get("preferred_action", "")))
        row = dict(record)
        row["preferred_action_group"] = group
        row["preferred_action_verb"] = action_verb(str(preference.get("preferred_action", "")))
        enriched.append(row)
    groups = {
        "all": enriched,
        "navigation": [record for record in enriched if record["preferred_action_group"] == "navigation"],
        "non_navigation": [
            record for record in enriched if record["preferred_action_group"] != "navigation"
        ],
    }
    return {
        "summary": {
            group: summarize_ranking(records)
            for group, records in groups.items()
        },
        "verb_counts": verb_counts(enriched),
        "records": enriched,
    }


def action_group(action: str) -> str:
    verb = action_verb(action)
    if verb == "go":
        return "navigation"
    if verb in {"take", "drop", "put", "open", "close", "unlock", "lock", "insert"}:
        return "manipulation"
    if verb in {"look", "inventory", "examine", "read"}:
        return "inspection"
    return "other"


def action_verb(action: str) -> str:
    tokens = action.strip().lower().split()
    return tokens[0] if tokens else ""


def verb_counts(records: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        verb = str(record.get("preferred_action_verb", ""))
        counts[verb] = counts.get(verb, 0) + 1
    return dict(sorted(counts.items()))


def read_jsonl(path: Path | None) -> list[Mapping[str, Any]]:
    if path is None:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
