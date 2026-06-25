from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from acta.models.reranker_dataset import normalize_action, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add replay-value soft targets to ReCAP preference records."
    )
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--multiplicity-rows", type=Path, required=True)
    parser.add_argument("--preferred-reward", type=float, default=1.0)
    parser.add_argument("--success-base", type=float, default=0.55)
    parser.add_argument("--success-bonus", type=float, default=0.45)
    parser.add_argument("--rejected-cap", type=float, default=0.20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = index_rows(tuple(read_jsonl(args.multiplicity_rows)))
    enriched = [
        enrich_record(
            record,
            rows.get(record_key(record)),
            preferred_reward=args.preferred_reward,
            success_base=args.success_base,
            success_bonus=args.success_bonus,
            rejected_cap=args.rejected_cap,
        )
        for record in read_jsonl(args.preferences)
    ]
    write_jsonl(args.out, enriched)
    summary = summarize(enriched)
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def index_rows(rows: tuple[Mapping[str, Any], ...]) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    return {record_key(row): row for row in rows}


def record_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["task_id"]),
        int(record.get("seed", 0)),
        int(record.get("step_index", 0)),
    )


def enrich_record(
    record: Mapping[str, Any],
    multiplicity: Mapping[str, Any] | None,
    preferred_reward: float,
    success_base: float,
    success_bonus: float,
    rejected_cap: float,
) -> dict[str, Any]:
    output = dict(record)
    candidates = [str(action) for action in record.get("candidates", ())]
    rewards = {action: 0.0 for action in candidates}
    if multiplicity is not None:
        suffix_by_action = successful_suffixes(multiplicity)
        for action in candidates:
            suffix_len = suffix_by_action.get(normalize_action(action))
            if suffix_len is None:
                continue
            rewards[action] = success_base + success_bonus / (1.0 + float(suffix_len))
    preferred = str(record.get("preferred_action", ""))
    rejected = str(record.get("rejected_action", ""))
    if rejected in rewards and rejected != preferred:
        rewards[rejected] = min(rewards[rejected], rejected_cap)
    if preferred in rewards:
        rewards[preferred] = max(rewards[preferred], preferred_reward)
    output["candidate_replay_values"] = rewards
    output["candidate_rewards"] = rewards
    output["replay_value_source"] = "candidate_branch_replay_success_with_preferred_branch_target"
    output["replay_value_successful_candidates"] = sum(
        1 for action, value in rewards.items() if value > 0 and action != rejected
    )
    return output


def successful_suffixes(row: Mapping[str, Any]) -> dict[str, int]:
    actions = [str(action) for action in row.get("successful_actions", ())]
    suffix_lens = [int(length) for length in row.get("successful_suffix_lens", ())]
    suffix_by_action: dict[str, int] = {}
    for action, suffix_len in zip(actions, suffix_lens):
        key = normalize_action(action)
        current = suffix_by_action.get(key)
        if current is None or suffix_len < current:
            suffix_by_action[key] = suffix_len
    return suffix_by_action


def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    with_values = [record for record in records if isinstance(record.get("candidate_replay_values"), Mapping)]
    multi_positive = [
        record
        for record in with_values
        if sum(1 for value in record["candidate_replay_values"].values() if float(value) > 0.0) > 1
    ]
    return {
        "records": len(records),
        "records_with_values": len(with_values),
        "records_with_multiple_positive_values": len(multi_positive),
        "multiple_positive_rate": len(multi_positive) / len(with_values) if with_values else 0.0,
    }


if __name__ == "__main__":
    main()
