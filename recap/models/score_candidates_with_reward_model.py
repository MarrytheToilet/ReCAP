from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from recap.models.cross_encoder_reranker import rank_candidates
from recap.models.reranker_dataset import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate ReCAP candidate records with dense reward-model scores."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--max-observation-chars", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    records = tuple(read_jsonl(args.input))
    model = CrossEncoder(str(args.model), max_length=args.max_length, device=args.device)
    annotated: list[dict[str, Any]] = []
    reward_spans: list[float] = []
    reward_top_matches = 0
    for record in records:
        enriched = dict(record)
        ranked = rank_candidates(
            enriched,
            model,
            batch_size=args.batch_size,
            max_history=args.max_history,
            max_observation_chars=args.max_observation_chars,
        )
        rewards = {action: float(score) for action, score in ranked}
        enriched["candidate_rewards"] = rewards
        enriched["candidate_reward_model"] = {
            "model_type": "recap_cross_encoder_reward",
            "model_path": str(args.model),
            "max_history": args.max_history,
            "max_observation_chars": args.max_observation_chars,
        }
        if rewards:
            values = list(rewards.values())
            reward_spans.append(max(values) - min(values))
            preferred = str(enriched.get("preferred_action", ""))
            top_action = max(rewards, key=lambda action: rewards[action])
            reward_top_matches += int(top_action == preferred)
        annotated.append(enriched)

    write_jsonl(args.out, annotated)
    summary = {
        "records": len(records),
        "reward_model": str(args.model),
        "reward_top_match_rate": reward_top_matches / len(records) if records else 0.0,
        "avg_reward_span": sum(reward_spans) / len(reward_spans) if reward_spans else 0.0,
        "max_history": args.max_history,
        "max_observation_chars": args.max_observation_chars,
    }
    summary_path = args.summary_out or args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
