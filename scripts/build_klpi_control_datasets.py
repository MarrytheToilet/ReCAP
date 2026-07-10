"""Build the KLPI attribution-control datasets (shuffled / loop-only rewards).

Both variants keep every field of the source records except the supervision:
  - shuffled: candidate_rewards values permuted within each record (seed 42);
    preferred_action re-derived as the argmax of the permuted rewards.
  - loop-only: rewards from structural rules only. A candidate is bad (-0.8)
    if it repeats one of the last four history actions, starts with a no-op
    verb (look/inventory/wait/examine), or is the inverse of the last go-move;
    all other candidates get +0.4. preferred_action is the first non-bad
    candidate in the raw order.

With the default seed this regenerates the exact training files used for the
paper's attribution controls (Appendix Table `klpi_controls`).
"""

from __future__ import annotations

import argparse
import json
import random


OPPOSITE = {"north": "south", "south": "north", "east": "west",
            "west": "east", "up": "down", "down": "up"}
NOOP_VERBS = ("look", "inventory", "wait", "examine")


def structural_bad(action: str, history: list[str]) -> bool:
    lowered = action.strip().lower()
    if any(lowered == past.strip().lower() for past in history[-4:]):
        return True
    if lowered.split()[0] in NOOP_VERBS:
        return True
    if history:
        last = history[-1].strip().lower()
        if last.startswith("go ") and lowered.startswith("go "):
            if OPPOSITE.get(last.split()[-1]) == lowered.split()[-1]:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="analysis/recap_xhard_700_online_policy_progress_pool60.jsonl",
    )
    parser.add_argument(
        "--out-shuffled",
        default="analysis/recap_xhard_700_online_policy_pool60_shuffled.jsonl",
    )
    parser.add_argument(
        "--out-looponly",
        default="analysis/recap_xhard_700_online_policy_pool60_looponly.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.source, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    rng = random.Random(args.seed)
    with open(args.out_shuffled, "w", encoding="utf-8") as handle:
        for record in rows:
            out = dict(record)
            rewards = record["candidate_rewards"]
            keys, values = list(rewards.keys()), list(rewards.values())
            rng.shuffle(values)
            out["candidate_rewards"] = dict(zip(keys, values))
            out["preferred_action"] = max(
                out["candidate_rewards"], key=out["candidate_rewards"].get
            )
            out["source"] = "shuffled_reward_control"
            handle.write(json.dumps(out) + "\n")

    with open(args.out_looponly, "w", encoding="utf-8") as handle:
        for record in rows:
            out = dict(record)
            history = record.get("history") or []
            out["candidate_rewards"] = {
                candidate: (-0.8 if structural_bad(candidate, history) else 0.4)
                for candidate in record["candidates"]
            }
            good = [c for c in record["candidates"] if out["candidate_rewards"][c] > 0]
            out["preferred_action"] = good[0] if good else record["candidates"][0]
            out["source"] = "loop_only_reward_control"
            handle.write(json.dumps(out) + "\n")

    print(f"wrote {len(rows)} records to each of:")
    print(" ", args.out_shuffled)
    print(" ", args.out_looponly)


if __name__ == "__main__":
    main()
