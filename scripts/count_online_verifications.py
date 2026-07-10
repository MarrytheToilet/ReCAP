"""Count branch verifications per step/episode for the online controllers.

Cost model from ``recap/controllers/replay_repair_controller.py``:
  - verified proposal: one branch check for the raw top action plus one per
    verified proposal (budget 1 in the deployed run);
  - full replay: one branch check per logged candidate.

Reproduces (100-game paired logs):
  verified proposal: 7.28 steps/ep, 1.98 verifications/step, 14.4/ep
  full replay:       7.14 steps/ep, 4.63 verifications/step, 33.1/ep
"""

from __future__ import annotations

import argparse
import glob
import json


def episodes(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle)
    return rows


def report(name: str, rows: list[dict], per_step) -> None:
    n_steps = sum(len(episode["steps"]) for episode in rows)
    total = sum(per_step(step) for episode in rows for step in episode["steps"])
    success = sum(1 for episode in rows if episode.get("success")) / len(rows)
    print(
        f"{name}: episodes={len(rows)} success={success:.2f} "
        f"steps/ep={n_steps / len(rows):.2f} "
        f"verifications/step={total / n_steps:.2f} per-episode={total / len(rows):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verified-proposal",
        nargs="+",
        default=[
            "analysis/online_verified_proposal/bge_verified_top1_shorter_9*episodes.jsonl",
            "analysis/online_verified_proposal/bge_verified_top1_shorter_existing20.episodes.jsonl",
        ],
    )
    parser.add_argument(
        "--full-replay",
        nargs="+",
        default=[
            "analysis/online_expanded_seq/recap_replay_9*episodes.jsonl",
            "analysis/online_expanded_seq/recap_replay_existing20.episodes.jsonl",
        ],
    )
    parser.add_argument("--proposal-budget", type=int, default=1)
    args = parser.parse_args()

    def vp_cost(step: dict) -> int:
        pool = step.get("candidates_before") or []
        raw_top = step.get("top1_action") or (pool[0] if pool else None)
        non_raw = sum(1 for candidate in pool if candidate != raw_top)
        return 1 + min(args.proposal_budget, non_raw)

    report("verified proposal", episodes(args.verified_proposal), vp_cost)
    report(
        "full replay",
        episodes(args.full_replay),
        lambda step: len(step.get("candidates_before") or []),
    )


if __name__ == "__main__":
    main()
