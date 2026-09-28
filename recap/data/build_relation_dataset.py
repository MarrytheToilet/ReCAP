from __future__ import annotations

import argparse
from pathlib import Path

from recap.envs.toy_adapter import ToyAdapter
from recap.probe import PairProbe, ProbeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small ReCAP relation JSONL file.")
    parser.add_argument("--env", choices=["toy"], default="toy")
    parser.add_argument("--out", type=Path, default=Path("data/relation_records.jsonl"))
    parser.add_argument("--mode", default="full", choices=["full", "observable", "goal"])
    args = parser.parse_args()

    if args.env != "toy":
        raise ValueError("Only the local toy environment is wired in this bootstrap script.")

    adapter = ToyAdapter()
    probe = PairProbe(adapter=adapter, env_name="toy", config=ProbeConfig(equivalence_mode=args.mode))
    examples = [
        ((), "look", "read note"),
        ((), "open door", "open door"),
        ((), "open door", "go kitchen"),
        (("open door", "go kitchen", "take apple"), "put apple in fridge", "put apple in sink"),
        ((), "toggle lamp", "toggle lamp"),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for prefix, action_a, action_b in examples:
            record = probe.probe_pair(
                task_id="toy-default",
                seed=0,
                prefix_actions=prefix,
                action_a=action_a,
                action_b=action_b,
            )
            handle.write(record.to_json() + "\n")


if __name__ == "__main__":
    main()

