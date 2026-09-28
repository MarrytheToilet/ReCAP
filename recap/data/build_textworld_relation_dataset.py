from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter
from pathlib import Path

from recap.envs.textworld_adapter import TextWorldAdapter
from recap.probe import PairProbe, ProbeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe action relations in a TextWorld game.")
    parser.add_argument("game_file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/textworld_relation_records.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="full", choices=["full", "observable", "goal"])
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-pairs-per-prefix", type=int, default=12)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-prefix-attempts", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--pair-filter",
        choices=["all", "one-effectful", "both-effectful"],
        default="all",
        help="Filter action pairs by whether single actions change the state signature.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    adapter = TextWorldAdapter()
    prefixes = collect_prefixes(
        adapter=adapter,
        game_file=str(args.game_file),
        seed=args.seed,
        rng=rng,
        num_prefixes=args.num_prefixes,
        max_depth=args.max_depth,
        max_attempts=args.max_prefix_attempts,
    )
    check_deterministic_replay(adapter, str(args.game_file), args.seed, prefixes, args.mode)

    probe = PairProbe(
        adapter=TextWorldAdapter(),
        env_name="textworld",
        config=ProbeConfig(equivalence_mode=args.mode),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    total = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for prefix in prefixes:
            replay = adapter.replay(str(args.game_file), prefix, args.seed)
            actions = tuple(sorted(adapter.admissible_actions(replay.state)))
            effects = action_effects(
                adapter=adapter,
                game_file=str(args.game_file),
                seed=args.seed,
                prefix=prefix,
                actions=actions,
                base_signature=adapter.signature(replay.state, mode=args.mode),
                mode=args.mode,
            )
            pairs = sample_pairs(actions, args.max_pairs_per_prefix, rng, effects, args.pair_filter)
            for action_a, action_b in pairs:
                if args.max_records is not None and total >= args.max_records:
                    break
                record = probe.probe_pair(
                    task_id=str(args.game_file),
                    seed=args.seed,
                    prefix_actions=prefix,
                    action_a=action_a,
                    action_b=action_b,
                )
                handle.write(record.to_json() + "\n")
                total += 1
                for relation, value in record.relations.items():
                    stats[f"{relation}={value}"] += 1
                if args.progress_every > 0 and total % args.progress_every == 0:
                    print(f"records={total}")
            if args.max_records is not None and total >= args.max_records:
                break

    print(f"prefixes={len(prefixes)} records={total} out={args.out}")
    for key, count in sorted(stats.items()):
        print(f"{key}: {count}")


def collect_prefixes(
    adapter: TextWorldAdapter,
    game_file: str,
    seed: int,
    rng: random.Random,
    num_prefixes: int,
    max_depth: int,
    max_attempts: int,
) -> tuple[tuple[str, ...], ...]:
    prefixes: list[tuple[str, ...]] = [()]
    seen = {()}
    prefix: tuple[str, ...] = ()
    attempts = 0

    while len(prefixes) < num_prefixes and attempts < max_attempts:
        attempts += 1
        replay = adapter.replay(game_file, prefix, seed)
        actions = [
            action
            for action in sorted(adapter.admissible_actions(replay.state))
            if action not in {"look", "inventory"}
        ]

        if replay.done or len(prefix) >= max_depth or not actions:
            prefix = rng.choice(prefixes)
            continue

        prefix = prefix + (rng.choice(actions),)
        if prefix not in seen:
            prefixes.append(prefix)
            seen.add(prefix)
        elif len(prefix) >= max_depth:
            prefix = rng.choice(prefixes)

    if len(prefixes) < num_prefixes:
        raise RuntimeError(
            f"collected {len(prefixes)} prefixes, requested {num_prefixes}; "
            f"increase --max-prefix-attempts or reduce --num-prefixes"
        )

    return tuple(prefixes)


def check_deterministic_replay(
    adapter: TextWorldAdapter,
    game_file: str,
    seed: int,
    prefixes: tuple[tuple[str, ...], ...],
    mode: str,
) -> None:
    for prefix in prefixes:
        first = adapter.replay(game_file, prefix, seed)
        second = adapter.replay(game_file, prefix, seed)
        sig_first = adapter.signature(first.state, mode=mode)
        sig_second = adapter.signature(second.state, mode=mode)
        if sig_first != sig_second or first.valid != second.valid:
            raise RuntimeError(f"non-deterministic replay for prefix: {prefix}")


def sample_pairs(
    actions: tuple[str, ...],
    max_pairs: int,
    rng: random.Random,
    effects: dict[str, bool],
    pair_filter: str,
) -> tuple[tuple[str, str], ...]:
    all_pairs = [
        (action_a, action_b)
        for action_a, action_b in itertools.product(actions, actions)
        if pair_matches_filter(action_a, action_b, effects, pair_filter)
    ]
    if len(all_pairs) <= max_pairs:
        return tuple(all_pairs)
    return tuple(rng.sample(all_pairs, max_pairs))


def action_effects(
    adapter: TextWorldAdapter,
    game_file: str,
    seed: int,
    prefix: tuple[str, ...],
    actions: tuple[str, ...],
    base_signature: object,
    mode: str,
) -> dict[str, bool]:
    effects: dict[str, bool] = {}
    for action in actions:
        replay = adapter.replay(game_file, prefix + (action,), seed)
        effects[action] = adapter.signature(replay.state, mode=mode) != base_signature
    return effects


def pair_matches_filter(
    action_a: str,
    action_b: str,
    effects: dict[str, bool],
    pair_filter: str,
) -> bool:
    effect_a = effects[action_a]
    effect_b = effects[action_b]
    if pair_filter == "all":
        return True
    if pair_filter == "one-effectful":
        return effect_a or effect_b
    if pair_filter == "both-effectful":
        return effect_a and effect_b
    raise ValueError(f"unknown pair filter: {pair_filter}")


if __name__ == "__main__":
    main()
