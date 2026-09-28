from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and probe multiple TextWorld games.")
    parser.add_argument("--game-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--game-dir", type=Path, default=Path("data/textworld_games"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/textworld_runs"))
    parser.add_argument("--merged-out", type=Path, default=Path("data/textworld_multi_seed.jsonl"))
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--nb-objects", type=int, default=8)
    parser.add_argument("--quest-length", type=int, default=3)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--num-prefixes", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-pairs-per-prefix", type=int, default=10)
    parser.add_argument("--max-records-per-game", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--pair-filter",
        choices=["all", "one-effectful", "both-effectful"],
        default="both-effectful",
    )
    parser.add_argument("--force-games", action="store_true")
    args = parser.parse_args()

    args.game_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.workers == 1:
        per_seed_outputs = [build_one_seed(game_seed, args) for game_seed in args.game_seeds]
    else:
        per_seed_outputs = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(build_one_seed, game_seed, args): game_seed
                for game_seed in args.game_seeds
            }
            for future in as_completed(futures):
                game_seed = futures[future]
                output_file = future.result()
                print(f"done_seed={game_seed} out={output_file}", flush=True)
                per_seed_outputs.append(output_file)

        output_by_seed = {
            int(path.stem.removeprefix("relations_seed")): path
            for path in per_seed_outputs
        }
        per_seed_outputs = [output_by_seed[game_seed] for game_seed in args.game_seeds]

    merge_jsonl(per_seed_outputs, args.merged_out)
    print(f"merged={args.merged_out} files={len(per_seed_outputs)}")


def build_one_seed(game_seed: int, args: argparse.Namespace) -> Path:
    game_file = args.game_dir / f"recap_seed{game_seed}.z8"
    output_file = args.out_dir / f"relations_seed{game_seed}.jsonl"
    ensure_game(game_file, game_seed, args)
    probe_game(game_file, output_file, args)
    return output_file


def ensure_game(game_file: Path, game_seed: int, args: argparse.Namespace) -> None:
    if game_file.exists() and not args.force_games:
        print(f"game_exists seed={game_seed} file={game_file}", flush=True)
        return

    cmd = [
        "tw-make",
        "custom",
        "--world-size",
        str(args.world_size),
        "--nb-objects",
        str(args.nb_objects),
        "--quest-length",
        str(args.quest_length),
        "--seed",
        str(game_seed),
        "--output",
        str(game_file),
        "--force",
        "--silent",
    ]
    print(f"generate_game seed={game_seed} file={game_file}", flush=True)
    subprocess.run(cmd, check=True)


def probe_game(game_file: Path, output_file: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "recap.data.build_textworld_relation_dataset",
        str(game_file),
        "--out",
        str(output_file),
        "--seed",
        str(args.probe_seed),
        "--num-prefixes",
        str(args.num_prefixes),
        "--max-depth",
        str(args.max_depth),
        "--max-pairs-per-prefix",
        str(args.max_pairs_per_prefix),
        "--max-records",
        str(args.max_records_per_game),
        "--pair-filter",
        args.pair_filter,
        "--progress-every",
        "0",
    ]
    print(f"probe_game file={game_file} out={output_file}", flush=True)
    subprocess.run(cmd, check=True)


def merge_jsonl(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for path in paths:
            with path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    if line.strip():
                        writer.write(line)


if __name__ == "__main__":
    main()
