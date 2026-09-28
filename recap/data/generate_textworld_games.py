from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic TextWorld game files.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/textworld_games"))
    parser.add_argument("--prefix", default="recap_seed")
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--nb-objects", type=int, default=8)
    parser.add_argument("--quest-length", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        output = args.out_dir / f"{args.prefix}{seed}.z8"
        if output.exists() and not args.force:
            print(f"exists seed={seed} file={output}", flush=True)
            continue
        generate_one(
            output=output,
            seed=seed,
            world_size=args.world_size,
            nb_objects=args.nb_objects,
            quest_length=args.quest_length,
            force=args.force,
        )


def generate_one(
    output: Path,
    seed: int,
    world_size: int,
    nb_objects: int,
    quest_length: int,
    force: bool,
) -> None:
    cmd = [
        "tw-make",
        "custom",
        "--world-size",
        str(world_size),
        "--nb-objects",
        str(nb_objects),
        "--quest-length",
        str(quest_length),
        "--seed",
        str(seed),
        "--output",
        str(output),
        "--silent",
    ]
    if force:
        cmd.append("--force")
    print(
        f"generate seed={seed} world_size={world_size} "
        f"nb_objects={nb_objects} quest_length={quest_length} file={output}",
        flush=True,
    )
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
