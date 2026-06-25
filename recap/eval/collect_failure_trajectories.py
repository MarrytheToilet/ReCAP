from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Mapping

from recap.agents import LLMConfig, MockLLMAgent, NoisyCandidateAgent, OpenAIChatAgent, RandomAgent
from recap.agents.llm_agent import load_env_file
from recap.envs.factory import build_adapter, default_task_dir, default_task_glob
from recap.eval.agent_loop import AgentEpisodeResult, run_episode
from recap.eval.eval_agent import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect failed TextWorld trajectories with logged top-k candidates."
    )
    parser.add_argument("--env", choices=["textworld", "alfworld"], default="textworld")
    parser.add_argument("--difficulty", default="xhard")
    parser.add_argument("--game-dir", type=Path, default=None)
    parser.add_argument("--game-glob", default=None)
    parser.add_argument("--game-list", type=Path, default=None)
    parser.add_argument("--min-game-seed", type=int, default=None)
    parser.add_argument("--max-game-seed", type=int, default=None)
    parser.add_argument("--num-games", type=int, default=None)
    parser.add_argument("--rollouts-per-game", type=int, default=1)
    parser.add_argument("--rollout-seed-start", type=int, default=0)
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--agent", choices=["random", "mock-llm", "openai"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--candidate-noise",
        choices=["none", "frontload-existing-structural", "frontload-structural"],
        default="none",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=4)
    parser.add_argument("--llm-retry-base-delay", type=float, default=2.0)
    parser.add_argument("--llm-call-delay", type=float, default=0.0)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--include-successes", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--write-errors", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    load_env_file(args.env_file)
    games = select_games(
        game_dir=args.game_dir or default_game_dir(args.env, args.difficulty),
        game_glob=args.game_glob or default_task_glob(args.env),
        game_list=args.game_list,
        min_game_seed=args.min_game_seed,
        max_game_seed=args.max_game_seed,
        num_games=args.num_games,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    error_count = 0
    tasks = collection_tasks(args, games)
    attempted = len(tasks)
    with args.out.open("w", encoding="utf-8") as writer:
        for completed, record in enumerate(run_collection_tasks(args, tasks), start=1):
            if record.get("error"):
                error_count += 1
            should_write = (
                (args.include_successes or not bool(record.get("success", False)))
                and (args.write_errors or not record.get("error"))
            )
            if should_write:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
                records.append(record)
            if args.progress:
                print(
                    f"episode={completed}/{attempted} game={record.get('task_id')} "
                    f"success={record.get('success')} steps={record.get('num_steps')} "
                    f"written={len(records)}",
                    flush=True,
                )

    summary = {
        "attempted_trajectories": attempted,
        "written_trajectories": len(records),
        "failed_trajectories": sum(1 for record in records if not record.get("success", False)),
        "successful_trajectories": sum(1 for record in records if record.get("success", False)),
        "errored_trajectories": error_count,
        "games": len(games),
        "rollouts_per_game": args.rollouts_per_game,
        "out": str(args.out),
    }
    summary_out = args.summary_out or args.out.with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")
    print(f"summary={summary_out}")


def collection_tasks(
    args: argparse.Namespace,
    games: tuple[Path, ...],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for game_index, game in enumerate(games):
        for rollout_index in range(args.rollouts_per_game):
            tasks.append(
                {
                    "game": game,
                    "rollout_index": rollout_index,
                    "rollout_seed": args.rollout_seed_start + rollout_index,
                    "agent_seed": (
                        args.agent_seed
                        + game_index * args.rollouts_per_game
                        + rollout_index
                    ),
                }
            )
    return tasks


def run_collection_tasks(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    if args.workers <= 1:
        for task in tasks:
            yield collect_one(args, task)
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_one, args, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def collect_one(args: argparse.Namespace, task: Mapping[str, Any]) -> dict[str, Any]:
    game = Path(task["game"])
    rollout_index = int(task["rollout_index"])
    rollout_seed = int(task["rollout_seed"])
    agent_seed = int(task["agent_seed"])
    adapter = build_adapter(args.env)
    try:
        agent = build_collection_agent(args, agent_seed=agent_seed)
        result = run_episode(
            adapter=adapter,
            agent=agent,
            task_id=str(game),
            seed=rollout_seed,
            max_steps=args.max_steps,
        )
        return trajectory_record(
            result=result,
            difficulty=args.difficulty,
            env_name=args.env,
            game=game,
            rollout_index=rollout_index,
            agent_name=args.agent,
            model=args.model,
            temperature=args.temperature,
            top_k=args.top_k,
            agent_seed=agent_seed,
        )
    except Exception as exc:
        if not args.continue_on_error:
            raise
        return error_record(
            game=game,
            difficulty=args.difficulty,
            env_name=args.env,
            rollout_seed=rollout_seed,
            rollout_index=rollout_index,
            agent_name=args.agent,
            model=args.model,
            temperature=args.temperature,
            top_k=args.top_k,
            agent_seed=agent_seed,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        adapter.close()


def build_collection_agent(args: argparse.Namespace, agent_seed: int) -> Any:
    if args.agent == "random":
        agent = RandomAgent(seed=agent_seed, max_candidates=args.top_k)
    elif args.agent == "mock-llm":
        agent = MockLLMAgent(max_candidates=args.top_k)
    elif args.agent == "openai":
        model = args.model or os.environ.get("RECAP_LLM_MODEL")
        if model is None:
            raise SystemExit("--model or RECAP_LLM_MODEL is required when --agent openai")
        agent = OpenAIChatAgent(
            LLMConfig(
                model=model,
                max_candidates=args.top_k,
                temperature=args.temperature,
                timeout=args.llm_timeout,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                max_retries=args.llm_max_retries,
                retry_base_delay=args.llm_retry_base_delay,
                call_delay=args.llm_call_delay,
                system_prompt=collection_system_prompt(args),
            )
        )
    else:
        raise ValueError(f"unknown agent: {args.agent}")

    if args.candidate_noise != "none":
        agent = NoisyCandidateAgent(
            agent,
            mode=args.candidate_noise,
            max_candidates=args.top_k,
        )
    return agent


def collection_system_prompt(args: argparse.Namespace) -> str:
    path = getattr(args, "system_prompt_file", None)
    if path is not None:
        prompt_path = Path(path)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
    return LLMConfig.__dataclass_fields__["system_prompt"].default


def trajectory_record(
    result: AgentEpisodeResult,
    difficulty: str,
    env_name: str,
    game: Path,
    rollout_index: int,
    agent_name: str,
    model: str | None,
    temperature: float,
    top_k: int,
    agent_seed: int,
) -> dict[str, Any]:
    record = to_jsonable(result)
    game_seed = infer_seed(game)
    record_game_id = game_id(game)
    record.update(
        {
            "trajectory_id": f"{record_game_id}:seed{result.seed}:rollout{rollout_index}",
            "env": env_name,
            "difficulty": difficulty,
            "game_id": record_game_id,
            "game_seed": game_seed,
            "rollout_seed": result.seed,
            "agent": agent_name,
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
            "agent_seed": agent_seed,
            "num_steps": len(result.steps),
            "final_score": result.total_reward,
        }
    )
    return record


def error_record(
    game: Path,
    difficulty: str,
    env_name: str,
    rollout_seed: int,
    rollout_index: int,
    agent_name: str,
    model: str | None,
    temperature: float,
    top_k: int,
    agent_seed: int,
    error: str,
) -> dict[str, Any]:
    game_seed = infer_seed(game)
    record_game_id = game_id(game)
    return {
        "trajectory_id": f"{record_game_id}:seed{rollout_seed}:rollout{rollout_index}",
        "task_id": str(game),
        "seed": rollout_seed,
        "success": False,
        "total_reward": 0.0,
        "steps": [],
        "error": error,
        "env": env_name,
        "difficulty": difficulty,
        "game_id": record_game_id,
        "game_seed": game_seed,
        "rollout_seed": rollout_seed,
        "agent": agent_name,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "agent_seed": agent_seed,
        "num_steps": 0,
        "final_score": 0.0,
    }


def select_games(
    game_dir: Path,
    game_glob: str,
    num_games: int | None,
    game_list: Path | None = None,
    min_game_seed: int | None = None,
    max_game_seed: int | None = None,
) -> tuple[Path, ...]:
    if game_list is not None:
        games = tuple(
            Path(line.strip())
            for line in game_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        games = tuple(sorted(game_dir.glob(game_glob)))
    if min_game_seed is not None:
        games = tuple(game for game in games if (infer_seed(game) or -1) >= min_game_seed)
    if max_game_seed is not None:
        games = tuple(game for game in games if (infer_seed(game) or 10**12) <= max_game_seed)
    if num_games is not None:
        games = games[:num_games]
    if not games:
        raise FileNotFoundError(f"no games matched {game_dir / game_glob}")
    return games


def default_game_dir(env_name: str, difficulty: str | None = None) -> Path:
    if difficulty is None:
        return default_task_dir("textworld", env_name)
    return default_task_dir(env_name, difficulty)


def infer_seed(game: Path) -> int | None:
    match = re.search(r"seed(\d+)", game.stem)
    return int(match.group(1)) if match else None


def game_id(game: Path) -> str:
    if game.name == "game.tw-pddl":
        return "/".join(game.parent.parts[-3:])
    return game.stem


if __name__ == "__main__":
    main()
