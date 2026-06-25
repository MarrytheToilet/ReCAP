from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recap.envs.scienceworld_adapter import ScienceWorldAdapter
from recap.eval.agent_loop import run_episode
from recap.eval.collect_failure_trajectories import build_collection_agent
from recap.eval.eval_agent import to_jsonable
from recap.agents.llm_agent import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect ScienceWorld trajectories with logged candidate lists."
    )
    parser.add_argument("--task-names", default="boil")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--variation-start", type=int, default=0)
    parser.add_argument("--num-variations", type=int, default=5)
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--agent", choices=["random", "mock-llm", "openai"], default="mock-llm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=4)
    parser.add_argument("--llm-retry-base-delay", type=float, default=2.0)
    parser.add_argument("--llm-call-delay", type=float, default=0.0)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--candidate-noise",
        choices=["none", "frontload-existing-structural", "frontload-structural"],
        default="none",
    )
    parser.add_argument("--include-successes", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    load_env_file(args.env_file)
    tasks = scienceworld_tasks(args.task_names, args.num_tasks)
    task_ids = [
        scienceworld_task_id(task, variation, args.simplification)
        for task in tasks
        for variation in range(args.variation_start, args.variation_start + args.num_variations)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    errors = 0
    adapter = ScienceWorldAdapter(env_step_limit=max(args.max_steps + 5, 50))
    try:
        with args.out.open("w", encoding="utf-8") as writer:
            for index, task_id in enumerate(task_ids):
                try:
                    agent = build_collection_agent(args, agent_seed=args.agent_seed + index)
                    result = run_episode(
                        adapter=adapter,
                        agent=agent,
                        task_id=task_id,
                        seed=0,
                        max_steps=args.max_steps,
                        equivalence_mode="goal",
                    )
                    record = to_jsonable(result)
                    record.update(
                        {
                            "trajectory_id": f"{task_id}:agentseed{args.agent_seed + index}",
                            "env": "scienceworld",
                            "difficulty": args.simplification,
                            "game_id": task_id,
                            "game_seed": None,
                            "rollout_seed": 0,
                            "agent": args.agent,
                            "model": args.model,
                            "temperature": args.temperature,
                            "top_k": args.top_k,
                            "agent_seed": args.agent_seed + index,
                            "num_steps": len(result.steps),
                            "final_score": result.total_reward,
                        }
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    errors += 1
                    record = {
                        "trajectory_id": f"{task_id}:agentseed{args.agent_seed + index}",
                        "task_id": task_id,
                        "seed": 0,
                        "success": False,
                        "total_reward": 0.0,
                        "steps": [],
                        "error": f"{type(exc).__name__}: {exc}",
                        "env": "scienceworld",
                        "difficulty": args.simplification,
                        "game_id": task_id,
                        "rollout_seed": 0,
                        "agent": args.agent,
                        "model": args.model,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "agent_seed": args.agent_seed + index,
                        "num_steps": 0,
                        "final_score": 0.0,
                    }
                should_write = args.include_successes or not bool(record.get("success", False))
                if should_write:
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    writer.flush()
                    records.append(record)
                if args.progress:
                    print(
                        f"episode={index + 1}/{len(task_ids)} task={task_id} "
                        f"success={record.get('success')} steps={record.get('num_steps')} "
                        f"written={len(records)}",
                        flush=True,
                    )
    finally:
        adapter.close()

    summary = {
        "attempted_trajectories": len(task_ids),
        "written_trajectories": len(records),
        "failed_trajectories": sum(1 for record in records if not record.get("success", False)),
        "successful_trajectories": sum(1 for record in records if record.get("success", False)),
        "errored_trajectories": errors,
        "tasks": tasks,
        "num_variations": args.num_variations,
        "out": str(args.out),
    }
    summary_out = args.summary_out or args.out.with_suffix(".summary.json")
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")
    print(f"summary={summary_out}")


def scienceworld_tasks(task_names: str, num_tasks: int | None) -> list[str]:
    if task_names == "all":
        from scienceworld import ScienceWorldEnv

        env = ScienceWorldEnv(envStepLimit=1)
        try:
            names = list(env.get_task_names())
        finally:
            env.close()
    else:
        names = [name.strip() for name in task_names.split(",") if name.strip()]
    return names[:num_tasks] if num_tasks is not None else names


def scienceworld_task_id(task_name: str, variation: int, simplification: str) -> str:
    return f"scienceworld://{task_name}/{variation}/{simplification}"


if __name__ == "__main__":
    main()
