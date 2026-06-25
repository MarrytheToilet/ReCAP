from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recap.controllers import PriorController, ControllerConfig
from recap.envs.textworld_adapter import TextWorldAdapter
from recap.rl import QLearningConfig, train_q_learning


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tabular Q-learning with optional ReCAP priors.")
    parser.add_argument("games", type=Path, nargs="+")
    parser.add_argument("--prior", choices=["none", "prior-soft", "prior-hard"], default="none")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--fast-controller", action="store_true")
    parser.add_argument("--noop-penalty", type=float, default=6.0)
    parser.add_argument("--seen-state-penalty", type=float, default=4.0)
    parser.add_argument("--recent-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--absorbed-penalty", type=float, default=3.0)
    parser.add_argument("--inverse-penalty", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("analysis/rl_eval.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    all_runs = []
    for game in args.games:
        for training_seed in args.training_seeds:
            adapter = TextWorldAdapter()
            controller = (
                PriorController(
                    adapter=adapter,
                    env_name="textworld",
                    config=ControllerConfig(
                        noop_penalty=args.noop_penalty,
                        seen_state_penalty=args.seen_state_penalty,
                        recent_repeat_penalty=args.recent_repeat_penalty,
                        absorbed_penalty=args.absorbed_penalty,
                        inverse_penalty=args.inverse_penalty,
                        enable_pair_penalties=not args.fast_controller,
                    ),
                )
                if args.prior != "none"
                else None
            )
            traces = train_q_learning(
                adapter=adapter,
                task_id=str(game),
                config=QLearningConfig(
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon=args.epsilon,
                    epsilon_decay=args.epsilon_decay,
                    min_epsilon=args.min_epsilon,
                    seed=training_seed,
                    prior=args.prior,
                ),
                controller=controller,
            )
            run = {
                "task_id": str(game),
                "training_seed": training_seed,
                "summary": summarize_traces(traces),
                "episodes": traces,
                "controller_cache_hits": dict(controller.cache_hits) if controller else {},
                "controller_cache_misses": dict(controller.cache_misses) if controller else {},
            }
            all_runs.append(run)
            if args.progress:
                print(
                    f"game={game} seed={training_seed} "
                    f"first_success={run['summary']['first_success_episode']} "
                    f"success_rate={run['summary']['success_rate']:.3f}",
                    flush=True,
                )
            adapter.close()

    payload = {
        "summary": summarize_runs(all_runs, prior=args.prior),
        "runs": all_runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(to_jsonable(payload["summary"]), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def summarize_traces(traces: tuple[Any, ...]) -> dict[str, Any]:
    episodes = len(traces)
    successes = [trace for trace in traces if trace.success]
    step_counts = [len(trace.steps) for trace in traces]
    noop_or_seen = sum(
        1
        for trace in traces
        for step in trace.steps
        if "noop" in step.recap_reasons or "seen_state" in step.recap_reasons
    )
    blocked_actions = sum(
        len(step.blocked_actions)
        for trace in traces
        for step in trace.steps
    )
    total_steps = sum(step_counts)
    first_success = next((trace.episode_index for trace in traces if trace.success), None)
    last_window = traces[-min(20, episodes) :] if episodes else ()
    return {
        "episodes": episodes,
        "success_rate": len(successes) / episodes if episodes else 0.0,
        "last_window_success_rate": (
            sum(trace.success for trace in last_window) / len(last_window)
            if last_window
            else 0.0
        ),
        "first_success_episode": first_success,
        "avg_return": sum(trace.total_reward for trace in traces) / episodes if episodes else 0.0,
        "avg_steps": sum(step_counts) / episodes if episodes else 0.0,
        "noop_or_seen_selected_rate": noop_or_seen / total_steps if total_steps else 0.0,
        "avg_blocked_actions_per_step": blocked_actions / total_steps if total_steps else 0.0,
    }


def summarize_runs(runs: list[dict[str, Any]], prior: str) -> dict[str, Any]:
    summaries = [run["summary"] for run in runs]
    return {
        "prior": prior,
        "runs": len(runs),
        "avg_success_rate": average(summary["success_rate"] for summary in summaries),
        "avg_last_window_success_rate": average(
            summary["last_window_success_rate"] for summary in summaries
        ),
        "avg_first_success_episode": average(
            summary["first_success_episode"]
            for summary in summaries
            if summary["first_success_episode"] is not None
        ),
        "avg_return": average(summary["avg_return"] for summary in summaries),
        "avg_steps": average(summary["avg_steps"] for summary in summaries),
        "avg_noop_or_seen_selected_rate": average(
            summary["noop_or_seen_selected_rate"] for summary in summaries
        ),
        "avg_blocked_actions_per_step": average(
            summary["avg_blocked_actions_per_step"] for summary in summaries
        ),
    }


def average(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_jsonable(getattr(value, key))
            for key in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
