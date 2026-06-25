from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from recap.agents import TraceFeedbackAgent
from recap.agents.llm_agent import load_env_file
from recap.controllers import PriorController
from recap.envs.textworld_adapter import TextWorldAdapter
from recap.eval.agent_loop import AgentEpisodeResult, run_episode
from recap.eval.eval_agent import (
    build_agent,
    build_controller_config,
    make_error_result,
    summarize,
    to_jsonable,
)
from recap.probe.trace_edit_probe import TraceEditConfig, TraceEditProbe, TraceEditReport
from recap.probe.trace_feedback import feedback_from_report, include_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a two-pass agent evaluation with counterfactual trace feedback."
    )
    parser.add_argument("games", type=Path, nargs="+")
    parser.add_argument("--agent", choices=["mock-llm", "openai"], default="openai")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument(
        "--candidate-noise",
        choices=["none", "frontload-existing-structural", "frontload-structural"],
        default="none",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm-max-retries", type=int, default=4)
    parser.add_argument("--llm-retry-base-delay", type=float, default=2.0)
    parser.add_argument("--llm-call-delay", type=float, default=0.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--first-controller", choices=["none", "prior"], default="none")
    parser.add_argument("--second-controller", choices=["none", "prior"], default="none")
    parser.add_argument("--fast-controller", action="store_true")
    parser.add_argument("--noop-penalty", type=float, default=6.0)
    parser.add_argument("--seen-state-penalty", type=float, default=4.0)
    parser.add_argument("--recent-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--absorbed-penalty", type=float, default=3.0)
    parser.add_argument("--inverse-penalty", type=float, default=4.0)
    parser.add_argument("--feedback-mode", choices=["safe", "oracle"], default="safe")
    parser.add_argument("--feedback-source", choices=["all", "failures", "successes"], default="failures")
    parser.add_argument("--feedback-max-items", type=int, default=6)
    parser.add_argument("--max-replacements-per-step", type=int, default=5)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--episode-delay", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("analysis/feedback_cycle.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    first_results = run_pass(
        args=args,
        pass_name="first",
        controller_name=args.first_controller,
        feedback_by_task={},
    )
    reports = compile_reports(first_results, args)
    feedback_by_task = feedback_by_task_from_reports(
        reports=reports,
        mode=args.feedback_mode,
        max_items=args.feedback_max_items,
        source=args.feedback_source,
    )
    second_results = run_pass(
        args=args,
        pass_name="second",
        controller_name=args.second_controller,
        feedback_by_task=feedback_by_task,
    )

    first_summary = summarize(first_results, controller=args.first_controller)
    second_summary = summarize(second_results, controller=args.second_controller)
    first_summary["agent"] = args.agent
    second_summary["agent"] = args.agent
    if args.agent == "openai":
        model = args.model or os.environ.get("RECAP_LLM_MODEL")
        first_summary["model"] = model
        second_summary["model"] = model

    trace_summary = summarize_reports(reports)
    output = {
        "summary": {
            "agent": args.agent,
            "feedback_mode": args.feedback_mode,
            "feedback_source": args.feedback_source,
            "episodes": len(first_results),
            "first_success_rate": first_summary["success_rate"],
            "second_success_rate": second_summary["success_rate"],
            "success_rate_delta": second_summary["success_rate"] - first_summary["success_rate"],
            "first_avg_steps": first_summary.get("avg_steps", 0.0),
            "second_avg_steps": second_summary.get("avg_steps", 0.0),
            "avg_steps_delta": second_summary.get("avg_steps", 0.0) - first_summary.get("avg_steps", 0.0),
            "feedback_tasks": len(feedback_by_task) // 2,
        },
        "first_run": {"summary": first_summary, "episodes": first_results},
        "trace_report": {"summary": trace_summary, "reports": reports},
        "second_run": {"summary": second_summary, "episodes": second_results},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(to_jsonable(output["summary"]), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def run_pass(
    args: argparse.Namespace,
    pass_name: str,
    controller_name: str,
    feedback_by_task: dict[str, str],
) -> list[AgentEpisodeResult]:
    adapter = TextWorldAdapter()
    controller = make_controller(adapter, controller_name, args)
    results: list[AgentEpisodeResult] = []
    args.feedback_by_task = feedback_by_task
    try:
        for index, game in enumerate(args.games):
            agent = build_agent(args, index)
            if feedback_by_task and not hasattr(agent, "base_agent"):
                agent = TraceFeedbackAgent(agent, feedback_by_task)
            try:
                result = run_episode(
                    adapter=adapter,
                    agent=agent,
                    task_id=str(game),
                    seed=args.seed,
                    max_steps=args.max_steps,
                    controller=controller,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                result = make_error_result(
                    adapter=adapter,
                    task_id=str(game),
                    seed=args.seed,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            if args.progress:
                print(
                    f"{pass_name} episode={index + 1}/{len(args.games)} "
                    f"game={game} success={result.success} "
                    f"steps={len(result.steps)} reward={result.total_reward}",
                    flush=True,
                )
            if args.episode_delay > 0 and index + 1 < len(args.games):
                time.sleep(args.episode_delay)
    finally:
        adapter.close()
    return results


def make_controller(
    adapter: TextWorldAdapter,
    controller_name: str,
    args: argparse.Namespace,
) -> PriorController | None:
    if controller_name == "none":
        return None
    return PriorController(
        adapter=adapter,
        env_name="textworld",
        config=build_controller_config(args),
    )


def compile_reports(
    results: list[AgentEpisodeResult],
    args: argparse.Namespace,
) -> list[TraceEditReport]:
    adapter = TextWorldAdapter()
    probe = TraceEditProbe(
        adapter=adapter,
        config=TraceEditConfig(max_replacements_per_step=args.max_replacements_per_step),
    )
    try:
        reports = [probe.probe_episode(to_jsonable(result)) for result in results]
    finally:
        adapter.close()
    return reports


def feedback_by_task_from_reports(
    reports: list[TraceEditReport],
    mode: str,
    max_items: int,
    source: str,
) -> dict[str, str]:
    feedback_by_task: dict[str, str] = {}
    for report in reports:
        report_payload = to_jsonable(report)
        if not include_report(report_payload, source):
            continue
        feedback = feedback_from_report(report_payload, mode=mode, max_items=max_items)
        if not feedback:
            continue
        task_id = str(report.task_id)
        feedback_by_task[task_id] = feedback
        feedback_by_task[Path(task_id).stem] = feedback
    return feedback_by_task


def summarize_reports(reports: list[TraceEditReport]) -> dict[str, Any]:
    metric_keys = sorted({key for report in reports for key in report.metrics})
    return {
        "episodes": len(reports),
        "successful_episodes": sum(report.original_success for report in reports),
        "replay_successful_episodes": sum(report.replay_success for report in reports),
        **{
            key: sum(report.metrics.get(key, 0.0) for report in reports)
            for key in metric_keys
        },
    }


if __name__ == "__main__":
    main()
