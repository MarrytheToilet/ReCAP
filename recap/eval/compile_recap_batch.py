from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Mapping

from recap.envs.factory import build_adapter
from recap.eval.build_action_preferences import build_recap_dataset, summarize_preferences
from recap.eval.eval_recap_failure_decomposition import summarize_decomposition
from recap.probe.trace_edit_probe import TraceEditConfig, TraceEditProbe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile ReCAP preferences and step ledger from batch trajectories."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument(
        "--env",
        choices=["auto", "textworld", "alfworld", "scienceworld"],
        default="auto",
    )
    parser.add_argument("--run-key", default=None)
    parser.add_argument("--trace-report", type=Path, default=None)
    parser.add_argument("--out-trace-report", type=Path, default=None)
    parser.add_argument("--equivalence-mode", default="full")
    parser.add_argument("--trace-progress", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel replay workers for generating trace reports when --trace-report is not provided.",
    )
    parser.add_argument(
        "--candidate-branch-repairs",
        action="store_true",
        help=(
            "Replay each logged non-executed candidate followed by the environment "
            "suffix policy. Useful for external benchmarks where the suffix first "
            "action is not itself a clean local repair."
        ),
    )
    parser.add_argument(
        "--suffix-match-repairs",
        action="store_true",
        help=(
            "Search the verified suffix for logged candidates and replay from the "
            "matched suffix point. This is faster than full candidate branch search "
            "and useful for external pilots."
        ),
    )
    parser.add_argument("--candidate-branch-limit", type=int, default=10)
    parser.add_argument("--suffix-match-limit-per-step", type=int, default=3)
    parser.add_argument(
        "--source",
        choices=["policy-repair", "gold", "both"],
        default="policy-repair",
    )
    parser.add_argument("--max-repair-suffix-len", type=int, default=None)
    parser.add_argument(
        "--out-preferences",
        type=Path,
        default=Path("analysis/recap_preferences.jsonl"),
    )
    parser.add_argument(
        "--out-ledger",
        type=Path,
        default=Path("analysis/recap_compilation_ledger.jsonl"),
    )
    parser.add_argument(
        "--out-trajectory-ledger",
        type=Path,
        default=Path("analysis/recap_trajectory_ledger.jsonl"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("analysis/recap_compile_summary.json"),
    )
    args = parser.parse_args()

    episodes = load_episodes(args.trajectories, run_key=args.run_key)
    run_payload = {"episodes": episodes}
    report_payload = build_report_payload(
        episodes=episodes,
        trace_report=args.trace_report,
        out_trace_report=args.out_trace_report,
        source=args.source,
        env_name=args.env,
        equivalence_mode=args.equivalence_mode,
        suffix_match_repairs=args.suffix_match_repairs,
        candidate_branch_repairs=args.candidate_branch_repairs,
        candidate_branch_limit=args.candidate_branch_limit,
        suffix_match_limit_per_step=args.suffix_match_limit_per_step,
        progress=args.trace_progress,
        workers=args.workers,
    )
    preferences, step_ledger, trajectory_ledger = build_recap_dataset(
        run_payload=run_payload,
        report_payload=report_payload,
        source=args.source,
        max_repair_suffix_len=args.max_repair_suffix_len,
    )

    write_jsonl(args.out_preferences, preferences)
    write_jsonl(args.out_ledger, step_ledger)
    write_jsonl(args.out_trajectory_ledger, trajectory_ledger)

    summary = {
        "preferences": summarize_preferences(preferences),
        "failure_decomposition": summarize_decomposition(
            tuple(step_ledger),
            tuple(preferences),
        ),
        "outputs": {
            "preferences": str(args.out_preferences),
            "ledger": str(args.out_ledger),
            "trajectory_ledger": str(args.out_trajectory_ledger),
            "trace_report": str(args.out_trace_report) if args.out_trace_report is not None else None,
        },
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary["failure_decomposition"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"preferences={args.out_preferences}")
    print(f"ledger={args.out_ledger}")
    print(f"summary={args.out_summary}")


def load_episodes(path: Path, run_key: str | None = None) -> list[Mapping[str, Any]]:
    payload = load_json_or_jsonl(path)
    if run_key is not None:
        if not isinstance(payload, Mapping):
            raise ValueError("--run-key requires a JSON object input")
        payload = payload[run_key]
    return extract_episodes(payload)


def load_reports(path: Path) -> list[Mapping[str, Any]]:
    return extract_reports(load_json_or_jsonl(path))


def build_report_payload(
    episodes: list[Mapping[str, Any]],
    trace_report: Path | None,
    out_trace_report: Path | None,
    source: str,
    env_name: str = "auto",
    equivalence_mode: str = "full",
    suffix_match_repairs: bool = False,
    candidate_branch_repairs: bool = False,
    candidate_branch_limit: int = 10,
    suffix_match_limit_per_step: int = 3,
    progress: bool = False,
    workers: int = 1,
) -> dict[str, Any] | None:
    if trace_report is not None:
        return {"reports": load_reports(trace_report)}
    if source not in {"policy-repair", "both"}:
        return None

    reports = probe_episodes(
        episodes=episodes,
        env_name=env_name,
        equivalence_mode=equivalence_mode,
        suffix_match_repairs=suffix_match_repairs,
        candidate_branch_repairs=candidate_branch_repairs,
        candidate_branch_limit=candidate_branch_limit,
        suffix_match_limit_per_step=suffix_match_limit_per_step,
        progress=progress,
        workers=workers,
    )
    payload = {
        "summary": {
            "episodes": len(reports),
            "generated_by": "compile_recap_batch",
            "probe_policy_repairs": True,
        },
        "reports": reports,
    }
    if out_trace_report is not None:
        out_trace_report.parent.mkdir(parents=True, exist_ok=True)
        out_trace_report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def probe_episodes(
    episodes: list[Mapping[str, Any]],
    env_name: str = "auto",
    equivalence_mode: str = "full",
    suffix_match_repairs: bool = False,
    candidate_branch_repairs: bool = False,
    candidate_branch_limit: int = 10,
    suffix_match_limit_per_step: int = 3,
    progress: bool = False,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers <= 1:
        reports = []
        adapter = build_adapter(resolve_env_name(episodes, env_name))
        probe = TraceEditProbe(
            adapter=adapter,
            config=trace_edit_config(
                equivalence_mode,
                suffix_match_repairs=suffix_match_repairs,
                candidate_branch_repairs=candidate_branch_repairs,
                candidate_branch_limit=candidate_branch_limit,
                suffix_match_limit_per_step=suffix_match_limit_per_step,
            ),
        )
        try:
            for index, episode in enumerate(episodes):
                report = to_jsonable(probe.probe_episode(episode))
                reports.append(report)
                if progress:
                    print_trace_progress(index + 1, len(episodes), report)
        finally:
            adapter.close()
        return reports

    reports: list[dict[str, Any] | None] = [None] * len(episodes)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                probe_one_episode,
                dict(episode),
                equivalence_mode,
                env_name,
                suffix_match_repairs,
                candidate_branch_repairs,
                candidate_branch_limit,
                suffix_match_limit_per_step,
            ): index
            for index, episode in enumerate(episodes)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            report = future.result()
            reports[index] = report
            completed += 1
            if progress:
                print_trace_progress(completed, len(episodes), report, input_index=index)
    return [report for report in reports if report is not None]


def probe_one_episode(
    episode: Mapping[str, Any],
    equivalence_mode: str = "full",
    env_name: str = "auto",
    suffix_match_repairs: bool = False,
    candidate_branch_repairs: bool = False,
    candidate_branch_limit: int = 10,
    suffix_match_limit_per_step: int = 3,
) -> dict[str, Any]:
    adapter = build_adapter(resolve_env_name([episode], env_name))
    probe = TraceEditProbe(
        adapter=adapter,
        config=trace_edit_config(
            equivalence_mode,
            suffix_match_repairs=suffix_match_repairs,
            candidate_branch_repairs=candidate_branch_repairs,
            candidate_branch_limit=candidate_branch_limit,
            suffix_match_limit_per_step=suffix_match_limit_per_step,
        ),
    )
    try:
        return to_jsonable(probe.probe_episode(episode))
    finally:
        adapter.close()


def resolve_env_name(episodes: list[Mapping[str, Any]], env_name: str) -> str:
    if env_name != "auto":
        return env_name
    for episode in episodes:
        value = episode.get("env")
        if value:
            return str(value)
    return "textworld"


def trace_edit_config(
    equivalence_mode: str,
    suffix_match_repairs: bool = False,
    candidate_branch_repairs: bool = False,
    candidate_branch_limit: int = 10,
    suffix_match_limit_per_step: int = 3,
) -> TraceEditConfig:
    return TraceEditConfig(
        equivalence_mode=equivalence_mode,
        probe_deletions=False,
        probe_swaps=False,
        probe_replacements=False,
        probe_policy_repairs=True,
        probe_suffix_match_repairs=suffix_match_repairs,
        probe_candidate_policy_repairs=candidate_branch_repairs,
        candidate_branch_limit=candidate_branch_limit,
        suffix_match_limit_per_step=suffix_match_limit_per_step,
    )


def print_trace_progress(
    completed: int,
    total: int,
    report: Mapping[str, Any],
    input_index: int | None = None,
) -> None:
    metrics = dict(report.get("metrics", {}))
    index_text = f" index={input_index}" if input_index is not None else ""
    print(
        f"trace_probe={completed}/{total}{index_text} "
        f"task={Path(str(report.get('task_id', 'unknown'))).stem} "
        f"success={report.get('original_success')} "
        f"policy_repairs={float(metrics.get('policy_repair_candidates', 0.0)):.0f}",
        flush=True,
    )


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def extract_episodes(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if "episodes" in payload:
            return [record for record in payload["episodes"]]
        if "episode" in payload:
            return [payload["episode"]]
        if "task_id" in payload and "steps" in payload:
            return [payload]
        episode_containers = [
            value
            for value in payload.values()
            if isinstance(value, Mapping) and "episodes" in value
        ]
        if len(episode_containers) == 1:
            return [record for record in episode_containers[0]["episodes"]]
    if isinstance(payload, list):
        episodes: list[Mapping[str, Any]] = []
        for record in payload:
            episodes.extend(extract_episodes(record))
        return episodes
    raise ValueError("could not find episodes in trajectory input")


def extract_reports(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if "reports" in payload:
            return [record for record in payload["reports"]]
        if "report" in payload:
            return [payload["report"]]
        if "task_id" in payload and "edits" in payload:
            return [payload]
        report_containers = [
            value
            for value in payload.values()
            if isinstance(value, Mapping) and "reports" in value
        ]
        if len(report_containers) == 1:
            return [record for record in report_containers[0]["reports"]]
    if isinstance(payload, list):
        reports: list[Mapping[str, Any]] = []
        for record in payload:
            reports.extend(extract_reports(record))
        return reports
    raise ValueError("could not find reports in trace-report input")


def write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


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
