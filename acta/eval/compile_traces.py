from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from acta.envs.textworld_adapter import TextWorldAdapter
from acta.probe.trace_edit_probe import TraceEditConfig, TraceEditProbe


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile agent traces with counterfactual replay edits.")
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--out", type=Path, default=Path("analysis/trace_edit_report.json"))
    parser.add_argument("--equivalence-mode", default="full")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-replacements-per-step", type=int, default=5)
    parser.add_argument("--no-deletions", action="store_true")
    parser.add_argument("--no-swaps", action="store_true")
    parser.add_argument("--no-replacements", action="store_true")
    parser.add_argument("--no-policy-repairs", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.run_json.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", ())
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    adapter = TextWorldAdapter()
    probe = TraceEditProbe(
        adapter=adapter,
        config=TraceEditConfig(
            equivalence_mode=args.equivalence_mode,
            probe_deletions=not args.no_deletions,
            probe_swaps=not args.no_swaps,
            probe_replacements=not args.no_replacements,
            probe_policy_repairs=not args.no_policy_repairs,
            max_replacements_per_step=args.max_replacements_per_step,
        ),
    )

    reports = []
    for index, episode in enumerate(episodes):
        report = probe.probe_episode(episode)
        reports.append(report)
        if args.progress:
            print(
                f"episode={index + 1}/{len(episodes)} "
                f"task={Path(report.task_id).stem} "
                f"success={report.original_success} "
                f"necessary={report.metrics['necessary_actions']:.0f} "
                f"redundant={report.metrics['redundant_actions']:.0f} "
                f"repairs={report.metrics['repair_candidates']:.0f}",
                flush=True,
            )

    output = {
        "source_run": str(args.run_json),
        "summary": summarize_reports(reports),
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(to_jsonable(output["summary"]), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def summarize_reports(reports: list[Any]) -> dict[str, Any]:
    episodes = len(reports)
    metric_keys = sorted({key for report in reports for key in report.metrics})
    summary = {
        key: sum(report.metrics.get(key, 0.0) for report in reports)
        for key in metric_keys
    }
    return {
        "episodes": episodes,
        "successful_episodes": sum(report.original_success for report in reports),
        "replay_successful_episodes": sum(report.replay_success for report in reports),
        **summary,
    }


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
