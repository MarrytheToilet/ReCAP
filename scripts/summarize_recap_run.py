#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"


METHOD_FILES = [
    ("Raw rank", None),
    ("Random rerank", "random_eval_t30"),
    ("Navigation prior", "navigation_prior_eval_t30"),
    ("Anti-static heuristic", "anti_static_eval_t30"),
    ("Learned verb prior", "verb_prior_eval_t30"),
    ("Exact memory", "exact_memory_eval_t30"),
    ("NN memory", "nn_eval_t30"),
    ("Feature reranker", "feature_eval_t30"),
    ("Feature reranker + margin 5", "feature_eval_t30_margin50"),
    ("Oracle ReCAP", None),
]


RETENTION_FILES = [
    ("0.0", "gold_demotion_feature_margin00"),
    ("2.0", "gold_demotion_feature_margin20"),
    ("5.0", "gold_demotion_feature_margin50"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact Markdown report for a ReCAP run.")
    parser.add_argument("--run", default=None, help="Run prefix, e.g. recap_xhard_500_mimo25_t1_top5.")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    run = args.run or choose_latest_run()
    report = build_report(run)
    out_json = args.out_json or ANALYSIS / f"{run}_auto_report.json"
    out_md = args.out_md or ANALYSIS / f"{run}_auto_report.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(run, report), encoding="utf-8")
    print(f"run={run}")
    print(f"wrote={out_json}")
    print(f"wrote={out_md}")


def choose_latest_run() -> str:
    candidates = sorted(ANALYSIS.glob("recap_xhard_*_mimo25_t1_top5_decomposition.json"))
    if not candidates:
        raise FileNotFoundError("No ReCAP decomposition files found in analysis/.")
    def stage(path: Path) -> int:
        parts = path.name.split("_")
        try:
            return int(parts[2])
        except (IndexError, ValueError):
            return -1
    return max(candidates, key=stage).name.removesuffix("_decomposition.json")


def build_report(run: str) -> dict[str, Any]:
    decomposition = read_json(required(run, "decomposition"))["summary"]
    by_trajectory = read_json(required(run, "decomposition")).get("by_trajectory", [])
    ranking = read_json(required(run, "candidate_ranking"))["summary"]
    bootstrap = read_json(required(run, "bootstrap"))
    preferences = read_jsonl(required(run, "preferences", suffix=".jsonl"))
    trajectories_summary = read_json(optional(run, "trajectories.summary") or Path())
    split_counts = read_split_counts(run)
    methods = read_method_results(run)
    retention = read_retention_results(run)
    certificates = Counter(str(row.get("certificate_level", "unknown")) for row in preferences)
    suffix_lengths = [
        int(row["repair_suffix_len"])
        for row in preferences
        if row.get("repair_suffix_len") is not None
    ]
    recappable = [
        row
        for row in by_trajectory
        if row.get("outcome") == "fail" and int(row.get("num_certified_preferences", 0)) > 0
    ]
    return {
        "data": {
            "rollouts": int(decomposition.get("trajectories", 0)),
            "successful_trajectories": int(trajectories_summary.get("successful_trajectories", 0)),
            "failed_trajectories": int(decomposition.get("failed_trajectories", 0)),
            "failed_candidate_steps": int(decomposition.get("candidate_steps", 0)),
            "error_records": int(trajectories_summary.get("errored_trajectories", 0)),
        },
        "failure_decomposition": {
            "certified_preferences": int(decomposition.get("certified_preferences", 0)),
            "recappable_failed_trajectories": len(recappable),
            "recappable_failed_trajectory_rate": safe_div(
                len(recappable), int(decomposition.get("failed_trajectories", 0))
            ),
            "certified_misranking_step_rate": metric_with_ci(
                decomposition,
                bootstrap,
                "decomposition",
                "certified_misranking_step_rate",
            ),
            "candidate_absent_step_rate": metric_with_ci(
                decomposition,
                bootstrap,
                "decomposition",
                "candidate_absent_step_rate",
            ),
            "repair_same_as_executed_step_rate": metric_with_ci(
                decomposition,
                bootstrap,
                "decomposition",
                "repair_same_as_executed_step_rate",
            ),
            "no_repair_found_step_rate": metric_with_ci(
                decomposition,
                bootstrap,
                "decomposition",
                "no_repair_found_step_rate",
            ),
            "invalid_replay": int(decomposition.get("invalid_replay", 0)),
        },
        "oracle_candidate_repair": {
            "preferred_in_candidates": int(ranking.get("preferred_in_candidates", 0)),
            "preferences": int(ranking.get("preferences", 0)),
            "preferred_top1_before": int(ranking.get("preferred_top1_before", 0)),
            "avg_preferred_rank_before": float(ranking.get("avg_preferred_rank_before", 0.0)),
            "raw_mrr": metric_with_ci(ranking, bootstrap, "ranking", "raw_mrr"),
            "oracle_recap_mrr": float(ranking.get("oracle_recap_mrr", 0.0)),
            "raw_ndcg": metric_with_ci(ranking, bootstrap, "ranking", "raw_ndcg"),
            "oracle_recap_ndcg": float(ranking.get("oracle_recap_ndcg", 0.0)),
            "certificate_distribution": dict(sorted(certificates.items())),
            "repair_suffix_len": summarize_numbers(suffix_lengths),
        },
        "split_counts": split_counts,
        "reranking": methods,
        "retention": retention,
    }


def read_method_results(run: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_summary: Mapping[str, Any] | None = None
    feature_path = optional(run, "feature_eval_t30")
    if feature_path is not None:
        feature_summary = read_json(feature_path)["summary"]
    for name, stem in METHOD_FILES:
        if name == "Raw rank":
            if feature_summary is None:
                continue
            rows.append(
                {
                    "method": name,
                    "mrr": float(feature_summary["raw_mrr"]),
                    "ndcg": float(feature_summary["raw_ndcg"]),
                    "top1_correction": 0.0,
                    "abstain_rate": 0.0,
                }
            )
            continue
        if name == "Oracle ReCAP":
            rows.append(
                {
                    "method": name,
                    "mrr": 1.0,
                    "ndcg": 1.0,
                    "top1_correction": 1.0,
                    "abstain_rate": None,
                }
            )
            continue
        path = optional(run, stem or "")
        if path is None:
            continue
        summary = read_json(path)["summary"]
        rows.append(
            {
                "method": name,
                "mrr": maybe_float(summary.get("learned_mrr")),
                "ndcg": maybe_float(summary.get("learned_ndcg")),
                "top1_correction": maybe_float(summary.get("learned_top1_correction_rate")),
                "abstain_rate": maybe_float(summary.get("learned_abstain_rate")),
            }
        )
    return rows


def read_retention_results(run: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for margin, stem in RETENTION_FILES:
        path = optional(run, stem)
        if path is None:
            continue
        summary = read_json(path)["summary"]
        rows.append(
            {
                "margin": margin,
                "abstain_rate": float(summary["abstain_rate"]),
                "intervention_rate": float(summary["intervention_rate"]),
                "gold_demotion_rate": float(summary["gold_demotion_rate_over_all_predicted"]),
                "gold_retention_rate": float(summary["gold_retention_rate_with_abstention"]),
            }
        )
    return rows


def read_split_counts(run: str) -> dict[str, int]:
    split_dir = ANALYSIS / f"{run}_splits_t30"
    if not split_dir.exists():
        return {}
    return {
        split: count_jsonl(split_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }


def render_markdown(run: str, report: Mapping[str, Any]) -> str:
    data = report["data"]
    failure = report["failure_decomposition"]
    oracle = report["oracle_candidate_repair"]
    lines = [
        f"# ReCAP Report: `{run}`",
        "",
        "## Data",
        "",
        f"- Rollouts: {data['rollouts']}.",
        f"- Successful trajectories: {data['successful_trajectories']}.",
        f"- Failed trajectories: {data['failed_trajectories']}.",
        f"- Failed candidate steps: {data['failed_candidate_steps']}.",
        f"- Error records: {data['error_records']}.",
        "",
        "## Failure Decomposition",
        "",
        f"- Certified preferences: {failure['certified_preferences']}.",
        (
            f"- Recappable failed trajectories: {failure['recappable_failed_trajectories']}/"
            f"{data['failed_trajectories']} = {failure['recappable_failed_trajectory_rate']:.3f}."
        ),
        f"- Certified misranking step rate: {format_ci(failure['certified_misranking_step_rate'])}.",
        f"- Candidate absent step rate: {format_ci(failure['candidate_absent_step_rate'])}.",
        f"- Repair same as executed step rate: {format_ci(failure['repair_same_as_executed_step_rate'])}.",
        f"- No repair found step rate: {format_ci(failure['no_repair_found_step_rate'])}.",
        f"- Invalid replay: {failure['invalid_replay']}.",
        "",
        "## Oracle Candidate Repair",
        "",
        f"- Preferred action in candidates: {oracle['preferred_in_candidates']}/{oracle['preferences']}.",
        f"- Preferred top-1 before reranking: {oracle['preferred_top1_before']}/{oracle['preferences']}.",
        f"- Average preferred rank before: {oracle['avg_preferred_rank_before']:.3f}.",
        f"- Raw MRR: {format_ci(oracle['raw_mrr'])}.",
        f"- Oracle ReCAP MRR: {oracle['oracle_recap_mrr']:.4f}.",
        f"- Raw NDCG: {format_ci(oracle['raw_ndcg'])}.",
        f"- Oracle ReCAP NDCG: {oracle['oracle_recap_ndcg']:.4f}.",
        f"- Certificate distribution: {json.dumps(oracle['certificate_distribution'], sort_keys=True)}.",
        f"- Repair suffix length: {format_suffix(oracle['repair_suffix_len'])}.",
        "",
    ]
    if report.get("split_counts"):
        split = report["split_counts"]
        lines.extend(
            [
                "## Held-Out Split",
                "",
                f"- Train/valid/test preferences: {split.get('train', 0)}/{split.get('valid', 0)}/{split.get('test', 0)}.",
                "",
            ]
        )
    if report.get("reranking"):
        lines.extend(
            [
                "## Held-Out Candidate Reranking",
                "",
                "| Method | MRR | NDCG | Top-1 correction | Abstain |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["reranking"]:
            lines.append(
                "| {method} | {mrr} | {ndcg} | {top1} | {abstain} |".format(
                    method=row["method"],
                    mrr=format_optional(row["mrr"]),
                    ndcg=format_optional(row["ndcg"]),
                    top1=format_optional(row["top1_correction"]),
                    abstain=format_optional(row["abstain_rate"]),
                )
            )
        lines.append("")
    if report.get("retention"):
        lines.extend(
            [
                "## Success-Step Retention",
                "",
                "| Margin | Abstain | Intervene | Gold demotion | Gold retention |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["retention"]:
            lines.append(
                "| {margin} | {abstain:.4f} | {intervene:.4f} | {demotion:.4f} | {retention:.4f} |".format(
                    margin=row["margin"],
                    abstain=row["abstain_rate"],
                    intervene=row["intervention_rate"],
                    demotion=row["gold_demotion_rate"],
                    retention=row["gold_retention_rate"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def required(run: str, stem: str, suffix: str = ".json") -> Path:
    path = ANALYSIS / f"{run}_{stem}{suffix}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def optional(run: str, stem: str, suffix: str = ".json") -> Path | None:
    path = ANALYSIS / f"{run}_{stem}{suffix}"
    return path if path.exists() else None


def read_json(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def metric_with_ci(
    summary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    group: str,
    key: str,
) -> dict[str, float]:
    ci = dict(dict(bootstrap.get(group, {})).get(key, {}))
    return {
        "point": float(summary.get(key, ci.get("point", 0.0))),
        "ci_low": float(ci.get("ci_low", summary.get(key, 0.0))),
        "ci_high": float(ci.get("ci_high", summary.get(key, 0.0))),
    }


def summarize_numbers(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": float(ordered[p90_index]),
    }


def format_ci(metric: Mapping[str, float]) -> str:
    return f"{metric['point']:.4f}, 95% CI [{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]"


def format_suffix(summary: Mapping[str, float]) -> str:
    return f"mean {summary['mean']:.2f}, median {summary['median']:.0f}, p90 {summary['p90']:.0f}"


def format_optional(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
