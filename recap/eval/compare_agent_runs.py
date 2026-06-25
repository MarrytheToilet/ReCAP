from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare agent evaluation JSON files.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("analysis/agent_comparison.md"))
    parser.add_argument("--paired-baseline", type=Path, default=None)
    parser.add_argument("--paired-treatment", type=Path, default=None)
    parser.add_argument("--paired-out", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [row_from_run(path) for path in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows), encoding="utf-8")
    print(f"wrote={args.out} rows={len(rows)}")
    if args.paired_baseline is not None and args.paired_treatment is not None:
        output = paired_summary(
            read_run(args.paired_baseline),
            read_run(args.paired_treatment),
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        paired_out = args.paired_out or args.out.with_suffix(".paired.json")
        paired_out.parent.mkdir(parents=True, exist_ok=True)
        paired_out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"paired={paired_out}")


def read_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_run(path: Path) -> dict[str, Any]:
    data = read_run(path)
    summary = data["summary"]
    failures = [
        Path(episode["task_id"]).stem
        for episode in data["episodes"]
        if not episode["success"]
    ]
    return {
        "run": path.stem,
        "agent": summary.get("agent", ""),
        "controller": summary["controller"],
        "model": summary.get("model", ""),
        "episodes": summary["episodes"],
        "success_rate": summary["success_rate"],
        "avg_steps": summary["avg_steps"],
        "avg_reward": summary["avg_reward"],
        "avg_repeat": summary["avg_repeated_action_rate"],
        "failures": ", ".join(failures) or "-",
    }


def paired_summary(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    baseline_episodes = episode_index(baseline)
    treatment_episodes = episode_index(treatment)
    shared = sorted(set(baseline_episodes) & set(treatment_episodes))
    if not shared:
        raise ValueError("paired comparison requires at least one shared task_id")
    rows = [
        paired_row(baseline_episodes[task_id], treatment_episodes[task_id])
        for task_id in shared
    ]
    delta = mean(row["treatment_success"] - row["baseline_success"] for row in rows)
    rescue = sum(1 for row in rows if row["baseline_success"] == 0 and row["treatment_success"] == 1)
    harm = sum(1 for row in rows if row["baseline_success"] == 1 and row["treatment_success"] == 0)
    bootstrap = bootstrap_delta(rows, iterations=iterations, seed=seed)
    return {
        "episodes": len(rows),
        "baseline_success_rate": mean(row["baseline_success"] for row in rows),
        "treatment_success_rate": mean(row["treatment_success"] for row in rows),
        "success_delta": delta,
        "success_delta_ci_low": bootstrap["ci_low"],
        "success_delta_ci_high": bootstrap["ci_high"],
        "rescued_failures": rescue,
        "harmed_successes": harm,
        "paired_sign_p_value": paired_sign_p_value(rescue, harm),
        "baseline_avg_steps": mean(row["baseline_steps"] for row in rows),
        "treatment_avg_steps": mean(row["treatment_steps"] for row in rows),
        "baseline_repeat_rate": mean(row["baseline_repeat_rate"] for row in rows),
        "treatment_repeat_rate": mean(row["treatment_repeat_rate"] for row in rows),
        "treatment_intervention_rate": mean(row["treatment_intervention_rate"] for row in rows),
        "treatment_gold_demotion_rate": mean(row["treatment_gold_demotion_rate"] for row in rows),
        "rows": rows,
        "bootstrap_iterations": iterations,
    }


def episode_index(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(episode["task_id"]): episode for episode in run.get("episodes", ())}


def paired_row(
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_metrics = baseline.get("metrics", {})
    treatment_metrics = treatment.get("metrics", {})
    return {
        "task_id": str(baseline["task_id"]),
        "baseline_success": 1 if baseline.get("success") else 0,
        "treatment_success": 1 if treatment.get("success") else 0,
        "baseline_steps": float(baseline_metrics.get("steps", len(baseline.get("steps", ())))),
        "treatment_steps": float(treatment_metrics.get("steps", len(treatment.get("steps", ())))),
        "baseline_repeat_rate": float(baseline_metrics.get("repeated_action_rate", 0.0)),
        "treatment_repeat_rate": float(treatment_metrics.get("repeated_action_rate", 0.0)),
        "treatment_intervention_rate": float(treatment_metrics.get("reranker_intervention_rate", 0.0)),
        "treatment_gold_demotion_rate": float(treatment_metrics.get("reranker_gold_demotion_rate", 0.0)),
        "treatment_interventions": int(treatment_metrics.get("reranker_interventions", 0.0)),
        "treatment_demotions": int(treatment_metrics.get("reranker_demoted_gold", 0.0)),
    }


def bootstrap_delta(
    rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        deltas.append(
            mean(row["treatment_success"] - row["baseline_success"] for row in sample)
        )
    deltas.sort()
    return {
        "ci_low": quantile(deltas, 0.025),
        "ci_high": quantile(deltas, 0.975),
    }


def paired_sign_p_value(rescue: int, harm: int) -> float | None:
    trials = rescue + harm
    if trials == 0:
        return None
    tail = min(rescue, harm)
    probability = sum(comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    value = 1
    for i in range(1, k + 1):
        value = value * (n - k + i) // i
    return value


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def mean(values: Any) -> float:
    items = list(values)
    return sum(float(item) for item in items) / len(items) if items else 0.0


def render(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "agent",
        "controller",
        "model",
        "episodes",
        "success_rate",
        "avg_steps",
        "avg_reward",
        "avg_repeat",
        "failures",
    ]
    lines = [
        "# Agent Run Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[header]) for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
