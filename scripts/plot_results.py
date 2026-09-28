from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "results" / "figures"

BLUE = "#86BDF2"
BLUE_DARK = "#3F82C4"
SKY = "#B9DBFF"
PINK = "#F5A8C7"
PINK_DARK = "#D76A9A"
ROSE = "#F2B6B6"
LAVENDER = "#C8B8F4"
GRAY = "#B7C0CF"
LIGHT_GRAY = "#EAF1F8"
VERY_LIGHT = "#F8FBFF"
DARK = "#2D3246"
MUTED = "#687083"

# Soft "candy" palette (light blue + light pink family) for the maintained
# figures. Variable names are kept generic so the layout code is unchanged.
INK = "#3A3F52"          # primary text / axes
SUBTLE = "#9AA1B4"       # secondary text
GRIDC = "#EEF1F7"        # gridlines
TEAL = "#7FB6EF"         # hero: ReCAP / addressable / method (candy blue)
TEAL_SOFT = "#C5DEF8"    # light candy blue (bar fill)
CORAL = "#F4A6C6"        # accent: online / candidate-absent (candy pink)
CORAL_SOFT = "#FBD0E0"
PERI = "#BBA9EE"         # candy lavender: rank-only / same-as-executed
PERI_SOFT = "#D8CCF5"
SKYB = "#A9CCF1"         # pale sky blue: semantic / NN
SAND = "#D2D8E2"         # neutral: no-repair / raw baseline point
OracleDK = "#6E76A6"     # oracle upper-bound point (deep periwinkle)


def choose_current_run() -> str:
    explicit = os.environ.get("RECAP_RUN")
    if explicit:
        return explicit
    for stage in (700, 500, 300, 130):
        candidate = f"recap_xhard_{stage}_mimo25_t1_top5"
        if (ANALYSIS / f"{candidate}_decomposition.json").exists():
            return candidate
    return "recap_xhard_500_mimo25_t1_top5"


CURRENT_RUN = choose_current_run()


def main() -> None:
    import argparse
    global ANALYSIS, FIGURES, CURRENT_RUN
    parser = argparse.ArgumentParser(description="Plot the diagnosis and learning/deployment results from saved evaluations.")
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    ANALYSIS, FIGURES = args.analysis, args.out
    CURRENT_RUN = choose_current_run()
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_style()
    make_diagnosis_summary()
    make_learning_deployment_summary()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#C9D0DA",
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )




def reranking_points() -> list[dict[str, Any]]:
    method_files = [("Raw rank", None), ("NN memory", f"{CURRENT_RUN}_nn_eval_t30.json"), ("Oracle", None)]
    bge_zeroshot_file = ANALYSIS / f"{CURRENT_RUN}_bge_reranker_zeroshot_eval_t30.json"
    if bge_zeroshot_file.exists():
        method_files.insert(-1, ("BGE w/o SFT", bge_zeroshot_file.name))
    policy_file = ANALYSIS / f"{CURRENT_RUN}_support_policy_best_eval_t30.json"
    if not policy_file.exists():
        policy_file = ANALYSIS / f"{CURRENT_RUN}_policy_eval_t30.json"
    if policy_file.exists():
        method_files.insert(-1, ("ReCAP policy", policy_file.name))
    raw_summary = read_json(f"{CURRENT_RUN}_feature_eval_t30.json")["summary"]
    points: list[dict[str, Any]] = []
    for name, filename in method_files:
        if name == "Raw rank":
            points.append({"name": name, "mrr": float(raw_summary["raw_mrr"]), "top1": 0.0, "kind": "raw"})
        elif name == "Oracle":
            points.append({"name": name, "mrr": 1.0, "top1": 1.0, "kind": "oracle"})
        else:
            summary = read_json(filename)["summary"]
            kind = "memory"
            if name in {"BGE w/o SFT", "BGE + SFT"}:
                kind = "semantic"
            if name == "ReCAP policy":
                kind = "policy"
            points.append({"name": name, "mrr": float(summary["learned_mrr"]), "top1": float(summary["learned_top1_correction_rate"]), "kind": kind})
    return points




def make_diagnosis_summary() -> None:
    """Anatomy of failure: how the failed steps partition, where the repairing
    action sat in the candidate list, and the certificate's operating range."""
    stages = [
        stage
        for stage in (130, 300, 500, 700)
        if (ANALYSIS / f"recap_xhard_{stage}_mimo25_t1_top5_decomposition.json").exists()
    ]
    summaries = {
        stage: read_json(f"recap_xhard_{stage}_mimo25_t1_top5_decomposition.json")["summary"]
        for stage in stages
    }
    final = summaries[stages[-1]]
    pref_rows = read_jsonl(f"{CURRENT_RUN}_preferences.jsonl")
    rank_counts = Counter(int(row["preferred_rank_before"]) for row in pref_rows)
    ranks = [2, 3, 4, 5]
    rank_values = [rank_counts.get(rank, 0) for rank in ranks]
    rank_total = sum(rank_values)
    rank_fracs = [value / rank_total if rank_total else 0.0 for value in rank_values]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.05))
    fig.subplots_adjust(left=0.088, right=0.99, top=0.82, bottom=0.25, wspace=0.52)

    # ---- (a) 100%-stacked decomposition, with the categories labelled directly
    #          to the right of the final bar (no detached legend) ----
    ax = axes[0]
    # The stacked category is candidate-present certified repairs; the strict
    # paired-cost audit further splits it (Table 1). Read the strict count so
    # the hero label reports both rates without hardcoding.
    strict_note = ""
    strict_path = ANALYSIS / f"{CURRENT_RUN}_strict_certificate.json"
    if strict_path.exists():
        strict_summary = json.loads(strict_path.read_text())
        strict_count = strict_summary["statuses"].get("strict_misrank", 0)
        denom = float(final["denominators"]["candidate_step_level"])
        strict_note = f"\n(strict {strict_count / denom * 100:.1f}%)"
    cats = [
        ("Cand.-present repair", "certified_misranking_step_rate", TEAL, True),
        ("Cand. absent", "candidate_absent_step_rate", CORAL, False),
        ("Same as exec.", "repair_same_as_executed_step_rate", PERI, False),
        ("No local repair", "no_repair_found_step_rate", SAND, False),
    ]
    fill = {"Cand.-present repair": TEAL, "Cand. absent": CORAL,
            "Same as exec.": PERI_SOFT, "No local repair": SAND}
    x_pos = list(range(len(stages)))
    width = 0.62
    bottoms = [0.0] * len(stages)
    centers = {}
    for label, key, _color, _hero in cats:
        vals = [float(summaries[s][key]) for s in stages]
        ax.bar(x_pos, vals, width, bottom=bottoms, color=fill[label],
               edgecolor="white", linewidth=0.8)
        centers[label] = bottoms[-1] + vals[-1] / 2.0
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    last_x = x_pos[-1] + width / 2
    for label, key, color, hero in cats:
        pct = float(final[key]) * 100
        text = f"{label}  {pct:.0f}%" + (strict_note if hero else "")
        ax.annotate(
            text,
            xy=(last_x, centers[label]), xytext=(last_x + 0.55, centers[label]),
            textcoords="data", va="center", ha="left",
            color=(color if hero else INK), fontsize=7.0,
            fontweight=("bold" if hero else "normal"),
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
        )
    ax.set_title("(a) Failure decomposition", loc="left", pad=7, color=INK)
    ax.set_ylabel("Share of failed steps")
    ax.set_xlabel("Collected rollouts")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in stages])
    ax.set_xlim(-0.6, x_pos[-1] + 3.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50", "100%"])
    ax.tick_params(length=0)
    clean_axes(ax)

    # ---- (b) where the repair sat in the logged list (horizontal bars) ----
    ax = axes[1]
    y_pos = list(range(len(ranks)))[::-1]  # rank 2 on top
    ax.barh(y_pos, rank_fracs, height=0.6, color=TEAL_SOFT,
            edgecolor=TEAL, linewidth=1.0)
    for y, value, frac in zip(y_pos, rank_values, rank_fracs):
        ax.text(frac + 0.02, y, f"{value}", ha="left", va="center",
                color=INK, fontsize=7.5)
    ax.set_title("(b) Repair rank in the list", loc="left", pad=7, color=INK)
    ax.set_xlabel("Share of certified prefs")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"rank {r}" for r in ranks])
    ax.set_xlim(0, max(rank_fracs) + 0.18)
    ax.set_xticks([0.0, 0.3, 0.6])
    ax.set_xticklabels(["0", "30", "60%"])
    ax.tick_params(axis="y", length=0)
    ax.text(0.97, 0.07, "avg rank 2.60\nnever ranked top-1", transform=ax.transAxes,
            ha="right", va="bottom", color=SUBTLE, fontsize=6.6, linespacing=1.25)
    ax.grid(axis="x", color=GRIDC, linewidth=0.8)
    clean_axes(ax)

    # ---- (c) certificate yield vs candidate width and suffix horizon ----
    ax = axes[2]
    sensitivity = read_json(f"{CURRENT_RUN}_sensitivity.json")
    full_prefs = float(sensitivity["summary"]["certified_preferences"])
    topk = sensitivity["top_k_truncation"]
    suffix = sensitivity["suffix_budget"]
    top_x = list(range(len(topk)))
    gap = 1
    suffix_x = [x + len(topk) + gap for x in range(len(suffix))]
    top_y = [row["certified_preferences_retained"] / full_prefs for row in topk]
    suffix_y = [row["certified_preferences_retained"] / full_prefs for row in suffix]
    ax.plot(top_x, top_y, marker="o", color=TEAL, linewidth=1.9, markersize=5,
            markeredgecolor="white", markeredgewidth=0.9)
    ax.plot(suffix_x, suffix_y, marker="o", color=CORAL, linewidth=1.9, markersize=5,
            markeredgecolor="white", markeredgewidth=0.9)
    # annotate the retained fraction at each operating point
    for xs, ys, color in [(top_x, top_y, TEAL), (suffix_x, suffix_y, CORAL)]:
        for x, y in zip(xs, ys):
            ax.annotate(f"{y * 100:.0f}%", xy=(x, y), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        color=color, fontsize=6.3)
    divider = len(topk) + gap / 2 - 0.5
    ax.axvline(divider, color="#E2E7EE", linewidth=1.0)
    ax.set_title("(c) Certificate yield", loc="left", pad=7, color=INK)
    ax.set_ylabel("Prefs retained")
    ax.set_ylim(0, 1.24)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50", "100%"])
    ax.set_xticks(top_x + suffix_x)
    ax.set_xticklabels(
        [str(row["top_k"]) for row in topk]
        + [str(row["suffix_budget"]) if str(row["suffix_budget"]).isdigit() else "full"
           for row in suffix]
    )
    ax.set_xlim(-0.5, suffix_x[-1] + 0.5)
    # region labels sit just under the tick row, coloured to match each curve
    ax.text((len(topk) - 1) / 2, -0.18, "candidate width $k$",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            color=TEAL, fontsize=6.8)
    ax.text((suffix_x[0] + suffix_x[-1]) / 2, -0.18, "suffix horizon",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            color=CORAL, fontsize=6.8)
    ax.grid(axis="y", color=GRIDC, linewidth=0.8)
    clean_axes(ax)

    save_figure(fig, "recap_diagnosis_summary", tight=False)
    plt.close(fig)


def make_learning_deployment_summary() -> None:
    """From certified labels to control: held-out reranking, evidence the gain
    is not the rank-2 shortcut, and closed-loop deployment in two regimes."""
    points = reranking_points()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.05))
    fig.subplots_adjust(left=0.082, right=0.99, top=0.82, bottom=0.25, wspace=0.42)

    # ---- (a) held-out reranking: MRR vs top-1 correction ----
    ax = axes[0]
    colors = {"raw": SAND, "memory": PERI, "semantic": SKYB, "policy": TEAL, "oracle": OracleDK}
    for start_name, end_name in [
        ("Raw rank", "NN memory"),
        ("BGE w/o SFT", "ReCAP policy"),
        ("NN memory", "ReCAP policy"),
        ("ReCAP policy", "Oracle"),
    ]:
        if not any(p["name"] == start_name for p in points) or not any(p["name"] == end_name for p in points):
            continue
        s = next(p for p in points if p["name"] == start_name)
        e = next(p for p in points if p["name"] == end_name)
        ax.annotate("", xy=(e["mrr"], e["top1"]), xytext=(s["mrr"], s["top1"]),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#CBD2DC",
                                "shrinkA": 7, "shrinkB": 7}, zorder=1)
    label_positions = {
        "Raw rank": ("Raw", 0, -12, "center"),
        "NN memory": ("NN memory", 0, 13, "center"),
        "BGE w/o SFT": ("BGE 0-shot", 8, 1, "left"),
        "ReCAP policy": ("ReCAP", -10, 4, "right"),
        "Oracle": ("Oracle", 0, 12, "center"),
    }
    for p in points:
        ax.scatter(p["mrr"], p["top1"], s=95, color=colors[p["kind"]],
                   edgecolor="white", linewidth=1.3, zorder=3)
        label, dx, dy, ha = label_positions[p["name"]]
        ax.annotate(label, xy=(p["mrr"], p["top1"]), xytext=(dx, dy),
                    textcoords="offset points", color=INK, va="center", ha=ha,
                    fontsize=7.0)
    ax.set_title("(a) Held-out reranking", loc="left", pad=7, color=INK)
    ax.set_xlabel("MRR")
    ax.set_ylabel("Top-1 correction")
    ax.set_xlim(0.36, 1.10)
    ax.set_ylim(-0.18, 1.24)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"])
    ax.grid(color=GRIDC, linewidth=0.8)
    clean_axes(ax)

    # ---- (b) top-1 correction by repair rank (raw is 0 by construction) ----
    ax = axes[1]
    rank_labels = ["2", "3", "4", "5"]
    rank_n = [74, 21, 8, 2]
    series = [
        ("Rank-only", [1.000, 0.000, 0.000, 0.000], PERI),
        ("ReCAP", [0.932, 0.667, 0.875, 0.000], TEAL),
    ]
    group_x = list(range(len(rank_labels)))
    bar_w = 0.38
    offsets = [(si - (len(series) - 1) / 2) * bar_w for si in range(len(series))]
    for si, (name, vals, color) in enumerate(series):
        xs = [g + offsets[si] for g in group_x]
        ax.bar(xs, vals, bar_w, color=color, edgecolor="white",
               linewidth=0.7, label=name)
        for x, v in zip(xs, vals):
            if v > 0.02:
                ax.text(x, v + 0.03, f"{v * 100:.0f}", ha="center", va="bottom",
                        fontsize=6.6, color=INK)
    # label both methods directly inside the tall rank-2 bars (no legend)
    for si, (name, vals, _color) in enumerate(series):
        ax.text(group_x[0] + offsets[si], vals[0] / 2, name, rotation=90,
                ha="center", va="center", color="white", fontsize=6.2,
                fontweight="bold")
    ax.set_title("(b) Correction by rank", loc="left", pad=7, color=INK)
    ax.set_xticks(group_x)
    ax.set_xticklabels([f"rank {r}\n($n$={n})" for r, n in zip(rank_labels, rank_n)],
                       fontsize=7.0)
    ax.set_ylabel("Top-1 correction")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50", "100%"])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRIDC, linewidth=0.8)
    clean_axes(ax)

    # ---- (c) closed-loop deployment, two regimes on one success axis ----
    ax = axes[2]
    def _succ(fname):
        eps = read_jsonl(f"localpolicy_eval/{fname}")
        return sum(1 for e in eps if e.get("success")) / len(eps)
    rows = []
    for label, basef, trtf in [
        ("LM hard", "hard_base.episodes.jsonl", "klpi_v3_hard.episodes.jsonl"),
        ("LM xhard", "base100_prog.episodes.jsonl", "klpi_v3_xhard.episodes.jsonl"),
    ]:
        rows.append({"label": label, "raw": _succ(basef),
                     "treat": _succ(trtf), "color": TEAL})
    for label, filename in [
        ("Verified prop.", "online_verified_proposal/bge_verified_top1_shorter_100_paired.json"),
        ("Full replay", "online_expanded_seq/recap_replay_100_paired.json"),
    ]:
        data = json.loads((ANALYSIS / filename).read_text(encoding="utf-8"))
        rows.append({"label": label, "raw": float(data["baseline_success_rate"]),
                     "treat": float(data["treatment_success_rate"]), "color": CORAL})
    y_positions = [3.0, 2.0, 1.0, 0.0]
    for row, y in zip(rows, y_positions):
        raw, treat, color = row["raw"], row["treat"], row["color"]
        ax.plot([raw, treat], [y, y], color=color, linewidth=2.4, alpha=0.4,
                solid_capstyle="round", zorder=1)
        ax.annotate("", xy=(treat, y), xytext=(raw, y),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": color,
                                "shrinkA": 2, "shrinkB": 2}, zorder=2)
        ax.scatter([raw], [y], s=48, color=SAND, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter([treat], [y], s=80, color=color, edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(-0.02, y, row["label"], ha="right", va="center", color=INK,
                fontsize=7.2, transform=ax.get_yaxis_transform())
        succ = f"{raw * 100:.0f}→{treat * 100:.0f}%"
        if treat < 0.78:
            ax.text(treat + 0.035, y, succ, ha="left", va="center", color=INK,
                    fontweight="bold", fontsize=7.0)
        else:
            ax.text(raw - 0.035, y, succ, ha="right", va="center", color=INK,
                    fontweight="bold", fontsize=7.0)
    ax.axhline(1.5, color="#E2E7EE", linewidth=1.0)
    ax.text(0.985, 0.97, "local LM agent", transform=ax.transAxes, ha="right",
            va="top", color=TEAL, fontsize=6.8)
    ax.text(0.015, 0.06, "replay-verified", transform=ax.transAxes, ha="left",
            va="bottom", color=CORAL, fontsize=6.8)
    ax.set_title("(c) Closed-loop control", loc="left", pad=7, color=INK)
    ax.set_yticks([])
    ax.set_xlabel("Task success")
    ax.set_xlim(0.0, 1.04)
    ax.set_ylim(-0.7, 3.8)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100%"])
    ax.grid(axis="x", color=GRIDC, linewidth=0.8)
    clean_axes(ax)

    save_figure(fig, "recap_learning_deployment", tight=False)
    plt.close(fig)










def read_json(name: str) -> dict[str, Any]:
    return json.loads((ANALYSIS / name).read_text(encoding="utf-8"))


def read_jsonl(name: str) -> list[dict[str, Any]]:
    path = ANALYSIS / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save_figure(fig, stem: str, tight: bool = True) -> None:
    if tight:
        fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=300)
    else:
        fig.savefig(FIGURES / f"{stem}.png", dpi=300)


if __name__ == "__main__":
    main()
