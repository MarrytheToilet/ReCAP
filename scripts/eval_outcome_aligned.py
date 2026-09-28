"""Outcome-aligned (multi-positive, cost-sensitive) held-out evaluation.

Instead of scoring agreement with the single verifier-selected branch, this
script scores each selector's chosen top-1 action against the per-candidate
verified suffix costs from the multiplicity audit:

  - min-cost top-1: rate of selecting ANY candidate whose verified suffix
    cost equals the minimum over the logged candidate list;
  - mean regret: l(selected) - min_a l(a), in verifier steps, over steps
    where the selected action has a verified successful suffix;
  - selected-fails: rate of selecting a candidate with no verified suffix;
  - mean selected cost: average l(selected) over successful selections.

Cost-based metrics distinguish successful action selection from agreement
with the verifier's tie-breaking.
"""

from __future__ import annotations

import argparse
import json


BASE = "analysis/recap_xhard_700_mimo25_t1_top5_"
PREDICTIONS = {
    "support_policy": BASE + "support_policy_best_predictions_t30.jsonl",
    "structured": BASE + "feature_predictions_t30.jsonl",
    "bge_sft_diag": BASE + "bge_h12_e5_gpu_predictions_t30.jsonl",
    "rank_only": BASE + "feature_rank_only_predictions_t30.jsonl",
    "no_rank": BASE + "feature_no_rank_predictions_t30.jsonl",
    "anti_static": BASE + "anti_static_predictions_t30.jsonl",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def key(record):
    return (record["task_id"], str(record.get("seed", 0)), str(record["step_index"]))


def ranked_of(row):
    ranked = row["ranked_actions"]
    if isinstance(ranked, str):
        ranked = json.loads(ranked.replace("'", '"'))
    return list(ranked)


def evaluate(name, top1_of, test, costs):
    n = mincost = fails = 0
    regrets, selected_costs = [], []
    for record in test:
        cost = costs.get(key(record))
        if cost is None:
            continue
        action = top1_of(record)
        if action is None:
            continue
        n += 1
        cmin = min(cost.values())
        c = cost.get(action)
        if c is None:
            fails += 1
            continue
        mincost += c == cmin
        regrets.append(c - cmin)
        selected_costs.append(c)
    print(
        f"{name:16s} n={n:3d} min-cost-top1={mincost / n:.3f} "
        f"mean-regret={sum(regrets) / len(regrets):.2f} "
        f"selected-fails={fails / n:.3f} "
        f"mean-selected-cost={sum(selected_costs) / len(selected_costs):.2f}"
    )
    return {
        "n": n,
        "min_cost_top1": round(mincost / n, 4),
        "mean_regret": round(sum(regrets) / len(regrets), 3),
        "selected_fails": round(fails / n, 4),
        "mean_selected_cost": round(sum(selected_costs) / len(selected_costs), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default=BASE + "splits_t30/test.jsonl")
    parser.add_argument(
        "--multiplicity", default=BASE + "repair_multiplicity_rows.jsonl"
    )
    parser.add_argument("--out", default=BASE + "outcome_aligned_eval.json")
    args = parser.parse_args()

    test = load_jsonl(args.test)
    costs = {}
    for row in load_jsonl(args.multiplicity):
        mapping = {
            action: int(length)
            for action, length in zip(
                row.get("successful_actions", ()), row.get("successful_suffix_lens", ())
            )
        }
        if mapping:
            costs[key(row)] = mapping

    results = {}
    results["raw_order"] = evaluate(
        "raw_order", lambda r: r["candidates"][0], test, costs
    )
    results["oracle_min_cost"] = evaluate(
        "oracle_min_cost",
        lambda r: min(
            costs[key(r)], key=costs[key(r)].get
        ) if key(r) in costs else None,
        test,
        costs,
    )
    for name, path in PREDICTIONS.items():
        predictions = {key(p): p for p in load_jsonl(path)}
        results[name] = evaluate(
            name,
            lambda r, _p=predictions: ranked_of(_p[key(r)])[0] if key(r) in _p else None,
            test,
            costs,
        )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("written", args.out)


if __name__ == "__main__":
    main()
