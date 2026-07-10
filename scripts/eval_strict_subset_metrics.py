"""Recompute held-out reranking metrics on the strict-audit subset.

Joins saved test predictions with the strict-audit rows and reports MRR and
top-1 correction on (a) all test preferences and (b) the strict subset.
No model is retrained; this reproduces Appendix Table `strict holdout`:

    raw_order        105 0.442/0.000   101 0.440/0.000
    rank_only        105 0.835/0.705   101 0.828/0.693
    no_rank          105 0.779/0.581   101 0.777/0.574
    bge_sft_diag     105 0.840/0.686   101 0.838/0.683
    structured       105 0.917/0.838   101 0.919/0.842
    support_policy   105 0.927/0.857   101 0.924/0.851
"""

from __future__ import annotations

import argparse
import json


BASE = "analysis/recap_xhard_700_mimo25_t1_top5_"
DEFAULT_PREDICTIONS = {
    "support_policy": BASE + "support_policy_best_predictions_t30.jsonl",
    "bge_sft_diag": BASE + "bge_h12_e5_gpu_predictions_t30.jsonl",
    "structured": BASE + "feature_predictions_t30.jsonl",
    "rank_only": BASE + "feature_rank_only_predictions_t30.jsonl",
    "no_rank": BASE + "feature_no_rank_predictions_t30.jsonl",
}


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def key(record: dict) -> tuple[str, str, str]:
    return (record["task_id"], str(record.get("seed", 0)), str(record["step_index"]))


def ranked_actions(row: dict) -> list[str]:
    ranked = row["ranked_actions"]
    if isinstance(ranked, str):
        ranked = json.loads(ranked.replace("'", '"'))
    return list(ranked)


def metrics(test_rows: list[dict], pred_rows: list[dict]) -> tuple[int, float, float]:
    predictions = {key(row): row for row in pred_rows}
    mrr, top1 = [], []
    for test_row in test_rows:
        row = predictions.get(key(test_row))
        if row is None:
            continue
        ranked = ranked_actions(row)
        preferred = test_row["preferred_action"]
        rank = ranked.index(preferred) + 1 if preferred in ranked else len(ranked) + 1
        mrr.append(1.0 / rank)
        top1.append(1.0 if rank == 1 else 0.0)
    n = len(mrr)
    return n, sum(mrr) / n, sum(top1) / n


def raw_metrics(rows: list[dict]) -> tuple[int, float, float]:
    mrr = [1.0 / row["preferred_rank_before"] for row in rows]
    top1 = [1.0 if row["preferred_rank_before"] == 1 else 0.0 for row in rows]
    return len(mrr), sum(mrr) / len(mrr), sum(top1) / len(mrr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default=BASE + "splits_t30/test.jsonl")
    parser.add_argument("--strict-rows", default=BASE + "strict_certificate_rows.jsonl")
    args = parser.parse_args()

    strict = {
        key(row)
        for row in load_jsonl(args.strict_rows)
        if row["status"] == "strict_misrank"
    }
    test = load_jsonl(args.test)
    strict_test = [row for row in test if key(row) in strict]
    print(f"strict subset of test: {len(strict_test)}/{len(test)}")
    print(f"{'method':16s} {'n':>4s} {'MRR':>6s} {'Top1':>6s}   {'n':>4s} {'MRR':>6s} {'Top1':>6s}")

    n, m, t = raw_metrics(test)
    ns, ms, ts = raw_metrics(strict_test)
    print(f"{'raw_order':16s} {n:4d} {m:6.3f} {t:6.3f}   {ns:4d} {ms:6.3f} {ts:6.3f}")
    for name, path in DEFAULT_PREDICTIONS.items():
        preds = load_jsonl(path)
        n, m, t = metrics(test, preds)
        ns, ms, ts = metrics(strict_test, preds)
        print(f"{name:16s} {n:4d} {m:6.3f} {t:6.3f}   {ns:4d} {ms:6.3f} {ts:6.3f}")


if __name__ == "__main__":
    main()
