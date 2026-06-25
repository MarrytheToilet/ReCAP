#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${1:-recap_xhard_500_mimo25_t1_top5}"
SPLIT_DIR="analysis/${RUN}_splits_t30"

python -m recap.eval.bootstrap_candidate_ranking \
  --preferences "${SPLIT_DIR}/test.jsonl" \
  --predictions "analysis/${RUN}_feature_predictions_t30.jsonl" \
  --iterations 1000 \
  --out "analysis/${RUN}_feature_bootstrap_t30.json"

python -m recap.eval.eval_candidate_ranking_by_action_type \
  --preferences "${SPLIT_DIR}/test.jsonl" \
  --predictions "analysis/${RUN}_feature_predictions_t30.jsonl" \
  --out "analysis/${RUN}_feature_by_action_type_t30.json"

python -m recap.eval.eval_candidate_ranking_by_action_type \
  --preferences "${RUN:+${SPLIT_DIR}/test.jsonl}" \
  --out "analysis/${RUN}_raw_by_action_type_t30.json"

for spec in \
  "pairwise_rejected --negative-scope rejected" \
  "listwise --loss listwise" \
  "no_cert_weight --no-certificate-weight" \
  "rank_only --feature-set rank-only" \
  "action_only --feature-set action-only" \
  "no_rank --feature-set no-rank" \
  "no_action_type --feature-set no-action-type" \
  "no_history --feature-set no-history"
do
  name="${spec%% *}"
  args="${spec#* }"
  python -m recap.models.train_action_reranker \
    --train "${SPLIT_DIR}/train.jsonl" \
    --valid "${SPLIT_DIR}/valid.jsonl" \
    ${args} \
    --out "analysis/${RUN}_feature_ablation_${name}_t30.json"
  python -m recap.models.eval_action_reranker \
    --test "${SPLIT_DIR}/test.jsonl" \
    --model "analysis/${RUN}_feature_ablation_${name}_t30.json" \
    --out-predictions "analysis/${RUN}_feature_ablation_${name}_predictions_t30.jsonl" \
    --out "analysis/${RUN}_feature_ablation_${name}_eval_t30.json"
done
