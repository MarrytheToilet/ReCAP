#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p analysis/logs data/textworld_xhard_games

MODEL="${ACTA_LLM_MODEL:-$(python - <<'PY'
from pathlib import Path
model = "mimo-v2.5"
env = Path(".env")
if env.exists():
    for line in env.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("ACTA_LLM_MODEL") and "=" in stripped:
            model = stripped.split("=", 1)[1].strip().strip('"').strip("'")
print(model)
PY
)}"
RUN="recap_xhard_700_mimo25_t1_top5"
NEW="recap_xhard_700_mimo25_t1_top5_701_900"
OLD="recap_xhard_500_mimo25_t1_top5"
MODEL_ROOT="${MODEL_ROOT:-../../models/recap/artifacts}"
mkdir -p "${MODEL_ROOT}"

echo "[$(date -Is)] stage=generate_games seeds=701-900"
python -m acta.data.generate_textworld_games \
  --seeds $(seq 701 900) \
  --out-dir data/textworld_xhard_games \
  --prefix acta_xhard_seed \
  --world-size 8 \
  --nb-objects 30 \
  --quest-length 8

echo "[$(date -Is)] stage=collect_trajectories model=${MODEL}"
python -m acta.eval.collect_failure_trajectories \
  --env textworld \
  --difficulty xhard \
  --game-dir data/textworld_xhard_games \
  --min-game-seed 701 \
  --max-game-seed 900 \
  --rollouts-per-game 1 \
  --agent openai \
  --model "${MODEL}" \
  --temperature 1.0 \
  --top-k 5 \
  --max-steps 12 \
  --include-successes \
  --continue-on-error \
  --write-errors \
  --workers "${COLLECT_WORKERS:-4}" \
  --llm-timeout "${LLM_TIMEOUT:-120}" \
  --llm-max-retries 6 \
  --llm-retry-base-delay 2.0 \
  --progress \
  --out "analysis/${NEW}_trajectories.jsonl"

echo "[$(date -Is)] stage=compile_recap_new"
python -m acta.eval.compile_recap_batch \
  --trajectories "analysis/${NEW}_trajectories.jsonl" \
  --out-trace-report "analysis/${NEW}_trace_report.json" \
  --out-preferences "analysis/${NEW}_preferences.jsonl" \
  --out-ledger "analysis/${NEW}_ledger.jsonl" \
  --out-trajectory-ledger "analysis/${NEW}_trajectory_ledger.jsonl" \
  --out-summary "analysis/${NEW}_compile_summary.json" \
  --workers "${COMPILE_WORKERS:-8}" \
  --trace-progress

echo "[$(date -Is)] stage=merge_700"
cat "analysis/${OLD}_trajectories.jsonl" "analysis/${NEW}_trajectories.jsonl" > "analysis/${RUN}_trajectories.jsonl"
cat "analysis/${OLD}_preferences.jsonl" "analysis/${NEW}_preferences.jsonl" > "analysis/${RUN}_preferences.jsonl"
cat "analysis/${OLD}_ledger.jsonl" "analysis/${NEW}_ledger.jsonl" > "analysis/${RUN}_ledger.jsonl"
cat "analysis/${OLD}_trajectory_ledger.jsonl" "analysis/${NEW}_trajectory_ledger.jsonl" > "analysis/${RUN}_trajectory_ledger.jsonl"

python -m acta.eval.eval_recap_failure_decomposition \
  --ledger "analysis/${RUN}_ledger.jsonl" \
  --preferences "analysis/${RUN}_preferences.jsonl" \
  --out "analysis/${RUN}_decomposition.json"

python -m acta.eval.eval_candidate_ranking \
  --preferences "analysis/${RUN}_preferences.jsonl" \
  --out "analysis/${RUN}_candidate_ranking.json"

python -m acta.eval.bootstrap_recap_metrics \
  --ledger "analysis/${RUN}_ledger.jsonl" \
  --preferences "analysis/${RUN}_preferences.jsonl" \
  --iterations 1000 \
  --out "analysis/${RUN}_bootstrap.json"

python -m acta.eval.make_recap_splits \
  --preferences "analysis/${RUN}_preferences.jsonl" \
  --split-by seed \
  --train-frac 0.6 \
  --valid-frac 0.1 \
  --test-frac 0.3 \
  --out-dir "analysis/${RUN}_splits_t30"

echo "[$(date -Is)] stage=reranker_baselines"
if [[ "${RUN_SIMPLE_BASELINES:-0}" == "1" ]]; then
  python -m acta.models.eval_random_reranker \
    --test "analysis/${RUN}_splits_t30/test.jsonl" \
    --seed 0 \
    --out-predictions "analysis/${RUN}_random_predictions_t30.jsonl" \
    --out "analysis/${RUN}_random_eval_t30.json"
fi

python -m acta.models.train_exact_memory_reranker \
  --train "analysis/${RUN}_splits_t30/train.jsonl" \
  --out "${MODEL_ROOT}/${RUN}_exact_memory_t30.json"

python -m acta.models.eval_exact_memory_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_exact_memory_t30.json" \
  --out-predictions "analysis/${RUN}_exact_memory_predictions_t30.jsonl" \
  --out "analysis/${RUN}_exact_memory_eval_t30.json"

python -m acta.models.train_nn_reranker \
  --train "analysis/${RUN}_splits_t30/train.jsonl" \
  --valid "analysis/${RUN}_splits_t30/valid.jsonl" \
  --out "${MODEL_ROOT}/${RUN}_nn_reranker_t30.json"

python -m acta.models.eval_nn_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_nn_reranker_t30.json" \
  --out-predictions "analysis/${RUN}_nn_predictions_t30.jsonl" \
  --out "analysis/${RUN}_nn_eval_t30.json"

python -m acta.models.train_action_reranker \
  --train "analysis/${RUN}_splits_t30/train.jsonl" \
  --valid "analysis/${RUN}_splits_t30/valid.jsonl" \
  --out "${MODEL_ROOT}/${RUN}_feature_reranker_t30.json"

python -m acta.models.eval_action_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_feature_reranker_t30.json" \
  --out-predictions "analysis/${RUN}_feature_predictions_t30.jsonl" \
  --out "analysis/${RUN}_feature_eval_t30.json"

python -m acta.models.eval_action_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_feature_reranker_t30.json" \
  --abstain-margin 5.0 \
  --out-predictions "analysis/${RUN}_feature_predictions_t30_margin50.jsonl" \
  --out "analysis/${RUN}_feature_eval_t30_margin50.json"

if [[ "${RUN_SIMPLE_BASELINES:-0}" == "1" ]]; then
  for strategy in navigation-prior anti-static learned-verb-prior; do
    extra_train=()
    stem="${strategy//-/_}"
    if [[ "$strategy" == "learned-verb-prior" ]]; then
      extra_train=(--train "analysis/${RUN}_splits_t30/train.jsonl")
      stem="verb_prior"
    fi
    python -m acta.models.eval_heuristic_reranker \
      "${extra_train[@]}" \
      --test "analysis/${RUN}_splits_t30/test.jsonl" \
      --strategy "$strategy" \
      --out-predictions "analysis/${RUN}_${stem}_predictions_t30.jsonl" \
      --out "analysis/${RUN}_${stem}_eval_t30.json"
  done
fi

echo "[$(date -Is)] stage=advanced_rerankers"
python -m acta.models.train_sklearn_reranker \
  --train "analysis/${RUN}_splits_t30/train.jsonl" \
  --model-type mlp \
  --out "${MODEL_ROOT}/${RUN}_mlp_reranker_t30.joblib"

python -m acta.models.eval_sklearn_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_mlp_reranker_t30.joblib" \
  --out-predictions "analysis/${RUN}_mlp_predictions_t30.jsonl" \
  --out "analysis/${RUN}_mlp_eval_t30.json"

python -m acta.models.train_sklearn_reranker \
  --train "analysis/${RUN}_splits_t30/train.jsonl" \
  --model-type gradient-boosting \
  --out "${MODEL_ROOT}/${RUN}_gbdt_reranker_t30.joblib"

python -m acta.models.eval_sklearn_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_gbdt_reranker_t30.joblib" \
  --out-predictions "analysis/${RUN}_gbdt_predictions_t30.jsonl" \
  --out "analysis/${RUN}_gbdt_eval_t30.json"

python -m acta.models.train_embedding_reranker \
  --train "analysis/${RUN}_splits_t30/train.jsonl" \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --local-files-only \
  --batch-size 64 \
  --out "${MODEL_ROOT}/${RUN}_embedding_minilm_t30.joblib"

python -m acta.models.eval_embedding_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_embedding_minilm_t30.joblib" \
  --batch-size 64 \
  --out-predictions "analysis/${RUN}_embedding_minilm_predictions_t30.jsonl" \
  --out "analysis/${RUN}_embedding_minilm_eval_t30.json"

python -m acta.models.train_policy_reranker \
  --train "analysis/${RUN}_splits_t30/train.jsonl" \
  --valid "analysis/${RUN}_splits_t30/valid.jsonl" \
  --hidden-dim 128 \
  --num-layers 2 \
  --dropout 0.05 \
  --epochs 400 \
  --learning-rate 0.0008 \
  --entropy-coef 0.005 \
  --out "${MODEL_ROOT}/${RUN}_support_policy_t30.pt"

python -m acta.models.eval_policy_reranker \
  --test "analysis/${RUN}_splits_t30/test.jsonl" \
  --model "${MODEL_ROOT}/${RUN}_support_policy_t30.pt" \
  --out-predictions "analysis/${RUN}_support_policy_predictions_t30.jsonl" \
  --out "analysis/${RUN}_support_policy_eval_t30.json"

if [[ "${RUN_BGE_FINETUNE:-0}" == "1" ]]; then
  BGE_BASE="${BGE_BASE:-/home/hanyu/models/recap/hf/bge-reranker-base}"
  BGE_OUT="${BGE_OUT:-/home/hanyu/models/recap/hf/bge-reranker-recap-t30}"
  WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
  python -m acta.models.train_cross_encoder_reranker \
    --train "analysis/${RUN}_splits_t30/train.jsonl" \
    --base-model "${BGE_BASE}" \
    --epochs "${BGE_EPOCHS:-3}" \
    --batch-size "${BGE_BATCH_SIZE:-4}" \
    --learning-rate "${BGE_LR:-1e-5}" \
    --max-length 256 \
    --out "${BGE_OUT}" \
    --summary-out "analysis/${RUN}_bge_finetune_summary_t30.json"

  python -m acta.models.eval_cross_encoder_reranker \
    --test "analysis/${RUN}_splits_t30/test.jsonl" \
    --model "${BGE_OUT}" \
    --batch-size 8 \
    --out-predictions "analysis/${RUN}_bge_reranker_finetuned_predictions_t30.jsonl" \
    --out "analysis/${RUN}_bge_reranker_finetuned_eval_t30.json"
fi

echo "[$(date -Is)] stage=retention_safety"
python -m acta.eval.build_success_retention_dataset \
  --trajectories "analysis/${RUN}_trajectories.jsonl" \
  --out "analysis/${RUN}_success_retention.jsonl"

for margin in 0.0 2.0 5.0; do
  tag="$(python - <<PY
margin=${margin}
print(f"margin{int(margin * 10):02d}")
PY
)"
  python -m acta.models.eval_action_reranker \
    --test "analysis/${RUN}_success_retention.jsonl" \
    --model "${MODEL_ROOT}/${RUN}_feature_reranker_t30.json" \
    --abstain-margin "$margin" \
    --out-predictions "analysis/${RUN}_retention_feature_predictions_${tag}.jsonl" \
    --out "analysis/${RUN}_retention_feature_eval_${tag}.json"
  python -m acta.eval.eval_gold_demotion \
    --retention "analysis/${RUN}_success_retention.jsonl" \
    --predictions "analysis/${RUN}_retention_feature_predictions_${tag}.jsonl" \
    --out "analysis/${RUN}_gold_demotion_feature_${tag}.json"
done

echo "[$(date -Is)] stage=done"
