#!/bin/bash
# KLPI closed-loop attribution controls (reviewer request).
# Three arms trained with IDENTICAL memory-safe settings so the comparison is
# same-pipeline: only the supervision source differs.
#   recap    - original candidate rewards (same-pipeline ReCAP reference arm)
#   shuffled - rewards permuted within each step's candidate list (seed 42)
#   looponly - rewards from structural rules only (repeat/no-op/inverse -> -0.8, else +0.4)
# Memory-safe deltas vs the original klpi_v3 run: --candidate-chunk-size 2 and
# --gradient-checkpointing (klpi_v3's 512x16 unchunked forward OOMs on a 24GB card).
set -e
cd /home/hanyu/research/ReCAP
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_one () {
  local name=$1 data=$2
  python -m recap.models.train_lm_candidate_policy \
    --train "$data" \
    --train-limit 1800 \
    --base-model /home/hanyu/models/gemma-2-2b-it-ms \
    --init-adapter /home/hanyu/models/recap/hf/gemma-2-2b-recap-online-pool12-traj2500 \
    --out /home/hanyu/models/recap/hf/gemma-2-2b-recap-klpi-ctrl-${name} \
    --summary-out analysis/localpolicy_eval/klpi_ctrl_${name}_train.json \
    --epochs 1 --learning-rate 2e-4 --batch-accum 8 --max-length 512 \
    --max-history 12 --max-observation-chars 360 \
    --candidate-chunk-size 2 --gradient-checkpointing \
    --max-train-candidates 16 --kl-coef 0.1 --keep-coef 0.0 \
    --rank-prior-temperature 1.0 --use-candidate-rewards --reward-loss softmax \
    --reward-temperature 0.8 --expected-reward-coef 0.5 --pairwise-advantage-coef 0.3 \
    --advantage-margin 0.0 --entropy-coef 0.01 --load-in-4bit \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --target-modules all-linear \
    --seed 0 > analysis/localpolicy_eval/klpi_ctrl_${name}_train.log 2>&1
  echo "[done] ${name} $(date +%H:%M:%S)"
}

train_one shuffled analysis/recap_xhard_700_online_policy_pool60_shuffled.jsonl
train_one looponly analysis/recap_xhard_700_online_policy_pool60_looponly.jsonl
train_one recap    analysis/recap_xhard_700_online_policy_progress_pool60.jsonl
echo "ALL TRAINING DONE"
