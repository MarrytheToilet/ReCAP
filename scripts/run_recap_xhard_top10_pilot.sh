#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p analysis/logs data/textworld_xhard_games

MODEL="${RECAP_LLM_MODEL:-$(python - <<'PY'
from pathlib import Path
model = "mimo-v2.5"
env = Path(".env")
if env.exists():
    for line in env.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("RECAP_LLM_MODEL") and "=" in stripped:
            model = stripped.split("=", 1)[1].strip().strip('"').strip("'")
print(model)
PY
)}"
RUN="recap_xhard_top10_pilot_901_1000"

rm -f \
  "analysis/${RUN}_trace_report.json" \
  "analysis/${RUN}_preferences.jsonl" \
  "analysis/${RUN}_ledger.jsonl" \
  "analysis/${RUN}_trajectory_ledger.jsonl" \
  "analysis/${RUN}_compile_summary.json" \
  "analysis/${RUN}_decomposition.json" \
  "analysis/${RUN}_candidate_ranking.json" \
  "analysis/${RUN}_sensitivity.json"

echo "[$(date -Is)] stage=generate_games seeds=901-1000"
python -m recap.data.generate_textworld_games \
  --seeds $(seq 901 1000) \
  --out-dir data/textworld_xhard_games \
  --prefix recap_xhard_seed \
  --world-size 8 \
  --nb-objects 30 \
  --quest-length 8

echo "[$(date -Is)] stage=collect_trajectories model=${MODEL} top_k=10"
python -m recap.eval.collect_failure_trajectories \
  --env textworld \
  --difficulty xhard \
  --game-dir data/textworld_xhard_games \
  --min-game-seed 901 \
  --max-game-seed 1000 \
  --rollouts-per-game 1 \
  --agent openai \
  --model "${MODEL}" \
  --temperature 1.0 \
  --top-k 10 \
  --max-steps 12 \
  --include-successes \
  --continue-on-error \
  --write-errors \
  --workers "${COLLECT_WORKERS:-4}" \
  --llm-timeout "${LLM_TIMEOUT:-120}" \
  --llm-max-retries 6 \
  --llm-retry-base-delay 2.0 \
  --progress \
  --out "analysis/${RUN}_trajectories.jsonl"

echo "[$(date -Is)] stage=compile_recap"
python -m recap.eval.compile_recap_batch \
  --trajectories "analysis/${RUN}_trajectories.jsonl" \
  --out-trace-report "analysis/${RUN}_trace_report.json" \
  --out-preferences "analysis/${RUN}_preferences.jsonl" \
  --out-ledger "analysis/${RUN}_ledger.jsonl" \
  --out-trajectory-ledger "analysis/${RUN}_trajectory_ledger.jsonl" \
  --out-summary "analysis/${RUN}_compile_summary.json" \
  --workers "${COMPILE_WORKERS:-8}" \
  --trace-progress

python -m recap.eval.eval_recap_failure_decomposition \
  --ledger "analysis/${RUN}_ledger.jsonl" \
  --preferences "analysis/${RUN}_preferences.jsonl" \
  --out "analysis/${RUN}_decomposition.json"

python -m recap.eval.eval_candidate_ranking \
  --preferences "analysis/${RUN}_preferences.jsonl" \
  --out "analysis/${RUN}_candidate_ranking.json"

python -m recap.eval.eval_recap_sensitivity \
  --ledger "analysis/${RUN}_ledger.jsonl" \
  --top-k 2 3 5 8 10 \
  --suffix-budget 3 5 7 \
  --out "analysis/${RUN}_sensitivity.json"

echo "[$(date -Is)] stage=done"
