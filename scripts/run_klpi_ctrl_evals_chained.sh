#!/bin/bash
# Chained variant: start each arm's hard-100 eval as soon as that arm's
# training summary appears, so evals overlap with the remaining training.
set -e
cd /home/hanyu/research/ReCAP
D=analysis/localpolicy_eval

eval_one () {
  local name=$1 games=$2 tag=$3
  python -m recap.eval.eval_agent $(cat ${D}/${games} | tr '\n' ' ') \
    --agent lm-policy --model /home/hanyu/models/gemma-2-2b-it-ms \
    --lm-adapter /home/hanyu/models/recap/hf/gemma-2-2b-recap-klpi-ctrl-${name} \
    --lm-load-in-4bit --max-steps 30 --max-candidates 5 --controller none --seed 0 \
    --lm-pool-ranker progress \
    --out ${D}/klpi_ctrl_${name}_${tag}.json \
    --episodes-out ${D}/klpi_ctrl_${name}_${tag}.episodes.jsonl \
    --continue-on-error > ${D}/klpi_ctrl_${name}_${tag}.log 2>&1
  python3 - "$D/klpi_ctrl_${name}_${tag}.episodes.jsonl" <<'PY'
import json,sys
eps=[json.loads(l) for l in open(sys.argv[1])]
sr=sum(1 for e in eps if e.get('success'))/max(len(eps),1)
st=sum(len(e.get('steps') or []) for e in eps)/max(len(eps),1)
print(f"[eval done] {sys.argv[1].split('/')[-1]}: n={len(eps)} success={sr:.2f} steps={st:.1f}")
PY
}

for name in shuffled looponly recap; do
  until [ -f ${D}/klpi_ctrl_${name}_train.json ]; do sleep 60; done
  echo "[adapter ready] ${name} $(date +%H:%M:%S)"
  eval_one ${name} ctrl_hard100_games.txt hard100
done
for name in shuffled looponly recap; do
  eval_one ${name} ctrl_xhard100_games.txt xhard100
done
echo "ALL CTRL EVALS DONE"
