# Reproducing the review-response analyses

This documents every number added to the paper during the pre-review revision:
the strict paired-cost certificate, its robustness checks, the online
verification-cost accounting, and the closed-loop attribution controls.
Steps 1–5 are CPU-only recomputation from logged artifacts (no environment
replays, no API calls). Step 6 needs one ~24GB GPU for roughly a day.

Inputs assumed present under `analysis/` (produced by the original xhard-700
pipeline): `recap_xhard_700_mimo25_t1_top5_preferences.jsonl` (355
preferences), `recap_xhard_*trace_report.json` (per-prefix verifier suffixes),
`recap_xhard_700_mimo25_t1_top5_splits_t30/` (212/38/105 splits), the saved
`*_predictions_t30.jsonl` files, the online episode logs, and
`recap_xhard_700_online_policy_progress_pool60.jsonl` (5,000 KLPI records).

## 1. Strict paired-cost certificate audit

```bash
python -m recap.eval.eval_strict_certificate \
  --preferences analysis/recap_xhard_700_mimo25_t1_top5_preferences.jsonl \
  --trace-reports 'analysis/recap_xhard_*trace_report.json' \
  --out-rows analysis/recap_xhard_700_mimo25_t1_top5_strict_certificate_rows.jsonl \
  --out-summary analysis/recap_xhard_700_mimo25_t1_top5_strict_certificate.json
```

Expected: 354/355 audited; strict pass 347 (δ=0), **346 (δ=1, 97.7%)**,
178 (δ=2, 50.3%); Δ-histogram {−2: 7, 0: 1, +1: 168, +2: 177, +10: 1}.
Unit tests: `pytest tests/test_strict_certificate.py`.

## 2. Verifier-consistency (non-tautology + noise bound)

```bash
python scripts/audit_verifier_consistency.py
```

Expected: 991 consecutive-step pairs; 8 violations (0.81%); drop histogram
{1: 479, 0: 274, −1: 229, 3: 8, −9: 1} — 48% of executed actions make one
step of verified progress, all 7 negative-Δ preference cases are provable
verifier suboptimality (drop 3 > legal max 1).

## 3. Strict-subset held-out metrics (no retraining)

```bash
python scripts/eval_strict_subset_metrics.py
```

Expected (all 105 → strict 101): support policy 0.927/0.857 → 0.924/0.851;
raw 0.442/0.000 → 0.440/0.000; BGE SFT 0.840/0.686 → 0.838/0.683; structured
0.917/0.838 → 0.919/0.842; rank-only 0.835/0.705 → 0.828/0.693; no-rank
0.779/0.581 → 0.777/0.574.

## 4. Strict-only retraining (CPU, minutes)

```bash
python scripts/make_strict_splits.py       # 212/38/105 -> 208/37/101
python -m recap.models.train_policy_reranker \
  --train analysis/recap_xhard_700_mimo25_t1_top5_splits_t30_strict/train.jsonl \
  --valid analysis/recap_xhard_700_mimo25_t1_top5_splits_t30_strict/valid.jsonl \
  --hidden-dim 128 --num-layers 2 --dropout 0.05 --epochs 400 \
  --learning-rate 0.0008 --entropy-coef 0.005 --seed 0 \
  --out models/recap_xhard_700_strict_only_policy.pt
python -m recap.models.eval_policy_reranker \
  --test analysis/recap_xhard_700_mimo25_t1_top5_splits_t30_strict/test.jsonl \
  --model models/recap_xhard_700_strict_only_policy.pt \
  --out analysis/recap_xhard_700_strict_only_policy_eval.json
```

Expected: 0.924 MRR / 0.851 top-1 on the strict test set; evaluating the same
model on the full 105-preference test set gives 0.927 / 0.857 — identical to
the full-label policy to three decimals.

## 5. Online verification-cost accounting

```bash
python scripts/count_online_verifications.py
```

Expected: verified proposal 100 eps, success 0.96, 7.28 steps/ep,
**1.98 verifications/step (14.4/ep)**; full replay 100 eps, success 0.97,
7.14 steps/ep, **4.63/step (33.1/ep)**. Cost model per
`recap/controllers/replay_repair_controller.py`: verified proposal checks the
raw action plus one learned proposal; full replay checks every logged
candidate.

## 6. Closed-loop attribution controls (GPU, ~1.5h/arm train + ~2h/eval)

```bash
# (a) control datasets — deterministic, byte-identical regeneration (seed 42)
python scripts/build_klpi_control_datasets.py

# (b) train three arms sequentially (identical pipeline, only rewards differ)
bash scripts/run_klpi_controls.sh

# (c) evaluate each arm on 100 hard + 100 xhard games; starts each arm's
#     eval as soon as its adapter appears, overlapping with training
bash scripts/run_klpi_ctrl_evals_chained.sh
```

Notes on fidelity to the deployed klpi_v3 run:
- The controls add `--candidate-chunk-size 2 --gradient-checkpointing`
  because klpi_v3's unchunked 512×16 forward OOMs on a 24GB card. The
  ReCAP-reward arm retrained under this memory-equivalent path reproduces the
  deployed policy (loss 1.384 vs 1.372; closed-loop 0.57/0.47 vs 0.55/0.48),
  which validates the comparison.
- The evaluation command was validated by an exact 20/20 per-game match
  against the deployed policy's logged hard episodes (SR 0.65, steps 16.8).
  The critical flag is `--lm-pool-ranker progress`; without it, the candidate
  pool the policy sees is ordered differently and success collapses to the
  base level.

Expected final table (100 games per difficulty):

| Supervision      | Loss | hard Succ./Steps | xhard Succ./Steps |
|------------------|-----:|-----------------:|------------------:|
| ReCAP rewards    | 1.38 |      0.57 / 18.4 |       0.47 / 22.5 |
| Loop-only rules  | 2.39 |      0.01 / 29.7 |       0.00 / 30.0 |
| Shuffled rewards | 3.13 |      0.00 / 30.0 |       0.00 / 30.0 |

All headline numbers are also collected in
`analysis/recap_xhard_700_mimo25_t1_top5_strict_ledger_summary.json`.
