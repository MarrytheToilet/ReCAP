# Experiments

Run commands from the repository root with Python 3.10 or newer. The manuscript
and appendix specify the datasets, comparisons, and reported configurations.
Manuscript sources are maintained separately from the public source tree. `analysis/` is
local, ignored input storage; a source checkout alone does not supply its saved
trajectories, game lists, or weights. Put new outputs under
`results/experiments/` or `results/benchmarks/`.

## Setup and tests

```bash
python -m pip install -e '.[test,textworld]'
python -m pip install torch numpy
python -m pytest -q
```

Local LM training/evaluation also requires `transformers` and `peft`. Install
ALFWorld or ScienceWorld separately when collecting or replaying those
environments. The ScienceWorld ranking command below uses saved preferences
and does not launch the simulator.

## Compile and rank logged candidates

Given `analysis/trajectories.jsonl` and its replayable games, this example
compiles preferences and ledgers, makes task-disjoint splits, trains a structured
scorer, and evaluates witness recovery:

```bash
python -m recap.eval.compile_recap_batch \
  --trajectories analysis/trajectories.jsonl --source policy-repair \
  --out-preferences results/experiments/ranking/preferences.jsonl \
  --out-ledger results/experiments/ranking/ledger.jsonl \
  --out-trajectory-ledger results/experiments/ranking/trajectory_ledger.jsonl \
  --out-summary results/experiments/ranking/compile_summary.json

python -m recap.eval.make_recap_splits \
  --preferences results/experiments/ranking/preferences.jsonl \
  --split-by task_id --out-dir results/experiments/ranking/splits

python -m recap.models.train_policy_reranker \
  --train results/experiments/ranking/splits/train.jsonl \
  --valid results/experiments/ranking/splits/valid.jsonl \
  --hidden-dim 128 --num-layers 2 --dropout 0.05 --epochs 400 \
  --learning-rate 0.0008 --entropy-coef 0.005 --weight-decay 0.0001 \
  --soft-target-weight 0 --seed 0 \
  --out results/experiments/ranking/policy.pt

python -m recap.models.eval_policy_reranker \
  --test results/experiments/ranking/splits/test.jsonl \
  --model results/experiments/ranking/policy.pt \
  --out-predictions results/experiments/ranking/predictions.jsonl \
  --out results/experiments/ranking/metrics.json
```

This is a workflow example. Exact TextWorld replication uses the saved
212/38/105 train/validation/test preferences grouped by game seed and trajectory.
A newly generated split need not reproduce the reported 0.927 MRR. Strict paired
cost auditing is available through `python -m recap.eval.eval_strict_certificate
--help`.

## Local policies

Use `python -m recap.models.train_lm_candidate_policy --help` for the training
interface. Local-policy experiments use separate planner-derived shaped rewards
and a wider admissible pool. The Gemma adapter was initialized from an earlier
candidate-policy checkpoint and trained for one epoch on 1,800 records, with
learning rate 0.0002 and no retention records. Reward-softmax, pairwise, and
direct-cost arms have distinct objectives; follow the appendix for each recipe.

With a local base checkpoint at `models/base`, an adapter at `models/adapter`,
and the saved game lists at
`analysis/localpolicy_eval/ctrl_{hard,xhard}100_games.txt`:

```bash
python -m scripts.eval_local_policy \
  --model models/base --adapter models/adapter \
  --out-dir results/experiments/local_policy --episode-timeout 600
```

The evaluator runs up to 30 actions per game and writes episode JSONL and a
summary. Omit `--adapter` for a base-model comparison; use `--rank 2` for the
second-ranked-action control and a separate output directory for each condition.
`python -m scripts.serve_local_policy --help` describes the local serving entry
point.

## ScienceWorld: fixed task-disjoint protocol

Required local inputs:

- `analysis/scienceworld_easy_api24_top15_tuned_preferences.jsonl`
- `analysis/scienceworld_easy_api24_strict_certificate_rows.jsonl`

Run the standalone evaluator:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.eval_scienceworld_crossfit \
  --workers 4 --out results/benchmarks/scienceworld_crossfit
```

The corpus contains 78 branch-certified witnesses across 19 task instances in
boiling, freezing, melting, and thermometer use. Sorted task IDs are shuffled
with seed 20260928 and assigned round-robin to five folds. Each fold is tested
once; the next fold supplies validation and the other three supply training.
Training seeds are 0, 1, and 2. The fixed head has two 128-unit hidden layers,
dropout 0.05, 400 epochs, learning rate 0.0008, weight decay 0.0001, and entropy
coefficient 0.005. Validation loss selects checkpoints.

Inference receives only candidates, preceding actions, and the executed action.
Rank-only and shuffled-training-target controls use seed zero and the same
folds. Results measure held-out task-instance witness recovery within the four
families.

| Method | MRR |
|---|---:|
| Raw order | 0.173 |
| Move rank 2 first | 0.192 |
| Rank-only learner | 0.275 |
| Shuffled-target control | 0.270 |
| ReCAP, mean of three seeds | **0.411** |

ReCAP top-1 recovery is 22.2%; the 72 strict witnesses yield 0.399 MRR. The paired
MRR gain is 0.238, with a task-cluster bootstrap 95% interval of [0.151, 0.344]
from 2,000 draws, averaging training seeds within each decision and conditioning
on the fitted folds.

The evaluator writes `manifest.json` with the input hash, folds, and fixed
configuration before training; `fold*/` contains checkpoints and predictions.
`all_predictions.json` and `results.json` contain pooled predictions and metrics.

## Plot saved results

```bash
python -m scripts.plot_results --analysis analysis --out results/figures
python -m scripts.summarize_local_policies
python -m scripts.summarize_interactive_results
```

These commands use local saved results and make no API calls. Generated data,
figures, weights, logs, and manuscript files are excluded from Git.

## Freeze validation-selected intervention gates

After collecting all validation runs, select gates without reading test
trajectories:

```bash
python -m scripts.calibrate_browser_selector \
  --root results/benchmarks/miniwob
python -m scripts.calibrate_tau2_selector \
  --root results/experiments/new_benchmarks
```

Both commands reject incomplete validation batches. They use the training
artifact and saved validation outcomes, save the chosen selector and freeze
metadata, and do not launch collection or evaluation jobs. Pass `--output` to
write calibrated artifacts to a separate directory.
