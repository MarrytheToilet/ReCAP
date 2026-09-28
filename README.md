# Hidden in Plain Sight: Replay-Certified Preference Learning over an Agent's Own Candidates

ReCAP learns from useful actions that an agent generated but ranked below its
executed choice. It preserves the logged candidate set, replays alternative
continuations, and compiles preferences with a ledger accounting for every
examined decision. Paired replay certifies strict improvements under a declared
outcome or cost comparison; outside-support witnesses identify opportunities
that require broader action generation.


## Method and experimental settings

- **Logged-candidate ranking:** a support-constrained scorer learns the
  compiler-selected witness using listwise cross-entropy with entropy
  regularization. It ranks only the original candidates. The ledger distinguishes
  strict improvements, non-strict witnesses, outside-support repairs, unchanged
  actions, unresolved decisions, and invalid replay.
- **Local-policy training:** a language model scores a wider admissible-action
  pool and acts directly. The Gemma experiment uses 1,800 separate records with
  privileged planner-path-derived shaped rewards and reward-softmax training,
  initialized from an earlier candidate-policy adapter. Its supervision is
  distinct from the 355 failure-only preferences. The loss combines reward-target
  cross-entropy, expected standardized reward, and KL to a candidate-index prior.
- **Online verified proposals:** branch replay checks proposed substitutions
  before committing them. In the evaluated batch, a rank-2 proposal matches the
  learned proposal's success, establishing the contribution of verification.

## Results

The following results summarize the evaluated configurations. Ranking metrics measure
recovery of the designated witness on candidate-present examples; success
metrics measure complete episodes.

| Setting | Metric | Baseline | ReCAP |
|---|---|---:|---:|
| TextWorld xhard, 105 held-out preferences | MRR | 0.442 | **0.927** |
| TextWorld xhard, same preferences | Top-1 witness recovery | 0.0% | **85.7%** |
| Gemma-2-2B, 100 hard games | Success | 11% | **55%** |
| Gemma-2-2B, 100 xhard games | Success | 7% | **48%** |
| Qwen, long-goal inputs, 100 hard games | Success | 76% | **97%** |
| Qwen, long-goal inputs, 100 xhard games | Success | 67% | **90%** |
| Online verified proposals, 100 games | Success | 84% | **96%** |
| ALFWorld, 42 held-out preferences | MRR | 0.263 | **0.512** |
| ScienceWorld, five-fold task-disjoint ranking | MRR | 0.173 | **0.411** |

The TextWorld audit finds 346 strict improvements among 354 auditable repair
pairs, spanning 81 of 91 failed trajectories. Online verification rescues 12
paired failures without harming a success. ScienceWorld averages three training
seeds over 78 witnesses from 19 task instances in four families; its top-1
recovery is 22.2%. Cross-environment rerankers are trained within each environment.
Raw top-1 is zero by construction on the held-out repair sets.

## Installation and experiments

Use Python 3.10 or newer from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Install `.[textworld]` for TextWorld replay, `.[models]` for model workflows,
and `.[plot]` for result figures. Other environments require their own dependencies.
The base package does not install these optional runtimes.

See [docs/experiments.md](docs/experiments.md) for compilation, training,
evaluation, and the standalone fixed ScienceWorld protocol. Experiments require
the corresponding trajectories, game files, splits, and model weights. Existing
`analysis/` artifacts remain local and ignored.

## Repository layout

| Path | Contents |
|---|---|
| `recap/agents/`, `recap/controllers/` | Agent policies and replay-based controllers |
| `recap/envs/` | Environment adapters |
| `recap/eval/` | Rollouts, preference compilation, ledgers, and evaluation |
| `recap/models/`, `recap/data/` | Trainable scorers and dataset utilities |
| `scripts/` | Standalone experiment and audit entry points |
| `tests/` | Replay, policy, and agent-loop tests |
| `results/experiments/` | Local-policy and TextWorld experiment artifacts |
| `results/benchmarks/` | Cross-environment benchmark artifacts |
| `docs/` | Experiment guide |

Manuscript files under `paper/` are maintained locally and excluded from this repository.

Certificates depend on the environment's restoration procedure, continuation
interface, and declared comparison. The ledger preserves that scope alongside
the preferences used for learning.
