# ActA / ReCAP Bootstrap

Current research direction:

```text
ReCAP: Replay-Certified Candidate Action Preferences
```

The active paper claim is no longer broad action algebra or generic failed-trace
preference supervision. ReCAP targets candidate ranking failures: cases where a
repairing action is already present in the LLM agent's own logged top-k
candidates but is ranked below the executed action. Environment replay certifies
the local repair and converts it into candidate-level reranker supervision.

See the current plan:

```text
analysis/recap_research_plan.md
```

This repo currently contains the first executable slice of ActA:

- deterministic environment replay interface
- state signatures
- action-pair probing
- JSONL relation records
- toy adapter tests
- TextWorld adapter and smoke dataset builder
- counterfactual trace-edit compiler
- replay-certified candidate-level preference extraction
- ReCAP-SFT cross-encoder reranker and offline candidate-policy reranker

## Verify The Core Probe

```bash
pytest -q
```

## Build Toy Records

```bash
python -m acta.data.build_relation_dataset --out data/relation_records.jsonl
```

## Build TextWorld Records

Install TextWorld:

```bash
python -m pip install textworld
```

Generate a small deterministic game:

```bash
mkdir -p data/textworld_games
tw-make custom --world-size 3 --nb-objects 8 --quest-length 3 --seed 11 --output data/textworld_games/acta_seed11.z8 --force --silent
```

Probe action relations:

```bash
python -m acta.data.build_textworld_relation_dataset data/textworld_games/acta_seed11.z8 --out data/textworld_relation_records.jsonl --seed 0 --num-prefixes 8 --max-pairs-per-prefix 12
```

Probe a larger effectful-action slice:

```bash
python -m acta.data.build_textworld_relation_dataset data/textworld_games/acta_seed11.z8 --out data/textworld_relation_records_effectful.jsonl --seed 0 --num-prefixes 100 --max-depth 8 --max-pairs-per-prefix 10 --max-records 1000 --pair-filter both-effectful
```

Summarize relation records:

```bash
python -m acta.data.summarize_relations data/textworld_relation_records.jsonl
```

Audit signature diffs behind labels:

```bash
python -m acta.data.audit_signatures data/textworld_relation_records_effectful.jsonl --relation commute --value false
```

Build a small multi-seed TextWorld dataset:

```bash
python -m acta.data.build_textworld_batch --game-seeds 11 12 13 --num-prefixes 40 --max-records-per-game 400 --workers 3 --merged-out data/textworld_multi_seed.jsonl
```

Generate a harder TextWorld evaluation suite:

```bash
python -m acta.data.generate_textworld_games \
  --seeds 101 102 103 104 105 \
  --out-dir data/textworld_hard_games \
  --prefix acta_hard_seed \
  --world-size 6 \
  --nb-objects 20 \
  --quest-length 6 \
  --force
```

Create a relation existence table:

```bash
python -m acta.data.relation_existence_table data/textworld_multi_seed.jsonl --out analysis/textworld_relation_existence.md
```

Evaluate normal-form rewriting on sampled traces:

```bash
python -m acta.eval.eval_normal_form data/textworld_multi_seed.jsonl --limit 100
```

Evaluate a random agent with and without ActA control:

```bash
python -m acta.eval.eval_agent data/textworld_games/acta_seed11.z8 data/textworld_games/acta_seed12.z8 --controller none --max-steps 20
python -m acta.eval.eval_agent data/textworld_games/acta_seed11.z8 data/textworld_games/acta_seed12.z8 --controller acta --max-steps 20 --max-candidates 8
python -m acta.eval.eval_agent data/textworld_games/acta_seed11.z8 data/textworld_games/acta_seed12.z8 --controller acta --fast-controller --max-steps 20 --max-candidates 8
```

Run the LLM-agent flow without external API calls:

```bash
python -m acta.eval.eval_agent data/textworld_games/acta_seed11.z8 --agent mock-llm --controller acta --fast-controller --max-candidates 5 --max-steps 12
```

Run with an OpenAI-compatible API after configuring `.env`:

```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY and ACTA_LLM_MODEL
python -m acta.eval.eval_agent data/textworld_games/acta_seed11.z8 --agent openai --controller acta --fast-controller --max-candidates 5 --max-steps 12
```

Stress-test reranking under structurally noisy candidates:

```bash
python -m acta.eval.eval_agent data/textworld_games/acta_seed11.z8 --agent openai --candidate-noise frontload-existing-structural --controller acta --fast-controller --max-candidates 5 --max-steps 12
```

For longer API-backed runs, write per-episode output and continue past transient API failures:

```bash
python -m acta.eval.eval_agent data/textworld_games/acta_seed{11..15}.z8 \
  --agent openai \
  --candidate-noise frontload-existing-structural \
  --controller acta \
  --fast-controller \
  --max-candidates 5 \
  --max-steps 6 \
  --llm-call-delay 2 \
  --episode-delay 5 \
  --continue-on-error \
  --episodes-out analysis/openai_noisy_existing_acta_fast_5games.episodes.jsonl \
  --out analysis/openai_noisy_existing_acta_fast_5games.json
```

Summarize gold-action recovery diagnostics when the environment exposes oracle policy commands:

```bash
python -m acta.eval.summarize_gold_diagnostics \
  analysis/textworld_hard_openai_base.json \
  analysis/textworld_hard_openai_acta_fast.json \
  --out analysis/textworld_hard_gold_diagnostics.md
```

Evaluate baseline top-1 vs ActA on the same oracle-path states:

```bash
python -m acta.eval.eval_gold_selector data/textworld_hard_games/*.z8 \
  --agent openai \
  --fast-controller \
  --max-candidates 5 \
  --max-steps 8 \
  --llm-call-delay 2 \
  --progress \
  --out analysis/textworld_hard_gold_selector_openai.json
```

Run an ActA ablation without recent-repeat penalties:

```bash
python -m acta.eval.eval_agent data/textworld_hard_games/*.z8 \
  --agent openai \
  --controller acta \
  --fast-controller \
  --recent-repeat-penalty 0 \
  --max-candidates 5 \
  --max-steps 20 \
  --out analysis/textworld_hard_openai_acta_no_repeat_penalty.json
```

Run a small tabular Q-learning comparison with ActA as an action prior:

```bash
python -m acta.eval.eval_rl data/textworld_games/acta_seed{11..15}.z8 \
  --prior acta-hard \
  --fast-controller \
  --episodes 30 \
  --max-steps 12 \
  --training-seeds 0 1 2 \
  --recent-repeat-penalty 0 \
  --out analysis/rl_textworld_easy_acta_hard_5games_3seeds.json

python -m acta.eval.compare_rl_runs \
  analysis/rl_textworld_easy_none_5games_3seeds.json \
  analysis/rl_textworld_easy_acta_soft_5games_3seeds.json \
  analysis/rl_textworld_easy_acta_hard_5games_3seeds.json \
  --out analysis/rl_textworld_easy_5games_3seeds_comparison.md
```

Compile an existing agent run into counterfactual trace-edit diagnostics:

```bash
python -m acta.eval.compile_traces analysis/textworld_hard_openai_base_5games.json \
  --progress \
  --out analysis/textworld_hard_openai_base_5games_trace_edits.json

python -m acta.eval.compare_trace_edits \
  analysis/textworld_hard_openai_base_5games_trace_edits.json \
  analysis/textworld_hard_openai_acta_fast_5games_trace_edits.json \
  --out analysis/textworld_hard_trace_edit_comparison.md
```

Rerun an API-backed agent with verified trace feedback:

```bash
python -m acta.eval.eval_agent data/textworld_games/acta_seed{11..15}.z8 \
  --agent openai \
  --feedback-report analysis/openai_noisy_existing_no_controller_5games_trace_edits.json \
  --feedback-mode safe \
  --feedback-source failures \
  --max-candidates 5 \
  --max-steps 10 \
  --llm-call-delay 2 \
  --progress \
  --out analysis/openai_trace_feedback_safe_5games.json
```

Run a full two-pass feedback cycle:

```bash
python -m acta.eval.eval_feedback_cycle data/textworld_games/acta_seed{11..15}.z8 \
  --agent openai \
  --feedback-mode safe \
  --max-candidates 5 \
  --max-steps 10 \
  --llm-call-delay 2 \
  --progress \
  --out analysis/openai_feedback_cycle_safe_5games.json
```

Build ReCAP candidate-level preferences from a failed run and trace report:

```bash
python -m acta.eval.build_action_preferences \
  analysis/openai_xhard_feedback_cycle_safe_3games.json \
  --run-key first_run \
  --trace-report analysis/openai_xhard_first_trace_report.json \
  --source policy-repair \
  --out analysis/openai_xhard_first_policy_repair_preferences.jsonl \
  --ledger-out analysis/openai_xhard_first_policy_repair_ledger.jsonl \
  --trajectory-ledger-out analysis/openai_xhard_first_policy_repair_trajectory_ledger.jsonl
```

Evaluate ReCAP preferences against recorded candidate lists:

```bash
python -m acta.eval.eval_action_preferences \
  analysis/openai_xhard_feedback_cycle_safe_3games.json \
  --run-key first_run \
  --preference-data analysis/openai_xhard_first_policy_repair_preferences.jsonl \
  --out analysis/openai_xhard_first_policy_repair_preference_eval.json
```

Summarize ReCAP failure decomposition from the ledger:

```bash
python -m acta.eval.eval_recap_failure_decomposition \
  --ledger analysis/openai_xhard_first_policy_repair_ledger.jsonl \
  --preferences analysis/openai_xhard_first_policy_repair_preferences.jsonl \
  --out analysis/openai_xhard_first_policy_repair_decomposition.json
```

Collect a pilot batch of failed TextWorld trajectories:

```bash
python -m acta.eval.collect_failure_trajectories \
  --env textworld \
  --difficulty xhard \
  --num-games 30 \
  --rollouts-per-game 1 \
  --agent openai \
  --top-k 5 \
  --max-steps 30 \
  --out analysis/recap_xhard_pilot_trajectories.jsonl
```

Run the batch-shaped ReCAP pipeline on an existing trajectory/run file:

```bash
python -m acta.eval.compile_recap_batch \
  --trajectories analysis/recap_xhard_pilot_trajectories.jsonl \
  --source policy-repair \
  --out-trace-report analysis/recap_xhard_pilot_trace_report.json \
  --out-preferences analysis/recap_xhard_pilot_preferences.jsonl \
  --out-ledger analysis/recap_xhard_pilot_ledger.jsonl \
  --out-trajectory-ledger analysis/recap_xhard_pilot_trajectory_ledger.jsonl \
  --out-summary analysis/recap_xhard_pilot_compile_summary.json
```

Evaluate raw/oracle/learned candidate-ranking metrics. Without a prediction file,
`learned_*` metrics intentionally remain null:

```bash
python -m acta.eval.eval_candidate_ranking \
  --preferences analysis/recap_xhard_pilot_preferences.jsonl \
  --out analysis/recap_xhard_pilot_candidate_ranking.json
```

Create leakage-checked train/valid/test splits once the preference dataset has
more than one game or seed:

```bash
python -m acta.eval.make_recap_splits \
  --preferences analysis/recap_xhard_pilot_preferences.jsonl \
  --split-by seed \
  --out-dir analysis/recap_xhard_pilot_splits
```

Train and evaluate the ReCAP rerankers. Model artifacts should live outside the
repo under `../../models/recap`:

```bash
python -m acta.models.train_cross_encoder_reranker \
  --train analysis/recap_xhard_pilot_splits/train.jsonl \
  --valid analysis/recap_xhard_pilot_splits/valid.jsonl \
  --base-model ../../models/recap/hf/bge-reranker-base \
  --out ../../models/recap/hf/bge-reranker-recap-pilot

python -m acta.models.eval_cross_encoder_reranker \
  --test analysis/recap_xhard_pilot_splits/test.jsonl \
  --model ../../models/recap/hf/bge-reranker-recap-pilot \
  --out-predictions analysis/recap_xhard_pilot_bge_predictions.jsonl \
  --out analysis/recap_xhard_pilot_bge_eval.json

python -m acta.models.train_policy_reranker \
  --train analysis/recap_xhard_pilot_splits/train.jsonl \
  --out ../../models/recap/artifacts/recap_policy_reranker_pilot.pt

python -m acta.models.eval_policy_reranker \
  --test analysis/recap_xhard_pilot_splits/test.jsonl \
  --model ../../models/recap/artifacts/recap_policy_reranker_pilot.pt \
  --out-predictions analysis/recap_xhard_pilot_policy_predictions.jsonl \
  --out analysis/recap_xhard_pilot_policy_eval.json
```

The structured feature reranker remains useful as a diagnostic/ablation model,
but the paper's main learned-model story is BGE-ReCAP-SFT plus ReCAP-PG.

The first target is not agent performance yet. The immediate acceptance check is that deterministic replay holds and relation records contain a useful mix of `commute=True`, `commute=False`, and `commute=null`.
