"""Reproducible task-disjoint ScienceWorld witness-ranking evaluation.

Fixed original TextWorld hyperparameters; no test-based model selection.
Each task is tested once per training seed. This is conditional witness recovery,
not environment task success. Public inference inputs exclude repair metadata.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recap.models.policy_reranker import train_policy_reranker, load_policy_model, rank_candidates_with_model

SOURCE = Path('analysis/scienceworld_easy_api24_top15_tuned_preferences.jsonl')
STRICT = Path('analysis/scienceworld_easy_api24_strict_certificate_rows.jsonl')
HP = dict(hidden_dim=128, num_layers=2, dropout=0.05, epochs=400,
          learning_rate=0.0008, entropy_coef=0.005, weight_decay=0.0001,
          soft_target_weight=0.0)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key(row):
    return row['task_id'], row.get('seed', 0), row['step_index']


def public(row):
    # Executed action is observed, unlike the preferred action or replay cost.
    return dict(candidates=row['candidates'], history=row['history'],
                rejected_action=row['executed_action'])


def train_job(job):
    out, fold, arm, seed, rows, split = job
    folder = Path(out) / f'fold{fold}' / f'{arm}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    train = [r for r in rows if r['task_id'] in split['train']]
    valid = [r for r in rows if r['task_id'] in split['valid']]
    test = [r for r in rows if r['task_id'] in split['test']]
    if arm == 'shuffled':
        # Permute the target *rank* across training rows; map into each support.
        # No validation/test labels are shuffled or used to fit model weights.
        rng = random.Random(1000 + fold)
        targets = [r['candidates'].index(r['preferred_action']) for r in train]
        rng.shuffle(targets)
        train = [dict(r, preferred_action=r['candidates'][j % len(r['candidates'])])
                 for r, j in zip(train, targets)]
    train_clean = [dict(public(r), preferred_action=r['preferred_action']) for r in train]
    valid_clean = [dict(public(r), preferred_action=r['preferred_action']) for r in valid]
    model = train_policy_reranker(train_clean, validation_preferences=valid_clean,
                                 feature_set='rank-only' if arm == 'rank_only' else 'all',
                                 seed=seed, **HP)
    torch.save(model, folder / 'model.pt')
    loaded = load_policy_model(model)
    predictions = []
    for r in test:
        order = [a for a, _ in rank_candidates_with_model(public(r), loaded, model)]
        assert len(order) == len(set(order)) == len(r['candidates'])
        assert set(order) == set(r['candidates'])
        rank = order.index(r['preferred_action']) + 1
        predictions.append(dict(task_id=r['task_id'], seed=r.get('seed', 0),
                                step_index=r['step_index'], fold=fold, arm=arm,
                                training_seed=seed, ranked_actions=order,
                                preferred_action=r['preferred_action'], rank=rank,
                                mrr=1 / rank, top1=int(rank == 1)))
    (folder / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    (folder / 'training_summary.json').write_text(json.dumps(model['summary'], indent=2))
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('results/benchmarks/scienceworld_crossfit'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read(SOURCE)
    assert len({key(r) for r in rows}) == len(rows)
    for r in rows:
        assert r['rejected_action'] == r['executed_action']
        assert r['preferred_action'] in r['candidates']
        assert len(r['history']) == r['step_index']
        assert len(set(r['candidates'])) == len(r['candidates'])
    tasks = sorted({r['task_id'] for r in rows})
    rng = random.Random(20260928)
    rng.shuffle(tasks)
    folds = [tasks[i::5] for i in range(5)]
    splits = []
    for i in range(5):
        split = dict(test=folds[i], valid=folds[(i+1) % 5],
                     train=[t for j, f in enumerate(folds) if j not in {i, (i+1) % 5} for t in f])
        assert not (set(split['train']) & set(split['valid']) or
                    set(split['train']) & set(split['test']) or
                    set(split['valid']) & set(split['test']))
        splits.append(split)
    manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    protocol='five-fold cross-fitting by task_id; next fold validation; fixed original hyperparameters',
                    split_seed=20260928, training_seeds=[0, 1, 2], hyperparameters=HP,
                    scope='conditional branch-certified witness ranking; not closed-loop success',
                    inference_fields=['candidates', 'history', 'executed_action'],
                    splits=splits, preferences=len(rows), task_instances=len(tasks))
    # Write the complete protocol before any model is trained/evaluated.
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    jobs = [(str(args.out), i, arm, seed, rows, split)
            for i, split in enumerate(splits)
            for arm, seeds in [('recap', [0, 1, 2]), ('rank_only', [0]), ('shuffled', [0])]
            for seed in seeds]
    predictions = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(train_job, jobs):
            predictions.extend(result)
            r = result[0]
            print(f"finished fold={r['fold']} arm={r['arm']} seed={r['training_seed']}", flush=True)
    strict = {key(r) for r in read(STRICT) if r['status'] == 'strict_misrank'}
    results = {}
    for arm in ['recap', 'rank_only', 'shuffled']:
        selected = [r for r in predictions if r['arm'] == arm]
        per_seed = []
        for seed in sorted({r['training_seed'] for r in selected}):
            subset = [r for r in selected if r['training_seed'] == seed]
            assert {key(r) for r in subset} == {key(r) for r in rows}
            per_seed.append(dict(seed=seed, mrr=float(np.mean([r['mrr'] for r in subset])),
                                 top1=float(np.mean([r['top1'] for r in subset]))))
        strict_rows = [r for r in selected if key(r) in strict]
        results[arm] = dict(mrr=float(np.mean([r['mrr'] for r in selected])),
                            top1=float(np.mean([r['top1'] for r in selected])), per_seed=per_seed,
                            strict_mrr=float(np.mean([r['mrr'] for r in strict_rows])),
                            strict_top1=float(np.mean([r['top1'] for r in strict_rows])))
    def rank2_order(r):
        c = r['candidates']
        return [c[1], c[0], *c[2:]] if len(c) > 1 else c
    for name, orderfn in [('raw', lambda r: r['candidates']), ('rank2', rank2_order)]:
        ranks = [orderfn(r).index(r['preferred_action']) + 1 for r in rows]
        results[name] = dict(mrr=float(np.mean([1/r for r in ranks])),
                             top1=float(np.mean([r == 1 for r in ranks])))
    # Paired task-cluster bootstrap, averaging model seeds within each decision.
    # This conditions on the fitted folds; it is not a refitting bootstrap.
    by_key = {key(r): [] for r in rows}
    for r in predictions:
        if r['arm'] == 'recap':
            by_key[key(r)].append(r['mrr'])
    task_deltas = {t: [] for t in tasks}
    for r in rows:
        delta = np.mean(by_key[key(r)]) - 1/(r['candidates'].index(r['preferred_action'])+1)
        task_deltas[r['task_id']].append(delta)
    boot_rng = np.random.default_rng(20260928)
    boot = [float(np.mean([v for j in boot_rng.integers(0, len(tasks), len(tasks))
                           for v in task_deltas[tasks[j]]])) for _ in range(2000)]
    summary = dict(preferences=len(rows), task_instances=len(tasks), strict_preferences=len(strict),
                   results=results, paired_mrr_gain=results['recap']['mrr']-results['raw']['mrr'],
                   paired_task_bootstrap_ci95=np.quantile(boot, [.025,.975]).tolist(),
                   uncertainty='paired task-cluster bootstrap conditional on fitted folds; seeds averaged per decision')
    (args.out / 'all_predictions.json').write_text(json.dumps(predictions, indent=2))
    (args.out / 'results.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
