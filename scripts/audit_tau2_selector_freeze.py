"""Verify a frozen selector's exact fitting sources and official task separation."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('results/experiments/new_benchmarks')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    folder = ROOT/'tau2_learned_selector'
    freeze = json.loads((folder/'freeze.json').read_text())
    model = json.loads((folder/'proposed_selector.json').read_text())
    splits = json.loads((ROOT/'full_baseline_manifest.json').read_text())['splits']
    data = ROOT/'tau2_training_branches_uniform'
    violations = []
    counts = {'train': 0, 'valid': 0}
    tasks = {'train': set(), 'valid': set()}
    for name, expected in model['source_sha256'].items():
        path = data/name
        if digest(path) != expected:
            violations.append([name, 'source_hash'])
        row = json.loads(path.read_text())
        domain, task = row['domain'], row['task_id']
        if task not in splits[domain]['train'] or task in splits[domain]['test']:
            violations.append([name, 'official_split'])
        expected_fit = 'valid' if int(hashlib.sha256(('tau2-selector-v1:'+domain+':'+task).encode()).hexdigest()[:8],16)%5 == 0 else 'train'
        if row['fit_split'] != expected_fit:
            violations.append([name, 'fit_split'])
        counts[expected_fit] += 1
        tasks[expected_fit].add((domain, task))
        if row['status'] != 'completed' or row['simulation']['termination_reason'] == 'timeout' or not row.get('prefix_restoration_verified'):
            violations.append([name, 'branch_verification'])
    for name, expected in model['plan_sha256'].items():
        if digest(data/name) != expected:
            violations.append([name, 'plan_hash'])
    for key, path in [('feature_code_sha256', Path('scripts/tau2_pair_selector.py')), ('training_code_sha256', Path('scripts/train_tau2_pair_selector.py'))]:
        if digest(path) != model[key]:
            violations.append([str(path), 'code_hash'])
    for gate in freeze['gates']:
        if digest(Path(gate['artifact'])) != gate['sha256']:
            violations.append([gate['stage'], 'gate_hash'])
    if counts['train'] != model['train_pairs'] or tasks['train'] & tasks['valid']:
        violations.append(['training', 'count_or_overlap'])
    if freeze['stage'] == 'final_selector_frozen' and digest(folder/'selector.json') != freeze['selector_sha256']:
        violations.append(['selector', 'final_hash'])
    out = dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage=freeze['stage'], pair_counts=counts, distinct_tasks={k:len(v) for k,v in tasks.items()}, source_files=len(model['source_sha256']), plan_files=len(model['plan_sha256']), violations=violations, note='Audit verifies saved provenance, split membership and restoration flags; independent raw replay checks are recorded separately.')
    (folder/'provenance_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    if violations:
        raise RuntimeError('Frozen-selector provenance violation')

if __name__ == '__main__':
    main()
