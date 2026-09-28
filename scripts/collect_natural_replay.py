"""Collect per-candidate verifier costs on all valid episodes, with resumable shards."""
from __future__ import annotations
import argparse, hashlib, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from recap.envs.textworld_adapter import TextWorldAdapter
from recap.eval.eval_repair_multiplicity import replay_candidate_with_policy_suffix

BASE = 'analysis/recap_xhard_700_mimo25_t1_top5_'

def collect(episode, output):
    path = Path(output)
    if path.exists():
        return len(json.loads(path.read_text())['records'])
    adapter = TextWorldAdapter()
    records, history = [], []
    try:
        for i, step in enumerate(episode['steps']):
            candidates = list(dict.fromkeys(step.get('candidates_before') or step.get('candidates') or []))
            costs, valid = {}, {}
            for action in candidates:
                result = replay_candidate_with_policy_suffix(adapter, episode['task_id'], int(episode.get('seed', 0)), tuple(history), action)
                costs[action] = result['suffix_len'] if result['success'] else None
                valid[action] = result['valid']
            if candidates:
                records.append(dict(task_id=episode['task_id'], seed=episode.get('seed',0), step_index=i,
                    trajectory_id=episode.get('trajectory_id'), history=list(history), candidates=candidates,
                    rejected_action=step['action'], executed_action=step['action'],
                    episode_success=episode['success'], gold_action=step.get('gold_action'),
                    candidate_costs=costs, candidate_replay_valid=valid))
            history.append(step['action'])
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(task_id=episode['task_id'],records=records)))
        temp.replace(path)
        return len(records)
    finally:
        adapter.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectories',default=BASE+'trajectories.jsonl')
    p.add_argument('--out-dir',default='results/experiments/natural')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int)
    a=p.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    eps=[json.loads(l) for l in open(a.trajectories)]
    valid=[e for e in eps if not e.get('error')]
    assert len({e['task_id'] for e in valid})==len(valid), 'multiple episodes per task need a richer shard key'
    split={}
    for s in ('train','valid','test'):
        for l in open(BASE+'splits_t30/'+s+'.jsonl'):
            task=json.loads(l)['task_id']
            assert task not in split or split[task]==s
            split[task]=s
    for e in valid:
        task=e['task_id']
        u=int(hashlib.sha256(('recap-natural-v1:'+task).encode()).hexdigest()[:8],16)/2**32
        split.setdefault(task,'train' if u<.6 else 'valid' if u<.7 else 'test')
    manifest=dict(source=a.trajectories,source_sha256=hashlib.sha256(Path(a.trajectories).read_bytes()).hexdigest(),
                  episodes=len(eps),excluded_api_errors=len(eps)-len(valid),splits=split)
    mp=out/'manifest.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest
    else: mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
    if a.limit: valid=valid[:a.limit]
    shards=out/'shards';shards.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(collect,e,str(shards/(Path(e['task_id']).stem+'.json'))):e['task_id'] for e in valid}
        for i,f in enumerate(as_completed(futures),1):
            n=f.result()
            if i%10==0 or i==len(valid): print(f'episodes={i}/{len(valid)} last_steps={n}',flush=True)
    rows=[]
    for e in valid:
        rows.extend(json.loads((shards/(Path(e['task_id']).stem+'.json')).read_text())['records'])
    for s in ('train','valid','test'):
        records=[r for r in rows if split[r['task_id']]==s]
        (out/(s+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps(dict(complete=not a.limit,episodes=len(valid),steps=len(rows))),flush=True)

if __name__=='__main__':main()
