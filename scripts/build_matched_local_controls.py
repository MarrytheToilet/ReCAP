"""Freeze the deployed first-1800 contexts and candidate subsets, then audit costs."""
from __future__ import annotations
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from recap.models.train_lm_candidate_policy import limited_record_candidates
from recap.eval.eval_repair_multiplicity import replay_candidate_with_policy_suffix
from recap.envs.textworld_adapter import TextWorldAdapter

def annotate(job):
    i,r,path=job;p=Path(path)
    if p.exists():return i,json.loads(p.read_text())
    adapter=TextWorldAdapter()
    try:
        costs={}
        for c in r['candidates']:
            q=replay_candidate_with_policy_suffix(adapter,r['task_id'],int(r.get('seed',0)),tuple(r['history']),c)
            costs[c]=q['suffix_len'] if q['success'] else None
        new=dict(r);new['candidate_costs']=costs
        p.write_text(json.dumps(new));return i,new
    finally:adapter.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=8);p.add_argument('--annotate',action='store_true');a=p.parse_args()
    src=Path('analysis/recap_xhard_700_online_policy_progress_pool60.jsonl');out=Path('results/experiments/local_controls');out.mkdir(parents=True,exist_ok=True)
    original=[json.loads(l) for l in src.open()][:1800]
    rows=[limited_record_candidates(r,16,True) for r in original]
    train_tasks={r['task_id'] for r in rows};overlap={}
    for difficulty in ('hard','xhard'):
        games=set(Path(f'analysis/localpolicy_eval/ctrl_{difficulty}100_games.txt').read_text().split())
        overlap[difficulty]=sorted(train_tasks & games);assert not overlap[difficulty]
    (out/'shaped_fixed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'manifest.json').write_text(json.dumps(dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        rows=len(rows),tasks=len(train_tasks),test_task_overlap=overlap,candidate_subsampling='original reward-best-preserving max16, frozen before arm assignment',
        supervision='policy_commands-derived shaped rewards; not a learned reward-model output',
        historical_effective_loss='reward_guided_scpo_loss: CE + expected standardized reward + rank-prior KL; pairwise and entropy flags inactive'),indent=2))
    if not a.annotate:return
    shard=out/'cost_shards';shard.mkdir(exist_ok=True);ordered=[None]*len(rows)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(annotate,(i,r,str(shard/f'{i:04d}.json'))) for i,r in enumerate(rows)]
        for n,f in enumerate(as_completed(futures),1):
            i,r=f.result();ordered[i]=r
            if n%50==0:print('annotated',n,flush=True)
    complete=[r for r in ordered if all(v is not None for v in r['candidate_costs'].values())]
    # Do not fill unknown costs with invented penalties. Use the identical complete subset for both controls.
    shaped=[];cost=[]
    for r in complete:
        shaped.append(r);q=dict(r);q['candidate_rewards']={c:-float(v) for c,v in r['candidate_costs'].items()};cost.append(q)
    for name,rs in [('shaped_complete',shaped),('cost_complete',cost)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs))
    (out/'cost_summary.json').write_text(json.dumps(dict(total=len(rows),complete=len(complete),censored=len(rows)-len(complete)),indent=2))
if __name__=='__main__':main()
