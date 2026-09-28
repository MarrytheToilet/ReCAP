"""Closed-loop fixed-rank controls on the exact local-LM admissible pool.

Only observations, admissible actions and executed history reach the policy.
No policy_commands, facts, progress rewards, or success labels enter selection.
"""
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from recap.agents.lm_policy_agent import rank_lm_policy_pool

def make_env(task):
    import textworld
    return textworld.start(task,request_infos=textworld.EnvInfos(admissible_commands=True,won=True,lost=True))

def episode(task,rank,horizon):
    env=make_env(task)
    try:
        state=env.reset();initial=state.feedback;history=[];done=False
        while not done and len(history)<horizon:
            pool=rank_lm_policy_pool(tuple(state.admissible_commands),tuple(history),objective=initial,mode='progress')[:20]
            if not pool:break
            action=pool[min(rank-1,len(pool)-1)];history.append(action);state,reward,done=env.step(action)
        return dict(task_id=task,success=bool(state.won),num_steps=len(history),actions=history)
    finally:env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=8);p.add_argument('--out-dir',default='results/experiments/pool_baselines');a=p.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);summary={}
    for difficulty in ('hard','xhard'):
        games=Path(f'analysis/localpolicy_eval/ctrl_{difficulty}100_games.txt').read_text().split()
        for rank in (1,2):
            name=f'{difficulty}_pool_rank{rank}';dest=out/(name+'.jsonl')
            existing=[json.loads(l) for l in dest.open()] if dest.exists() else [];seen={e['task_id'] for e in existing};rows=list(existing)
            with ProcessPoolExecutor(max_workers=a.workers) as pool,dest.open('a') as handle:
                futures=[pool.submit(episode,g,rank,30) for g in games if g not in seen]
                for f in as_completed(futures):
                    r=f.result();rows.append(r);handle.write(json.dumps(r)+'\n');handle.flush()
            assert len(rows)==len(games)
            summary[name]=dict(n=len(rows),success=sum(r['success'] for r in rows)/len(rows),steps=sum(r['num_steps'] for r in rows)/len(rows))
            print(name,summary[name],flush=True);(out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
