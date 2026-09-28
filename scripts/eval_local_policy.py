"""Verifier-free closed-loop evaluation of a local LM, including its rank-2 control."""
from __future__ import annotations
import argparse,json,signal
from pathlib import Path
from recap.agents.base import AgentContext
from recap.agents.lm_policy_agent import LocalLMPolicyAgent
from scripts.eval_pool_baselines import make_env

def timeout_handler(_signum, _frame):
    raise TimeoutError('episode wall-clock budget exceeded')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--adapter',type=Path)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--rank',type=int,default=1);p.add_argument('--limit',type=int)
    p.add_argument('--episode-timeout',type=int,default=600)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    a=p.parse_args();assert a.num_shards>0 and 0<=a.shard_index<a.num_shards
    a.out_dir.mkdir(parents=True,exist_ok=True)
    agent=LocalLMPolicyAgent(a.model,adapter=a.adapter,pool_ranker='progress',candidate_chunk_size=4)
    signal.signal(signal.SIGALRM,timeout_handler)
    summary={}
    for difficulty in ('hard','xhard'):
        games=Path(f'analysis/localpolicy_eval/ctrl_{difficulty}100_games.txt').read_text().split()
        if a.limit:games=games[:a.limit]
        games=[task for i,task in enumerate(games) if i%a.num_shards==a.shard_index]
        path=a.out_dir/(difficulty+'.jsonl');rows=[json.loads(l) for l in path.open()] if path.exists() else [];done_tasks={r['task_id'] for r in rows}
        with path.open('a') as f:
            for task in games:
                if task in done_tasks:continue
                env=make_env(task)
                history=[]
                try:
                    signal.alarm(a.episode_timeout)
                    state=env.reset();initial=state.feedback;history=[];done=False
                    while not done and len(history)<30:
                        context=AgentContext(task,0,len(history),state.feedback,tuple(state.admissible_commands),tuple(history),(),(),initial)
                        candidates=agent.candidates(context)
                        if not candidates:break
                        action=candidates[min(a.rank-1,len(candidates)-1)].action
                        state,_,done=env.step(action);history.append(action)
                    r=dict(task_id=task,success=bool(state.won),num_steps=len(history),actions=history,error=None)
                except TimeoutError as exc:
                    r=dict(task_id=task,success=False,num_steps=len(history),actions=history,error=str(exc))
                finally:
                    signal.alarm(10)
                    try:env.close()
                    except TimeoutError:pass
                    finally:signal.alarm(0)
                rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
                print(difficulty,len(rows),r['success'],r['num_steps'],r['error'],flush=True)
        summary[difficulty]=dict(n=len(rows),success=sum(r['success'] for r in rows)/len(rows),steps=sum(r['num_steps'] for r in rows)/len(rows),timeouts=sum(bool(r.get('error')) for r in rows),episode_timeout_seconds=a.episode_timeout)
        (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
