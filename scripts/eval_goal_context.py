"""Matched goal-context evaluation, sharded across authorized GPUs0/1."""
import json,os,subprocess,sys,time
from pathlib import Path
from scripts import eval_local_policy as evaluator
from recap.agents import lm_policy_agent as agent_module
from recap.models.goal_context import encode_goal_batch
agent_module.encode_candidate_batch=encode_goal_batch
class GoalAgent(agent_module.LocalLMPolicyAgent):
 def __init__(self,*args,**kwargs):
  kwargs.update(max_length=1024,max_observation_chars=360)
  super().__init__(*args,**kwargs)
evaluator.LocalLMPolicyAgent=GoalAgent

def parallel(module='scripts.eval_goal_context'):
 args=sys.argv[1:];position=args.index('--out-dir')+1;out=Path(args[position]);out.mkdir(parents=True,exist_ok=True);processes=[];handles=[]
 for worker in range(2):
  folder=out/f'shard{worker}';folder.mkdir(exist_ok=True);log=(folder/'driver.log').open('a');handles.append(log);worker_args=list(args);worker_args[position]=str(folder)
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(worker)}
  processes.append(subprocess.Popen([sys.executable,'-m',module,*worker_args,'--shard-index',str(worker),'--num-shards','2'],env=env,stdout=log,stderr=subprocess.STDOUT))
 def merge():
  summary={}
  for difficulty in ['hard','xhard']:
   rows=[]
   for worker in range(2):
    p=out/f'shard{worker}'/(difficulty+'.jsonl')
    if p.exists():
     text=p.read_text();lines=text.splitlines() if text.endswith('\n') else text[:text.rfind('\n')+1].splitlines()
     rows.extend(json.loads(l) for l in lines if l)
   assert len({r['task_id'] for r in rows})==len(rows)
   if rows:
    target=out/(difficulty+'.jsonl');tmp=target.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in sorted(rows,key=lambda r:r['task_id'])));tmp.replace(target)
    summary[difficulty]={'n':len(rows),'success':sum(r['success'] for r in rows)/len(rows),'steps':sum(r['num_steps'] for r in rows)/len(rows),'timeouts':sum(bool(r.get('error')) for r in rows),'episode_timeout_seconds':600,'num_shards':2}
  tmp=out/'summary.tmp';tmp.write_text(json.dumps(summary,indent=2));tmp.replace(out/'summary.json')
 while any(p.poll() is None for p in processes):merge();time.sleep(30)
 merge()
 for f in handles:f.close()
 if any(p.returncode!=0 for p in processes):raise RuntimeError('A goal-context evaluation shard failed; see shard driver logs')
if __name__=='__main__':
 if '--num-shards' in sys.argv:evaluator.main()
 else:parallel()
