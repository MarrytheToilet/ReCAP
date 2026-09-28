"""Join local-policy runs by exact game ID and retain incomplete denominators."""
import datetime,json,math
from pathlib import Path
import numpy as np
ROOT=Path('results/experiments')
ARMS=['softmax','pairwise','shaped_complete','cost_complete','goal_context','chat_goal_context','base_rank1','base_rank2','base_chat_goal','base_goal_context']
def read(path):
 if not path.exists():return {}
 rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];d={r['task_id']:r for r in rows}
 assert len(d)==len(rows),'Duplicate evaluation task';return d

def compare(a,b,tasks):
 common=sorted(set(a)&set(b));delta=np.array([int(b[t]['success'])-int(a[t]['success']) for t in common]);rescues=int(sum(delta>0));harms=int(sum(delta<0));n=rescues+harms
 result={'planned':len(tasks),'paired':len(common),'final':len(common)==len(tasks),'rescues':rescues,'harms':harms,'net':rescues-harms}
 if len(delta):
  rng=np.random.default_rng(20260928);result.update(delta=float(delta.mean()),ci95=np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975]).tolist(),mcnemar_exact_p=min(1.,2*sum(math.comb(n,i) for i in range(min(rescues,harms)+1))/2**n) if n else 1.)
 return result

def main():
 out={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'note':'Same100 games per difficulty, pool20,30steps. Each new adapter has one training seed. Task-bootstrap intervals do not capture training-seed variation. Goal-context changes input budget as registered.','suites':{}}
 for diff in ['hard','xhard']:
  tasks=Path(f'analysis/localpolicy_eval/ctrl_{diff}100_games.txt').read_text().split();loaded={};metrics={}
  for arm in ARMS:
   folder=ROOT/('qwen3_4b_'+arm);path=(folder if arm.startswith('base_') else folder/'eval')/(diff+'.jsonl');d=read(path);assert set(d)<=set(tasks);loaded[arm]=d
   metrics[arm]={'planned':len(tasks),'scored':len(d),'success':sum(bool(r['success']) for r in d.values()),'errors':sum(bool(r.get('error')) for r in d.values()),'final':len(d)==len(tasks)}
  for name,path in [('pool_rank1',ROOT/'pool_baselines'/f'{diff}_pool_rank1.jsonl'),('public_plan_pool20',ROOT/'public_plan'/f'{diff}_pool20.jsonl')]:loaded[name]=read(path)
  pairs={}
  for a,b in [('base_goal_context','goal_context'),('base_goal_context','base_chat_goal'),('base_rank1','base_goal_context'),('base_rank1','base_chat_goal'),('softmax','base_chat_goal'),('base_rank1','softmax'),('softmax','pairwise'),('shaped_complete','cost_complete'),('softmax','goal_context'),('goal_context','chat_goal_context'),('base_chat_goal','chat_goal_context'),('pool_rank1','softmax'),('public_plan_pool20','softmax'),('public_plan_pool20','goal_context'),('public_plan_pool20','chat_goal_context')]:pairs[b+'_vs_'+a]=compare(loaded[a],loaded[b],tasks)
  out['suites'][diff]={'arms':metrics,'comparisons':pairs}
 (ROOT/'gpu_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
