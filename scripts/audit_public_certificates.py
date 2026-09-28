"""Independent environment re-execution audit of a fixed hash sample of certificates."""
import hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scripts.eval_pool_baselines import make_env
ROOT=Path('results/experiments/public_plan_progress_replay')
def audit(item):
 r,a,t=item;env=make_env(r['task_id'])
 try:
  s=env.reset()
  for act in r['history']+[a]+t['suffix']:s,_,_=env.step(act)
  assert bool(s.won)==t['success'],(r['task_id'],r['step_index'],a)
  assert t['cost'] is None or t['cost']==len(t['suffix'])
  return {'task_id':r['task_id'],'step_index':r['step_index'],'candidate':a,'success_reproduced':True,'finite_cost':t['cost']}
 finally:env.close()
def main():
 items=[]
 for split in ['train','valid','test']:
  for r in map(json.loads,open(ROOT/f'{split}.jsonl')):
   for a,t in r['trials'].items():items.append((r,a,t))
 items.sort(key=lambda item:hashlib.sha256(f"certificate-audit:{item[0]['task_id']}:{item[0]['step_index']}:{item[1]}".encode()).hexdigest())
 with ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(audit,items[:120]))
 result={'sample_size':len(results),'all_reproduced':True,'sample_selection':'First 120 by SHA256 of fixed audit prefix/task/step/candidate; includes censored and successful branches.','records':results}
 (ROOT/'independent_audit.json').write_text(json.dumps(result,indent=2));print('independent replays passed',len(results),flush=True)
if __name__=='__main__':main()
