"""Paired primary results with explicit planned denominators and protocol boundaries."""
import collections,datetime,json,math
from pathlib import Path
import numpy as np
ROOT=Path('results/benchmarks');WEB=ROOT/'miniwob';TAU=Path('results/experiments/new_benchmarks')

def paired(rows,strata):
 if not rows:return None
 delta=np.array([r['new']-r['raw'] for r in rows]);rescues=int((delta>0).sum());harms=int((delta<0).sum());n=rescues+harms
 p=min(1.,2*sum(math.comb(n,i) for i in range(min(rescues,harms)+1))/2**n) if n else 1.
 rng=np.random.default_rng(42);samples=np.zeros(3000)
 for group in sorted(set(strata)):
  values=delta[np.array(strata)==group];samples+=rng.choice(values,(3000,len(values)),replace=True).sum(axis=1)
 samples/=len(delta)
 return {'paired':len(rows),'rescues':rescues,'harms':harms,'delta':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'mcnemar_exact_p':p,'note':'Paired bootstrap within fixed task-type/domain strata; no training-seed or independent API-trial uncertainty.'}

def web_results():
 protocol=json.loads((WEB/'protocol.json').read_text());expected=[(t,s) for t in protocol['tasks'] for s in protocol['test_seeds']];arms={};raw={}
 for task,seed in expected:
  p=WEB/'test'/f'{task}_{seed}_raw_rank1.json'
  if p.exists():
   d=json.loads(p.read_text())
   if d['status']=='completed':raw[(task,seed)]=d
 names=['raw_rank1','first_rank2_once','semantic_once','learned','loop','recovery_reasoned','recovery_heuristic']
 for arm in names:
  records={};missing=[];errors=[]
  for task,seed in expected:
   p=WEB/'test'/f'{task}_{seed}_{arm}.json' if arm in names[:3] else WEB/f'{arm}_test'/f'{task}_{seed}.json'
   if not p.exists():missing.append((task,seed));continue
   d=json.loads(p.read_text());assert d['task']==task and d['seed']==seed
   if d['status']!='completed':errors.append({'task':task,'seed':seed,'error':d.get('error')});continue
   records[(task,seed)]=d
  rows=[{'raw':raw[k]['reward'],'new':d['reward']} for k,d in records.items() if k in raw];strata=[k[0] for k in records if k in raw]
  per_type={t:{'planned':10,'scored':sum(k[0]==t for k in records),'success':sum(d['reward'] for k,d in records.items() if k[0]==t)} for t in protocol['tasks']}
  arms[arm]={'planned':60,'scored':len(records),'success':sum(d['reward'] for d in records.values()),'complete':len(records)==60,'missing':missing,'errors':errors,'paired_vs_raw':paired(rows,strata),'per_type':per_type,'capped_action_cost_sum':sum(len(d['records']) if d['reward'] else 12 for d in records.values()),'note':'Action cost excludes extra critic calls. Learned frozen artifact may abstain; report interventions separately.'}
  if arm.startswith('recovery'):
   arms[arm]['recovery_calls']=sum(len(d.get('recovery',[])) for d in records.values());arms[arm]['generated_actions']=sum(r['origin']=='generated' for d in records.values() for r in d.get('recovery',[]));arms[arm]['intervened_tasks']=sum(any(r['chosen_action']!=r['raw_candidates'][0] for r in d.get('recovery',[])) for d in records.values())
  else:arms[arm]['intervened_tasks']=sum(bool(d.get('intervened')) for d in records.values())
 return {'tasks':protocol['tasks'],'test_instances':60,'arms':arms,'validation_freeze':json.loads((WEB/'selector_freeze.json').read_text()) if (WEB/'selector_freeze.json').exists() else None}

def tau_results():
 ids=json.loads((TAU/'full_baseline_manifest.json').read_text())['splits'];out={}
 for arm,folder in [('original_learned',TAU/'tau2_learned_test'),('semantic_repair',ROOT/'rescue/semantic_policy_v1/test'),('guarded_raw',ROOT/'rescue/guard_test_raw')]:
  rows=[];missing=[]
  for domain,split in ids.items():
   for task in split['test']:
    p=folder/(f'{domain}_{task}.json' if arm=='semantic_repair' else f'{domain}_{task}_rank1.json')
    if not p.exists():missing.append((domain,task));continue
    d=json.loads(p.read_text());raw=json.loads((TAU/'tau2_aliyun'/f'{domain}_{task}_rank1.json').read_text())
    if d['status']!='completed':missing.append((domain,task));continue
    reward=d['reward'] if arm=='semantic_repair' else d['reward']['reward'];rows.append({'domain':domain,'task':task,'raw':raw['reward']['reward'],'new':reward})
  out[arm]={'planned':60,'scored':len(rows),'success':sum(r['new'] for r in rows),'complete':len(rows)==60,'missing':missing,'paired_vs_raw':paired(rows,[r['domain'] for r in rows]),'changed_tasks':[r for r in rows if r['raw']!=r['new']]}
 return out

def main():
 out={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tau':tau_results(),'web':web_results(),'interpretation':'Guarded tau changes simulator termination handling and parser; not a learned-selector gain or an official unchanged-protocol leaderboard score. Semantic tau is an exploratory repair on previously inspected tasks. Web final seeds2000–2009 were excluded from fitting/calibration. Recovery widens action support and adds reasoning calls, distinct from the learned frozen-support selector.'}
 (ROOT/'FINAL_RESULTS.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({'tau':{k:{i:v[i] for i in ['scored','success','complete']} for k,v in out['tau'].items()},'web':{k:{i:v[i] for i in ['scored','success','complete','intervened_tasks']} for k,v in out['web']['arms'].items()}},indent=2))
if __name__=='__main__':main()
