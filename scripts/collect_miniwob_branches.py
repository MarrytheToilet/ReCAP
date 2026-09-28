"""Seed-reset browser counterfactuals over every alternative at hashed train prefixes."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
from scripts import run_miniwob_recap as bench
ROOT=bench.OUT;OUT=ROOT/'branches';OUT.mkdir(exist_ok=True)
def branch(task,seed,source,decision,choice):
 original=bench.api;seen=[];forced=False
 def forced_api(model,messages):
  nonlocal forced
  assert model=='qwen3.8-flash';state=json.loads(messages[-1]['content']);i=len(seen)
  if i<=decision:
   assert state==source['records'][i]['state'],f'Observed prefix mismatch at step {i}'
  seen.append(state)
  if i==decision:
   forced=True;pool=source['records'][i]['candidates'];return json.dumps({'candidates':[pool[choice]]})
  return original(model,messages)
 bench.api=forced_api
 try:result=bench.run(task,seed,'raw_rank1')
 finally:bench.api=original
 assert forced,'Prefix ended before forced action'
 result.update(decision=decision,choice=choice,prefix_match_verified=True,raw_reward=source['reward'],raw_suffix_steps=len(source['records'])-decision,alternative_suffix_steps=len(result['records'])-decision)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);args=p.parse_args()
 jobs=[(t,s) for t in bench.TASKS for s in range(20)];jobs.sort(key=lambda x:hashlib.sha256((str(args.worker)+str(x)).encode()).hexdigest())
 while True:
  pending=0;progress=False
  for task,seed in jobs:
   done=OUT/f'{task}_{seed}_summary.json'
   if done.exists() and json.loads(done.read_text()).get('complete'):continue
   pending+=1;src=ROOT/'train'/f'{task}_{seed}_raw_rank1.json'
   if not src.exists():continue
   source=json.loads(src.read_text())
   if source['status']!='completed':continue
   with done.with_suffix('.lock').open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:continue
    if done.exists() and json.loads(done.read_text()).get('complete'):continue
    points=[i for i,r in enumerate(source['records']) if len(r.get('candidates',[]))>1]
    ordered=sorted(points,key=lambda i:hashlib.sha256(f'miniwob-recap-v1:{task}:{seed}:{i}'.encode()).hexdigest());chosen=sorted(ordered[:(1 if source['reward'] else 3)])
    plan={'task':task,'seed':seed,'fit_split':'fit','raw_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'sampling':'Uniform fixed-hash prefix sampling: 1 per success, up to3 per failure; all alternatives.','points':chosen,'eligible':len(points)}
    (OUT/f'{task}_{seed}_plan.json').write_text(json.dumps(plan,indent=2));outputs=[]
    for i in chosen:
     for j in range(1,len(source['records'][i]['candidates'])):
      path=OUT/f'{task}_{seed}_d{i}_c{j}.json'
      if path.exists() and json.loads(path.read_text()).get('status') in ['completed','prefix_mismatch']:outputs.append(path.name);continue
      print(json.dumps({'event':'branch_start','task':task,'seed':seed,'decision':i,'choice':j}),flush=True)
      try:r=branch(task,seed,source,i,j)
      except AssertionError as e:r={'status':'prefix_mismatch','error':str(e)}
      except Exception as e:
       import traceback;traceback.print_exc();r={'status':'error','error':repr(e)}
      r.update(task=task,seed=seed,decision=i,choice=j,fit_split=plan['fit_split'],raw_source=str(src),raw_sha256=plan['raw_sha256'],sampling_weight=len(points)/len(chosen))
      tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(r,indent=2,default=str));tmp.replace(path);outputs.append(path.name);progress=True
      print(json.dumps({k:r.get(k) for k in ['task','seed','decision','choice','status','raw_reward','reward','error']}),flush=True)
    complete=all(json.loads((OUT/x).read_text())['status'] in ['completed','prefix_mismatch'] for x in outputs)
    done.write_text(json.dumps({**plan,'complete':complete,'outputs':outputs},indent=2));progress=True
  if not pending:break
  if not progress:time.sleep(10)
if __name__=='__main__':main()
