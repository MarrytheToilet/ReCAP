"""Closed-loop evaluation of a portable learned selector or a fixed loop rule."""
import argparse,fcntl,hashlib,json,os
from scripts import run_miniwob_recap as bench
from scripts.browser_pair_features import select

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--artifact',default=str(bench.OUT/'trained_selector.json'));p.add_argument('--threshold',type=float);p.add_argument('--policy',choices=['learned','loop'],default='learned');p.add_argument('--worker',type=int,default=0);args=p.parse_args()
 if args.policy=='learned':
  from pathlib import Path
  path=Path(args.artifact);artifact=json.loads(path.read_text());assert artifact['feature_code_sha256']==hashlib.sha256(Path('scripts/browser_pair_features.py').read_bytes()).hexdigest()
  if args.threshold is not None:artifact['threshold']=args.threshold
 else:artifact={'name':'first-loop-rank2','threshold':None}
 digest=hashlib.sha256(json.dumps(artifact,sort_keys=True).encode()).hexdigest();stage=f"{args.policy}_{args.split}"+(f"_{args.threshold}" if args.threshold is not None else '')
 folder=bench.OUT/stage;folder.mkdir(exist_ok=True);original=bench.api
 def intercept(model,messages):
  if model!='qwen3.8-max':return original(model,messages)
  context=json.loads(messages[-1]['content']);c=context.pop('candidates')
  if args.policy=='loop':
   previous=context['previous_actions'];chosen=next((i for i,a in enumerate(c[1:],1) if a not in previous),0) if previous.count(c[0])>=2 else 0
  else:chosen=select(artifact,context,c)
  return json.dumps({'selected':chosen,'reason':args.policy+' frozen selector'})
 bench.api=intercept
 seeds=list(range(100,110)) if args.split=='validation' else list(range(2000,2010));jobs=[(t,s) for t in bench.TASKS for s in seeds];jobs.sort(key=lambda x:hashlib.sha256((str(args.worker)+str(x)).encode()).hexdigest())
 for task,seed in jobs:
  path=folder/f'{task}_{seed}.json'
  with path.with_suffix('.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if path.exists():
    d=json.loads(path.read_text());assert d['selector_sha256']==digest
    if d['status']=='completed':continue
   try:result=bench.run(task,seed,'semantic_once')
   except Exception as e:
    import traceback;traceback.print_exc();result={'status':'error','error':repr(e)}
   result.update(task=task,seed=seed,split=args.split,policy=args.policy,selector_sha256=digest,threshold=artifact['threshold']);tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(result,indent=2,default=str));tmp.replace(path);print(json.dumps({k:v for k,v in result.items() if k not in ['records','goal','terminal_info']}),flush=True)
if __name__=='__main__':main()
