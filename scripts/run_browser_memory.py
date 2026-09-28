"""Replay-preference retrieval for full-context semantic candidate selection.
Uses only fixed training comparisons. This is in-context conditioning, not an
additional weight fine-tune. Shuffled labels use the SAME retrieved source examples as the original-label arm.
"""
import argparse,ast,collections,fcntl,hashlib,json,math,os,re
from pathlib import Path
from scripts import run_miniwob_recap as bench
EXPLANATION='''The following are retrieved comparisons from separate TRAINING browser instances, obtained by resetting the environment to the same prefix, forcing each logged action, then continuing with the original agent. A better outcome or shorter successful suffix certifies only this PAIR under that continuation. Do not infer dominance over other actions. Apply these examples only if their public circumstances match the current state. Example element IDs belong to other pages: never copy them to the current page. Use the current page IDs. Prefer preserving candidate0 unless there is a concrete supported improvement. Return the same JSON selected-index/reason schema as requested.'''

def signature(state):
 text=state['goal']+' '+ ' '.join(str(e.get('html_id',''))+' '+str(e.get('tag',''))+' '+str(e.get('classes','')) for e in state.get('visible_dom_elements',[]))
 return set(re.findall(r'[a-z_]{2,}',text.lower()))
def compact(state,actions):
 bids=[]
 for action in actions:
  try:
   node=ast.parse(action,mode='eval').body;bids.append(str(ast.literal_eval(node.args[0])))
  except Exception:pass
 return {'goal':state['goal'],'recent_actions':state['previous_actions'][-4:],'last_action_error':state.get('last_action_error','')[:500],'action_targets':[e for e in state.get('visible_dom_elements',[]) if str(e['bid']) in bids],'accessibility_tree':state['accessibility_tree'][:1800]}
def build_memory():
 out=[]
 for p in sorted((bench.OUT/'branches').glob('*_d*_c*.json')):
  d=json.loads(p.read_text())
  if d['status']!='completed':continue
  raw=json.loads(Path(d['raw_source']).read_text());r=raw['records'][d['decision']];a=r['candidates'][0];b=r['candidates'][d['choice']];delta=d['reward']-d['raw_reward']
  label=int(delta)
  if not delta and d['reward']:label=int(d['alternative_suffix_steps']<d['raw_suffix_steps'])-int(d['alternative_suffix_steps']>d['raw_suffix_steps'])
  if not label:continue
  assert 0<=d['seed']<20
  out.append({'source':p.name,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tokens':sorted(signature(r['state'])),'context':compact(r['state'],[a,b]),'candidate0':a,'alternative':b,'label':label,'raw_success':d['raw_reward'],'alternative_success':d['reward'],'raw_suffix_steps':d['raw_suffix_steps'],'alternative_suffix_steps':d['alternative_suffix_steps']})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--variant',choices=['memory','shuffled'],default='memory');p.add_argument('--worker',type=int,default=0);args=p.parse_args();folder=bench.OUT/f'{args.variant}_{args.split}';folder.mkdir(exist_ok=True)
 memory=build_memory();assert memory
 import random
 labels=[r['label'] for r in memory];random.Random(0).shuffle(labels)
 if args.variant=='shuffled':
  # Shuffle the target label only; do not expose unshuffled outcome/cost fields.
  for r,label in zip(memory,labels):r['selection_label']=r['label'];r['label']=label
 protocol={'explanation':EXPLANATION,'retrieval':'Public goal+DOM tag/id/class token Jaccard. Highest-similarity three strict pairs, prefer one positive and one negative example when available. No validation or test examples.','generator':'qwen3.8-flash','critic':'qwen3.8-max','enable_thinking':False,'source_hashes':{r['source']:r['source_sha256'] for r in memory},'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'variant':args.variant,'note':'In-context replay-preference conditioning, not parameter fine-tuning. Threshold-free fixed prompt; one intervention budget. Fixed before final test outcomes.'}
 path=bench.OUT/f'{args.variant}_protocol.json'
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if path.exists():assert json.loads(path.read_text())==protocol
  else:path.write_text(json.dumps(protocol,indent=2))
 original=bench.api;retrievals=[]
 def retrieve(model,messages):
  if model!='qwen3.8-max':return original(model,messages)
  current=json.loads(messages[-1]['content']);query=signature(current);ordered=sorted(memory,key=lambda r:(-len(query&set(r['tokens']))/max(len(query|set(r['tokens'])),1),r['source']))
  picked=[]
  for label in [1,-1]:
   match=next((r for r in ordered if r.get('selection_label',r['label'])==label),None)
   if match:picked.append(match)
  for r in ordered:
   if len(picked)>=3:break
   if r not in picked:picked.append(r)
  examples=[{'context':r['context'],'candidate0':r['candidate0'],'alternative':r['alternative'],'verified_pair_preference':'alternative preferred' if r['label']>0 else 'candidate0 preferred'} for r in picked]
  retrievals.append([r['source'] for r in picked]);request=[dict(m) for m in messages];request[0]['content']+='\n\n'+EXPLANATION+'\nTRAINING COMPARISONS:\n'+json.dumps(examples)
  return original(model,request)
 bench.api=retrieve;seeds=range(100,110) if args.split=='validation' else range(2000,2010);jobs=[(t,s) for t in bench.TASKS for s in seeds];jobs.sort(key=lambda x:hashlib.sha256((str(args.worker)+str(x)).encode()).hexdigest())
 for task,seed in jobs:
  dest=folder/f'{task}_{seed}.json'
  with dest.with_suffix('.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if dest.exists() and json.loads(dest.read_text()).get('status')=='completed':continue
   retrievals=[]
   try:result=bench.run(task,seed,'semantic_once')
   except Exception as e:
    import traceback;traceback.print_exc();result={'status':'error','error':repr(e)}
   result.update(task=task,seed=seed,split=args.split,variant=args.variant,retrievals=retrievals,protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest());tmp=dest.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(result,indent=2,default=str));tmp.replace(dest);print(json.dumps({k:v for k,v in result.items() if k not in ['records','goal','terminal_info','retrievals']}),flush=True)
if __name__=='__main__':main()
