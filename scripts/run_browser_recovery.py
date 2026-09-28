"""Bounded multi-step recovery; public error/affordance triggers, no oracle input.
A distinct wider-support controller, compared with the same-trigger loop heuristic.
"""
import argparse,fcntl,hashlib,json,os,time,urllib.request,urllib.error
from pathlib import Path
from dotenv import dotenv_values
from scripts import run_miniwob_recap as bench
from scripts.browser_pair_features import action_features
PROMPT='''The browser agent has proposed an action that repeats an ineffective operation, repeats a just-failed operation, or conflicts with an observed readonly/disabled field. Resolve this specific problem while making progress toward the exact goal. Read the CURRENT page state carefully. Prefer an adequate action from the supplied candidates; otherwise propose one legal action using current visible element IDs. A repeated action is sometimes necessary when the visible state has changed, so inspect state and previous errors. Do not fill readonly fields. Do not click a parent container when the intended control has its own ID. Return ONLY JSON {"selected": integer-or-null, "new_action": string-or-null, "reason": string}. For a logged action give its index and null new_action; for a new action give null selected. Each action is exactly one supported function call with literal arguments, with string element IDs. No scripts, hidden state, or task modification.'''

def reasoned(messages):
 body={'model':'qwen3.8-max','messages':messages,'temperature':0,'enable_thinking':True,'thinking_budget':1024,'max_tokens':2048};root=bench.OUT/'recovery_reasoned_cache';root.mkdir(exist_ok=True);key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();p=root/(key+'.json');config=dotenv_values(bench.ROOT/'.env.aliyun')
 with p.with_suffix('.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if p.exists():return json.loads(p.read_text())['text']
  for attempt in range(4):
   try:
    req=urllib.request.Request(config['OPENAI_BASE_URL'].rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+config['OPENAI_API_KEY'],'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as r:d=json.load(r)
    result={'request':body,'text':d['choices'][0]['message']['content'],'usage':d.get('usage')};tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(result));tmp.replace(p);return result['text']
   except (urllib.error.URLError,TimeoutError):
    if attempt==3:raise
    time.sleep(2**attempt)

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--policy',choices=['reasoned','heuristic'],default='reasoned');p.add_argument('--worker',type=int,default=0);args=p.parse_args();folder=bench.OUT/f'recovery_{args.policy}_{args.split}';folder.mkdir(exist_ok=True)
 protocol={'prompt':PROMPT,'policy':args.policy,'trigger':'raw repeat_count>=2 AND unchanged visible DOM since previous decision; or repeated last-action error; or raw fill-readonly / disabled target.','budget':4,'support':'Prefer original candidates; reasoned policy may synthesize one legal action. Heuristic selects first admissible nonrepeated logged alternative. Each triggered attempt consumes budget, including abstentions.','generator':'qwen3.8-flash nonthinking','recovery':'qwen3.8-max thinking_budget1024','horizon':12,'note':'Distinct wider-support recovery controller, not a claim for the frozen-support learned classifier. Fixed based on training/validation diagnostics before final-test outcome inspection.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 path=bench.OUT/f'recovery_{args.policy}_protocol.json'
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if path.exists():assert json.loads(path.read_text())==protocol
  else:path.write_text(json.dumps(protocol,indent=2))
 original=bench.api;seeds=range(100,110) if args.split=='validation' else range(2000,2010);jobs=[(t,s) for t in bench.TASKS for s in seeds];jobs.sort(key=lambda x:hashlib.sha256((str(args.worker)+str(x)).encode()).hexdigest())
 for task,seed in jobs:
  dest=folder/f'{task}_{seed}.json'
  with dest.with_suffix('.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if dest.exists() and json.loads(dest.read_text()).get('status')=='completed':continue
   audit=[];previous_dom=None
   def route(model,messages):
    nonlocal previous_dom
    response=original(model,messages);state=json.loads(messages[-1]['content']);dom=state['visible_dom_elements'];unchanged=dom==previous_dom;previous_dom=dom
    if len(audit)>=4:return response
    try:pool=bench.parse(response)['candidates'];raw=pool[0];f=action_features(state,raw)
    except Exception:return response
    trigger=(f.get('repeat',0)>=.4 and unchanged) or f.get('repeat_after_error',0)>0 or f.get('fill_readonly',0)>0 or f.get('disabled',0)>0
    if not trigger:return response
    if args.policy=='heuristic':
     options=[]
     for j,action in enumerate(pool[1:],1):
      af=action_features(state,action)
      if bench.legal(action) and not any(af.get(k,0) for k in ['invalid','fill_readonly','disabled','repeat']):options.append(j)
     j=options[0] if options else 0;action=pool[j];origin='logged';explanation='First admissible nonrepeated alternative; otherwise retain raw'
    else:
     request={**state,'candidates':pool,'supported_actions':'click(bid), fill(bid,text), press(bid,key_combination), select_option(bid,option), hover(bid), scroll(delta_x,delta_y)'}
     d=bench.parse(reasoned([{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(request)}]));j=d.get('selected');explanation=d.get('reason')
     if type(j) is int and 0<=j<len(pool):action=pool[j];origin='logged'
     elif j is None and isinstance(d.get('new_action'),str):action=d['new_action'];origin='generated'
     else:raise ValueError('Invalid recovery response')
     assert bench.legal(action)
    audit.append({'decision':len(state['previous_actions']),'raw_candidates':pool,'chosen_action':action,'origin':origin,'reason':explanation,'features':f,'unchanged_dom':unchanged})
    return json.dumps({'candidates':[action]})
   bench.api=route
   try:result=bench.run(task,seed,'raw_rank1')
   except Exception as e:
    import traceback;traceback.print_exc();result={'status':'error','error':repr(e)}
   finally:bench.api=original
   result.update(task=task,seed=seed,split=args.split,policy=args.policy,recovery=audit,protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest());tmp=dest.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(result,indent=2,default=str));tmp.replace(dest);print(json.dumps({k:v for k,v in result.items() if k not in ['records','goal','terminal_info','recovery']}),flush=True)
if __name__=='__main__':main()
