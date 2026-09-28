"""Fixed-prefix simulator sensitivity audit, using training cases only."""
import concurrent.futures,json
from pathlib import Path
from scripts.run_tau2_candidates import Backend,ROOT
DATA=ROOT/'results/experiments/new_benchmarks';OUT=ROOT/'results/benchmarks/rescue/user_stop_audit';OUT.mkdir(parents=True,exist_ok=True)
GUIDANCE='''Termination clarification: agreeing to a requested action does not mean the action has been completed. Do not emit ###STOP### in the same turn that you authorize an operation which the agent has not yet executed. Wait for the agent to perform and confirm the authorized operation before ending. You may still end normally when your goal is fulfilled or you choose to abandon the request. Do not invent new requests or change your scenario.'''
cases={}
for domain,t in [('retail','44'),('retail','82'),('airline','11')]:
 d=json.loads((DATA/'tau2_aliyun'/f'{domain}_{t}_rank1.json').read_text());cases[d['simulation']['messages'][-1]['content']]=(domain,t)
requests=[]
for p in (DATA/'tau2_aliyun/cache').glob('*.json'):
 d=json.loads(p.read_text())
 if d.get('text') in cases:requests.append((cases[d['text']],d,p.name))
assert len(requests)==3

def work(item,variant):
 (domain,t),d,name=item;messages=[dict(m) for m in d['request']['messages']]
 model='qwen3.8-max' if variant=='stronger_user' else 'qwen3.8-flash'
 if variant=='termination_clarification':messages[0]['content']+='\n\n'+GUIDANCE
 backend=Backend(OUT/'cache',ROOT/'.env.aliyun',model);answer=backend.call(messages)
 out={'domain':domain,'task_id':t,'official_split':'train','variant':variant,'model':model,'source_request':name,'original_response':d['text'],'response':answer,'contains_stop':'###STOP###' in answer,'note':'Diagnostic fixed-prefix comparison, not a full task score.'}
 (OUT/f'{domain}_{t}_{variant}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 futures=[pool.submit(work,item,variant) for item in requests for variant in ['stronger_user','termination_clarification']]
 for f in concurrent.futures.as_completed(futures):f.result()
