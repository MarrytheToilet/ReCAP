"""Restore hash-selected training prefixes, force raw action, compare full outcomes."""
import hashlib,importlib,json
from pathlib import Path
from scripts.run_tau2_branches import ROOT,Backend,run_branch,user_module,judge_module
root=ROOT/'results/experiments/new_benchmarks';source=root/'tau2_aliyun';out=root/'tau2_raw_replay_audit';out.mkdir(exist_ok=True)
plans=sorted((root/'tau2_training_branches_uniform').glob('*_plan.json'),key=lambda p:hashlib.sha256(('raw-audit-v1:'+p.name).encode()).hexdigest())[:12]
manifest=out/'manifest.json'
if manifest.exists():names=json.loads(manifest.read_text())
else:names=[p.name for p in plans];manifest.write_text(json.dumps(names,indent=2))
backend=Backend(source/'cache',ROOT/'.env.aliyun','qwen3.8-flash');judge=Backend(source/'judge_cache',ROOT/'.env.aliyun','qwen3.8-max');user_module.generate=backend.user;judge_module.generate=judge.judge
bydomain={d:{t.id:t for t in importlib.import_module(f'tau2.domains.{d}.environment').get_tasks(task_split_name='train')} for d in ['retail','airline']}
def canonical(messages):return [{k:m.get(k) for k in ['role','content','tool_calls','tool_call_id','name']} for m in messages]
rows=[]
for name in names:
 plan=json.loads((root/'tau2_training_branches_uniform'/name).read_text());domain=plan['domain'];task=bydomain[domain][plan['task_id']];decision=plan['points'][0]['decision'];path=out/f'{domain}_{task.id}_d{decision}.json'
 if path.exists():rows.append(json.loads(path.read_text()));continue
 original=json.loads((source/f'{domain}_{task.id}_rank1.json').read_text())
 try:
  result=run_branch(task,original,decision,0,backend,judge,domain)
  result['raw_outcome_reproduced']=result.get('reward',{}).get('reward')==original['reward']['reward']
  result['raw_trajectory_reproduced']=canonical(result['simulation']['messages'])==canonical(original['simulation']['messages'])
  result['raw_termination_reproduced']=result['simulation']['termination_reason']==original['simulation']['termination_reason']
 except Exception as e:
  import traceback;traceback.print_exc();result={'domain':domain,'task_id':task.id,'decision':decision,'status':'error','error':repr(e)}
 path.write_text(json.dumps(result,indent=2));rows.append(result)
 summary={'planned':len(names),'finished':len(rows),'scored':sum(r.get('status')=='completed' for r in rows),'outcome_matches':sum(r.get('raw_outcome_reproduced',False) for r in rows),'trajectory_matches':sum(r.get('raw_trajectory_reproduced',False) for r in rows),'termination_matches':sum(r.get('raw_termination_reproduced',False) for r in rows)}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
