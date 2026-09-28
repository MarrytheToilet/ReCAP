"""Regrade discordant training branches on fixed transcripts with a second model."""
import argparse,datetime,importlib,json,time
from pathlib import Path
from scripts.run_tau2_candidates import ROOT,Backend,SimulationRun,judge_module,evaluate_simulation,EvaluationType,RewardType
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-hours', type=float, default=1)
args=parser.parse_args()
if args.max_hours<=0:parser.error('--max-hours must be positive')
root=ROOT/'results/experiments/new_benchmarks';source=root/'tau2_training_branches_uniform';out=root/'tau2_secondary_judge_audit';out.mkdir(exist_ok=True)
backend=Backend(root/'tau2_aliyun/judge_cache',ROOT/'.env.aliyun','deepseek-v4.1-flash');judge_module.generate=backend.judge
bydomain={d:{t.id:t for t in importlib.import_module(f'tau2.domains.{d}.environment').get_tasks(task_split_name='train')} for d in ['retail','airline']}
def uses_nl(task):
 return bool(task.evaluation_criteria and task.evaluation_criteria.nl_assertions and RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis)
end=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=args.max_hours);attempts={}
def regrade(name,task,data,domain):
 path=out/(name+'.json')
 if path.exists():
  saved=json.loads(path.read_text())
  if saved['status']=='completed':return saved
 reward=evaluate_simulation(SimulationRun.model_validate(data['simulation']),task,EvaluationType.ALL,False,domain)
 result={'status':'completed','judge_model':backend.model,'primary_reward':data['reward']['reward'],'secondary_reward':reward.reward,'reward':reward.model_dump(mode='json')}
 path.write_text(json.dumps(result,indent=2));return result
while datetime.datetime.now(datetime.timezone.utc)<end:
 changed=False
 for p in sorted(source.glob('*_d*_c*.json')):
  if datetime.datetime.now(datetime.timezone.utc)>=end:break
  d=json.loads(p.read_text())
  if d.get('status')!='completed' or d['reward']['reward']==d['raw_reward']:continue
  assert d['official_split']=='train';task=bydomain[d['domain']][d['task_id']]
  if not uses_nl(task):continue
  dest=out/(p.stem+'_comparison.json')
  if dest.exists() or attempts.get(p.name,0)>=2:continue
  attempts[p.name]=attempts.get(p.name,0)+1
  try:
   raw=json.loads((root/'tau2_aliyun'/f"{d['domain']}_{d['task_id']}_rank1.json").read_text())
   a=regrade(f"{d['domain']}_{d['task_id']}_raw",task,raw,d['domain']);b=regrade(p.stem,task,d,d['domain'])
   primary=int(d['reward']['reward']==1)-int(d['raw_reward']==1);secondary=int(b['secondary_reward']==1)-int(a['secondary_reward']==1)
   result={'source':p.name,'official_split':'train','primary_label':primary,'secondary_label':secondary,'label_agrees':primary==secondary,'raw_reward_agrees':a['primary_reward']==a['secondary_reward'],'alternative_reward_agrees':b['primary_reward']==b['secondary_reward']}
   dest.write_text(json.dumps(result,indent=2));changed=True;print(json.dumps(result),flush=True)
  except Exception as e:
   import traceback;traceback.print_exc()
   with (out/'errors.jsonl').open('a') as f:f.write(json.dumps({'source':p.name,'error':repr(e),'attempt':attempts[p.name]})+'\n')
 all_comparisons=[json.loads(p.read_text()) for p in out.glob('*_comparison.json')]
 comparisons=[]
 for item in all_comparisons:
  record=json.loads((source/item['source']).read_text())
  if uses_nl(bydomain[record['domain']][record['task_id']]):comparisons.append(item)
 summary={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'secondary_judge':backend.model,'training_only':True,'excluded_deterministic_rechecks':len(all_comparisons)-len(comparisons),'discordant_primary_pairs_audited':len(comparisons),'label_agreement':sum(d['label_agrees'] for d in comparisons),'disagreements':[d for d in comparisons if not d['label_agrees']],'note':'Fixed saved transcripts; same original task evaluator with only NL judge changed. Primary labels and model selection are unchanged. Only tasks whose reward_basis includes NL_ASSERTION and whose assertions are nonempty count as judge sensitivity checks. Other saved rechecks are deterministic and excluded. Agreement is not a proof of correctness.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 done=[p for p in source.glob('*_summary.json') if json.loads(p.read_text()).get('candidate_scope')=='all-logged-alternatives-v1']
 if len(done)>=104 and not changed:break
 time.sleep(20)
