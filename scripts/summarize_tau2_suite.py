"""Task-matched benchmark summary with planned/error/unfinished denominators."""
import datetime,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'results/experiments/new_benchmarks'
ARMS={'native_tools':('tau2_native_test',1),'raw_rank1':('tau2_aliyun',1),'always_rank2':('tau2_aliyun',2),'validity_filter':('tau2_validity_test',1),'first_rank2_once':('tau2_first_rank2_test',1),'learned_once':('tau2_learned_test',1)}
def record(folder,domain,task,rank):
 p=ROOT/folder/f'{domain}_{task}_rank{rank}.json'
 if not p.exists():return None
 try:return json.loads(p.read_text())
 except json.JSONDecodeError:return {'status':'incomplete_write'}
def summarize(records,planned):
 scored=[d for d in records if d and d.get('status')=='completed'];success=sum(d['reward']['reward']==1 for d in scored)
 work={}
 for key,extract in [('agent_turns',lambda m:int(m.get('role')=='assistant')),('tool_calls',lambda m:len(m.get('tool_calls') or []) if m.get('role')=='assistant' else 0),('user_turns',lambda m:int(m.get('role')=='user'))]:
  values=[sum(extract(m) for m in d['simulation']['messages']) for d in scored]
  work[key]={'total':sum(values),'mean':float(np.mean(values)) if values else None,'median':float(np.median(values)) if values else None}
 return {'planned':planned,'scored':len(scored),'success':success,'pending':sum(d is None for d in records),'error':sum(d is not None and d.get('status')!='completed' for d in records),'success_rate_on_scored':success/len(scored) if scored else None,'final':len(scored)==planned,'selector_hashes':sorted({d.get('selector_sha256') or 'none' for d in scored}),'trajectory_work_on_scored':work,'work_note':'Logical turns and tool calls in saved trajectories, including cache hits; not billed API requests, tokens or wallclock cost. Native tools may dispatch multiple calls per assistant turn.'}
def paired(records,planned):
 pairs=[(domain,int(a['reward']['reward']==1),int(b['reward']['reward']==1)) for domain,a,b in records if a and b and a.get('status')==b.get('status')=='completed']
 n=len(pairs);rescues=sum(b>a for _,a,b in pairs);harms=sum(a>b for _,a,b in pairs);out={'planned':planned,'paired':n,'final':n==planned,'rescues':rescues,'harms':harms,'net':rescues-harms,'success_difference':(rescues-harms)/n if n else None}
 if n:
  rng=np.random.default_rng(20260928);draws=np.zeros(10000)
  # Paired resampling by task, stratified by domain in a pooled result.
  for domain in sorted({d for d,_,_ in pairs}):
   delta=np.array([b-a for d,a,b in pairs if d==domain]);draws+=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].sum(1)
  out['paired_task_bootstrap_95ci']=np.quantile(draws/n,[.025,.975]).tolist()
  discordant=rescues+harms
  out['mcnemar_exact_two_sided']=min(1.,2*sum(math.comb(discordant,k) for k in range(min(rescues,harms)+1))/2**discordant) if discordant else 1.
 return out

def main():
 manifest=json.loads((ROOT/'full_baseline_manifest.json').read_text());splits=manifest['splits'];summary={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'benchmark_commit':manifest['commit'],'agent_model':manifest['agent_model'],'nl_judge_model':manifest['nl_judge_model'],'note':'Partial success rates and intervals describe completed tasks only; use final=true for full planned comparisons. One trial per task. Bootstrap reflects task variation, not repeated user/API stochasticity.','groups':{}}
 for split in ['train','test']:
  for group in [*splits,'pooled']:
   ids=[(d,t) for d in splits if group in [d,'pooled'] for t in splits[d][split]];arms={};loaded={}
   for name,(folder,rank) in ARMS.items():
    if split=='train' and name not in ['raw_rank1','always_rank2']:continue
    loaded[name]=[record(folder,d,t,rank) for d,t in ids];arms[name]=summarize(loaded[name],len(ids))
   comparisons={}
   for name in loaded:
    if name=='raw_rank1':continue
    comparisons[name+'_vs_raw']=paired([(d,a,b) for (d,t),a,b in zip(ids,loaded['raw_rank1'],loaded[name])],len(ids))
   if 'learned_once' in loaded:
    for name in ['validity_filter','first_rank2_once']:
     comparisons['learned_vs_'+name]=paired([(d,a,b) for (d,t),a,b in zip(ids,loaded[name],loaded['learned_once'])],len(ids))
   summary['groups'][group+'/'+split]={'arms':arms,'comparisons':comparisons}
 definitions={d:{str(t['id']):bool((t.get('evaluation_criteria') or {}).get('nl_assertions')) and 'NL_ASSERTION' in (t.get('evaluation_criteria') or {}).get('reward_basis',[]) for t in json.loads((ROOT.parents[2]/'third_party/tau2-bench/data/tau2/domains'/d/'tasks.json').read_text())} for d in splits}
 summary['test_by_evaluator_kind']={}
 for kind,wants_nl in [('deterministic_only',False),('includes_nl_judge',True)]:
  ids=[(d,t) for d in splits for t in splits[d]['test'] if definitions[d][str(t)]==wants_nl]
  summary['test_by_evaluator_kind'][kind]={name:summarize([record(folder,d,t,rank) for d,t in ids],len(ids)) for name,(folder,rank) in ARMS.items()}
 counts=Counter();tasks=set();sources=[];rescue_tasks=set();harm_tasks=set();failed_tasks_probed=set();by_kind={'deterministic_only':Counter(),'includes_nl_judge':Counter()}
 planned_sources=set()
 for plan_path in (ROOT/'tau2_training_branches_uniform').glob('*_plan.json'):
  plan=json.loads(plan_path.read_text())
  for point in plan['points']:
   for choice in point['choices']:planned_sources.add(f"{plan['domain']}_{plan['task_id']}_d{point['decision']}_c{choice}.json")
 for p in sorted((ROOT/'tau2_training_branches_uniform').glob('*_d*_c*.json')):
  if p.name not in planned_sources:counts['excluded_unplanned_files']+=1;continue
  d=json.loads(p.read_text());counts['attempted_files']+=1
  if d.get('status')!='completed':counts['unscored']+=1;continue
  assert d['official_split']=='train';tasks.add((d['domain'],d['task_id']));counts['scored']+=1;label=int(d['reward']['reward'])-int(d['raw_reward']);counts[{1:'rescue',0:'tie',-1:'harm'}[label]]+=1;counts[d['fit_split']+'_pairs']+=1;sources.append(p.name)
  if label>0:rescue_tasks.add((d['domain'],d['task_id']))
  if label<0:harm_tasks.add((d['domain'],d['task_id']))
  if d['raw_reward']==0:failed_tasks_probed.add((d['domain'],d['task_id']))
  kind='includes_nl_judge' if definitions[d['domain']][str(d['task_id'])] else 'deterministic_only';by_kind[kind][{1:'rescue',0:'tie',-1:'harm'}[label]]+=1
 summary['training_branch_ledger']={**dict(counts),'distinct_tasks':len(tasks),'distinct_rescued_tasks':len(rescue_tasks),'distinct_harmed_tasks':len(harm_tasks),'failed_tasks_with_completed_branch':len(failed_tasks_probed),'by_evaluator_kind':{k:dict(v) for k,v in by_kind.items()},'source_list_sha256':hashlib.sha256('\n'.join(sources).encode()).hexdigest(),'sampling':'Hash-selected prefixes, up to3 from failed raw tasks and1 from successful tasks; all logged alternatives attempted. Partial coverage cannot establish candidate absence.'}
 path=ROOT/'suite_summary.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(path)
 print(json.dumps({'test':summary['groups']['pooled/test'],'training_branch_ledger':summary['training_branch_ledger']},indent=2))
if __name__=='__main__':main()
