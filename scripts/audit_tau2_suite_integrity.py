"""Audit formal task records against the frozen experiment manifest."""
import datetime,hashlib,json
from pathlib import Path
from scripts.summarize_tau2_suite import ARMS
ROOT=Path('results/experiments/new_benchmarks')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((ROOT/'full_baseline_manifest.json').read_text());splits=manifest['splits']
 definitions={d:{str(t['id']):t for t in json.loads(Path(f'third_party/tau2-bench/data/tau2/domains/{d}/tasks.json').read_text())} for d in splits}
 legacy=json.loads((ROOT/'legacy_protocol_equivalence_audit.json').read_text());assert not legacy['violations'];legacy={v['file']:v for v in legacy['checks']}
 frozen=json.loads((ROOT/'tau2_learned_selector/freeze.json').read_text())
 expected_hash={'raw_rank1':None,'always_rank2':None,'validity_filter':'validity-filter-v1','native_tools':'native-tool-api-v1','first_rank2_once':sha(ROOT/'first_rank2_selector.json'),'learned_once':frozen['selector_sha256']}
 violations=[];pending=[];records=[];planned=0
 for arm,(folder,rank) in ARMS.items():
  for domain,partitions in splits.items():
   tasks=partitions['train']+partitions['test'] if arm in ['raw_rank1','always_rank2'] else partitions['test']
   for task in tasks:
    planned+=1;path=ROOT/folder/f'{domain}_{task}_rank{rank}.json'
    if not path.exists():pending.append(str(path));continue
    d=json.loads(path.read_text())
    if d.get('status')!='completed':pending.append(str(path));continue
    problems=[];criteria=definitions[domain][task]['evaluation_criteria'];uses_nl=bool(criteria.get('nl_assertions')) and 'NL_ASSERTION' in criteria.get('reward_basis',[])
    for field,value in [('domain',domain),('task_id',task),('rank',rank),('model',manifest['agent_model']),('user_model',manifest['agent_model']),('max_steps',60),('selector_sha256',expected_hash[arm])]:
     if d.get(field)!=value:problems.append(field)
    if uses_nl and (d.get('grading_version')!='strict-nl-v1' or d.get('judge_model')!=manifest['nl_judge_model']):problems.append('NL_grading')
    protocol='upstream-LLMAgent-native-tools-v1' if arm=='native_tools' else 'candidate-local-validation-v2'
    if d.get('action_protocol_version')!=protocol:
     old=legacy.get(path.name)
     if folder!='tau2_aliyun' or old is None or old['sha256']!=sha(path):problems.append('action_protocol')
    normal=d['simulation']['termination_reason'] in ['agent_stop','user_stop']
    expected_basis=criteria.get('reward_basis') if normal else None
    if d['reward'].get('reward_basis')!=expected_basis:problems.append('reward_basis')
    if not normal and d['reward']['reward']!=0:problems.append('premature_termination_reward')
    if d['reward']['reward'] not in [0,1] or d['simulation']['termination_reason']=='timeout':problems.append('reward_or_timeout')
    if problems:violations.append({'file':str(path),'fields':problems})
    records.append({'file':str(path),'sha256':sha(path),'arm':arm,'task_requires_NL_on_normal_termination':uses_nl,'uses_NL_judge':uses_nl and normal})
 out={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned':planned,'scored':len(records),'pending':pending,'violations':violations,'records':records,'complete':not pending and not violations}
 (ROOT/'suite_integrity_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['planned','scored','pending','violations','complete']}))
 if violations:raise RuntimeError('Formal suite integrity violation')
if __name__=='__main__':main()
