"""Verify every alternative at sampled official-training prefixes; never read test tasks."""
import argparse,datetime,fcntl,hashlib,importlib,json,time
from scripts.run_tau2_branches import run_branch,ROOT,Backend,user_module,judge_module
SCOPE='all-logged-alternatives-v1'
def finished(path):
 return path.exists() and json.loads(path.read_text()).get('candidate_scope')==SCOPE

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--max-hours',type=float,default=6);args=p.parse_args()
 if args.max_hours<=0:p.error('--max-hours must be positive')
 root=ROOT/'results/experiments/new_benchmarks';source=root/'tau2_aliyun';out=root/'tau2_training_branches_uniform';out.mkdir(exist_ok=True);(out/'locks').mkdir(exist_ok=True)
 backend=Backend(source/'cache',ROOT/'.env.aliyun','qwen3.8-flash');judge=Backend(source/'judge_cache',ROOT/'.env.aliyun','qwen3.8-max');user_module.generate=backend.user;judge_module.generate=judge.judge
 tasks=[]
 for domain in ['retail','airline']:
  module=importlib.import_module(f'tau2.domains.{domain}.environment');tasks.extend((domain,t) for t in module.get_tasks(task_split_name='train'))
 tasks.sort(key=lambda dt:hashlib.sha256((str(args.worker)+dt[0]+dt[1].id).encode()).hexdigest())
 deadline=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=args.max_hours);attempts={}
 while datetime.datetime.now(datetime.timezone.utc)<deadline:
  progress=False;remaining=0
  for domain,task in tasks:
   if datetime.datetime.now(datetime.timezone.utc)>=deadline:return
   dest=out/f'{domain}_{task.id}_summary.json'
   if finished(dest):continue
   remaining+=1;path=source/f'{domain}_{task.id}_rank1.json'
   if not path.exists():continue
   d=json.loads(path.read_text())
   if d.get('status')!='completed':continue
   lock=(out/'locks'/f'{domain}_{task.id}.lock').open('a')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();continue
   try:
    if finished(dest):continue
    eligible=[i for i,r in enumerate(d['decisions']) if len(r.get('candidates',[]))>=2]
    if not eligible:
     dest.write_text(json.dumps({'domain':domain,'task_id':task.id,'status':'no_multicandidate_decision','candidate_scope':SCOPE,'branches':[]}));progress=True;continue
    ordered=sorted(eligible,key=lambda i:hashlib.sha256(f'tau2-prefix-v1:{domain}:{task.id}:{i}'.encode()).hexdigest())
    points=sorted(ordered[:3] if d['reward']['reward']==0 else ordered[:1])
    fit='valid' if int(hashlib.sha256(('tau2-selector-v1:'+domain+':'+task.id).encode()).hexdigest()[:8],16)%5==0 else 'train'
    plan={'domain':domain,'task_id':task.id,'official_split':'train','fit_split':fit,'raw_reward':d['reward']['reward'],'candidate_scope':SCOPE,'points':[{'decision':i,'record':d['decisions'][i],'sampling_weight':len(eligible)/len(points),'choices':list(range(1,len(d['decisions'][i]['candidates'])))} for i in points]}
    (out/f'{domain}_{task.id}_plan.json').write_text(json.dumps(plan,indent=2));completed=[];expected=sum(len(p['choices']) for p in plan['points'])
    for point in plan['points']:
     decision=point['decision']
     for choice in point['choices']:
      if datetime.datetime.now(datetime.timezone.utc)>=deadline:return
      branch_path=out/f'{domain}_{task.id}_d{decision}_c{choice}.json'
      if branch_path.exists() and json.loads(branch_path.read_text()).get('status')=='completed':completed.append(branch_path.name);continue
      key=(domain,task.id,decision,choice)
      if attempts.get(key,0)>=3:continue
      attempts[key]=attempts.get(key,0)+1
      print(json.dumps({'event':'training_branch','domain':domain,'task':task.id,'decision':decision,'choice':choice,'attempt':attempts[key]}),flush=True)
      try:
       result=run_branch(task,d,decision,choice,backend,judge,domain)
       result.update(action_protocol_version='candidate-local-validation-v2',logged_decision=point['record'],prefix_sampling_weight=point['sampling_weight'],eligible_prefixes=len(eligible),sampled_prefixes=len(points),raw_reward=d['reward']['reward'],raw_suffix_agent_decisions=len(d['decisions'])-decision,official_split='train',fit_split=fit)
      except Exception as e:
       import traceback;traceback.print_exc();result={'status':'error','domain':domain,'task_id':task.id,'decision':decision,'candidate_index':choice,'error':repr(e)}
      with (out/f'attempts_worker{args.worker}.jsonl').open('a') as f:f.write(json.dumps({'domain':domain,'task_id':task.id,'decision':decision,'candidate_index':choice,'status':result['status'],'error':result.get('error'),'grading_error':result.get('grading_error')})+'\n')
      tmp=branch_path.with_suffix(f'.{args.worker}.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(branch_path);progress=True
      if result.get('status')=='completed':completed.append(branch_path.name)
    if len(completed)==expected:dest.write_text(json.dumps({'domain':domain,'task_id':task.id,'status':'completed','candidate_scope':SCOPE,'branches':completed}))
   finally:lock.close()
  if not remaining:break
  if not progress:time.sleep(20)
 print('training collector finished',flush=True)
if __name__=='__main__':main()
