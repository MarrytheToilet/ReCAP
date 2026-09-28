"""Official LLMAgent prompt/history with native API tool calls, same evaluator/user."""
import argparse,fcntl,hashlib,importlib,json,os,time
from pathlib import Path
from scripts.run_tau2_candidates import (ROOT,Backend,LLMAgent,UserSimulator,Orchestrator,
 AssistantMessage,ToolCall,SimulationRun,CandidateFormatError,to_litellm_messages,
 user_module,judge_module,evaluate_simulation,EvaluationType)
import tau2.agent.llm_agent as agent_module

def native_generate(backend,messages,tools,**kwargs):
 body={'model':backend.model,'messages':to_litellm_messages(messages),'tools':[t.openai_schema for t in tools],'temperature':0,'max_tokens':4096}
 key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();path=backend.cache/(key+'.json')
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if path.exists():response=json.loads(path.read_text())['response']
  else:
   result=backend.client.chat.completions.create(**body);response=result.choices[0].message.model_dump(mode='json')
   tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps({'request':body,'response':response,'usage':result.usage.model_dump()}));tmp.replace(path)
 try:
  calls=[ToolCall(id=t['id'],name=t['function']['name'],arguments=json.loads(t['function']['arguments'])) for t in response.get('tool_calls') or []]
  return AssistantMessage(role='assistant',content=response.get('content'),tool_calls=calls or None,cost=0)
 except (ValueError,TypeError,KeyError) as e:raise CandidateFormatError('Malformed native action: '+str(e))

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--split',default='test',choices=['train','test']);p.add_argument('--limit',type=int,default=114);a=p.parse_args()
 root=ROOT/'results/experiments/new_benchmarks';out=root/('tau2_native_test' if a.split=='test' else 'tau2_native_train_smoke');out.mkdir(exist_ok=True);(out/'task_locks').mkdir(exist_ok=True)
 backend=Backend(root/'tau2_aliyun/cache',ROOT/'.env.aliyun','qwen3.8-flash');judge=Backend(root/'tau2_aliyun/judge_cache',ROOT/'.env.aliyun','qwen3.8-max')
 user_module.generate=backend.user;judge_module.generate=judge.judge
 agent_module.generate=lambda messages,tools,**kw:native_generate(backend,messages,tools,**kw)
 domains=['retail','airline'] if a.worker%2==0 else ['airline','retail']
 for sweep in range(2):
  for domain in domains:
   module=importlib.import_module(f'tau2.domains.{domain}.environment')
   for task in module.get_tasks(task_split_name=a.split)[:a.limit]:
    lock=(out/'task_locks'/f'{domain}_{task.id}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();continue
    try:
     path=out/f'{domain}_{task.id}_rank1.json';old=json.loads(path.read_text()) if path.exists() else {}
     was_timeout=old.get('simulation',{}).get('termination_reason')=='timeout'
     if old.get('status')=='completed' and not was_timeout:continue
     if old:
      (out/'errors').mkdir(exist_ok=True);path.replace(out/'errors'/(path.stem+f'.{time.time_ns()}.json'))
     if was_timeout:old={}
     env=module.get_environment();agent=LLMAgent(tools=env.get_tools(),domain_policy=env.get_policy(),llm=backend.model);user=UserSimulator(llm=backend.model,instructions=str(task.user_scenario))
     result={'domain':domain,'task_id':task.id,'rank':1,'status':'error','model':backend.model,'user_model':backend.model,'judge_model':judge.model,'max_steps':60,'selector_sha256':'native-tool-api-v1','action_protocol_version':'upstream-LLMAgent-native-tools-v1','decisions':[]}
     print('start',domain,task.id,flush=True)
     try:
      sim=SimulationRun.model_validate(old['simulation']) if old.get('simulation') else Orchestrator(domain=domain,agent=agent,user=user,environment=env,task=task,max_steps=60,seed=0,timeout=1800).run()
      if sim.termination_reason.value=='timeout':
       result['interrupted_simulation']=sim.model_dump(mode='json');raise TimeoutError('Infrastructure wall-clock guard; resume cached trajectory')
      result['simulation']=sim.model_dump(mode='json');reward=evaluate_simulation(sim,task,EvaluationType.ALL,False,domain);result.update(status='completed',reward=reward.model_dump(mode='json'),grading_version='strict-nl-v1')
     except Exception as e:
      import traceback;traceback.print_exc();result['error']=repr(e)
     tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path);print('result',domain,task.id,result['status'],result.get('reward',{}).get('reward'),flush=True)
    finally:lock.close()
if __name__=='__main__':main()
