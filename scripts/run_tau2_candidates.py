"""Exploratory tau-bench candidate baselines; official environment and grading."""
import argparse, ast, fcntl, hashlib, json, os, sys, time
from scripts.tau2_grade_validation import validate_judge_output
from scripts.tau2_pair_selector import select as select_pair
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TAU2_DATA_DIR', str(ROOT/'third_party/tau2-bench/data'))
from loguru import logger
logger.remove(); logger.add(sys.stderr, level='WARNING')
from openai import OpenAI
import httpx
from dotenv import dotenv_values
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType
from tau2.user.user_simulator import UserSimulator
import tau2.user.user_simulator as user_module
import tau2.evaluator.evaluator_nl_assertions as judge_module
from tau2.utils.llm_utils import to_litellm_messages
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
import importlib

class CandidateFormatError(ValueError):
    pass

def candidate_error(action, tools):
    if not isinstance(action,dict):return 'Candidate is not an object'
    if action.get('kind')=='tool':
        if action.get('name') not in {t.name for t in tools}:return 'Unknown tool'
        if not isinstance(action.get('arguments'),dict):return 'Arguments are not an object'
        return None
    if action.get('kind')=='message' and isinstance(action.get('content'),str) and action['content'].strip():return None
    return 'Invalid action kind or empty message'

class ScoredOrchestrator(Orchestrator):
    def step(self):
        try:
            return super().step()
        except CandidateFormatError:
            # A malformed model action is an agent failure, not a transport outage.
            self.done=True
            self.termination_reason=TerminationReason.AGENT_ERROR
            self.step_count+=1

Orchestrator=ScoredOrchestrator

class Backend:
    def __init__(self, cache, env_file=None, model="Qwen2.5-3B-Instruct"):
        config=dotenv_values(env_file) if env_file else {}
        self.model=model
        self.cache=cache; cache.mkdir(parents=True,exist_ok=True)
        base=config.get('OPENAI_BASE_URL','http://127.0.0.1:18913/v1')
        self.client=OpenAI(base_url=base,api_key=config.get('OPENAI_API_KEY','local'),timeout=240,max_retries=3,http_client=httpx.Client(trust_env=not base.startswith('http://127.0.0.1')))
    def call(self,messages):
        body=dict(model=self.model,messages=messages,temperature=0,max_tokens=(1024 if self.model=="Qwen2.5-3B-Instruct" else 4096))
        key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
        path=self.cache/(key+'.json')
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists(): return json.loads(path.read_text())['text']
            result=self.client.chat.completions.create(**body)
            text=result.choices[0].message.content
            tmp=path.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps({'request':body,'text':text,'usage':result.usage.model_dump()}));tmp.replace(path)
            return text

    def judge(self, messages, **kwargs):
        converted=to_litellm_messages(messages)
        expected=ast.literal_eval(converted[-1]['content'].split('expectedOutcomes:',1)[1].strip())
        if not isinstance(expected,list): raise ValueError('Unexpected official assertion prompt')
        error=None
        for attempt in range(3):
            request=list(converted)
            if attempt:
                request.append({'role':'user','content':f'Formatting retry {attempt}. Return ONLY the required JSON object with a results array. Include every expected outcome exactly once, unchanged. Each entry must contain expectedOutcome (string), reasoning (string), metExpectation (boolean). No other key names. Do not omit any outcome.'})
            raw=self.call(request)
            try:
                return AssistantMessage(role='assistant',content=validate_judge_output(raw,expected),cost=0)
            except (ValueError,TypeError,KeyError) as exc:
                error=exc
        raise ValueError('Judge output validation failed after 3 attempts: '+str(error))
    def user(self, messages, **kwargs):
        return AssistantMessage(role='assistant',content=self.call(to_litellm_messages(messages)),cost=0)

class CandidateAgent(LLMAgent):
    def __init__(self, backend, rank, records, selector=None, validity_filter=False, repair_json=False, **kwargs):
        super().__init__(**kwargs); self.backend=backend; self.rank=rank; self.records=records; self.selector=selector; self.validity_filter=validity_filter
        self.intervened=False; self.repair_json=repair_json
    def _generate_next_message(self,message,state):
        state.messages.extend(message.tool_messages if isinstance(message,MultiToolMessage) else [message])
        history=[]
        for m in state.messages:
            d={'role':m.role,'content':m.content}
            if getattr(m,'tool_calls',None): d['tool_calls']=[{'name':t.name,'arguments':t.arguments} for t in m.tool_calls]
            if getattr(m,'name',None): d['name']=m.name
            history.append(d)
        instruction='''Return ONLY a JSON object {"candidates": [action1, action2, action3]} containing up to THREE distinct plausible next actions, ranked best first. An action is either {"kind":"tool","name":"tool_name","arguments":{...}} or {"kind":"message","content":"message to customer"}. Use exactly one tool per action. Follow policy, do not invent identifiers. If only one action makes sense return one. Do not include explanations outside JSON.'''
        prompt=[{'role':'system','content':self.system_prompt+'\n'+instruction+'\nAVAILABLE TOOLS:\n'+json.dumps([t.openapi_schema if hasattr(t,'openapi_schema') else t.openai_schema for t in self.tools])}, {'role':'user','content':'CONVERSATION:\n'+json.dumps(history)+'\nChoose the next actions.'}]
        raw=self.backend.call(prompt)
        text=(raw or '').strip()
        if text.startswith('```'): text=text.split('\n',1)[1].rsplit('```',1)[0]
        try:
            if self.repair_json:
                from scripts.tau2_candidate_json import parse_candidates
                candidates=parse_candidates(text)
            else:candidates=json.loads(text)['candidates']
            unique=[]
            if not isinstance(candidates,list):raise ValueError('Candidates must be an array')
            for a in candidates:
                if a not in unique: unique.append(a)
            if not unique:raise ValueError('Empty candidate array')
        except Exception as e:
            self.records.append({'format_error':str(e),'raw':raw}); raise CandidateFormatError('Invalid candidate JSON: '+str(e))
        selected=min(self.rank-1,len(unique)-1)
        errors=[candidate_error(c,self.tools) for c in unique]
        if self.selector is not None and not self.intervened:
            selected=select_pair(self.selector,{'candidates':unique,'history':history,'candidate_errors':errors})
            if selected!=0:self.intervened=True
        if self.validity_filter and errors[selected]:selected=next((i for i,e in enumerate(errors) if e is None),selected)
        a=unique[selected]
        record={'candidates':unique,'selected':selected,'raw':raw,'history':history,'candidate_errors':errors,'selector_intervened':self.intervened}
        self.records.append(record)
        if errors[selected]:
            record['format_error']=errors[selected]
            raise CandidateFormatError(errors[selected])
        print(json.dumps({'event':'decision','rank':self.rank,'n':len(self.records),'action':a}),flush=True)
        if a['kind']=='message': return AssistantMessage(role='assistant',content=a['content'],cost=0)
        return AssistantMessage(role='assistant',tool_calls=[ToolCall(id=f'call_{len(self.records)}',name=a['name'],arguments=a['arguments'])],cost=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repair-json',action='store_true');p.add_argument('--guard-user-stop',action='store_true');p.add_argument('--task-ids',nargs='+');p.add_argument('--env-file');p.add_argument('--selector',type=Path);p.add_argument('--validity-filter',action='store_true');p.add_argument('--cache-dir',type=Path);p.add_argument('--task-split',default='base',choices=['base','train','test']);p.add_argument('--judge-model',default='qwen3.8-max');p.add_argument('--model',default='Qwen2.5-3B-Instruct');p.add_argument('--domains',nargs='+',default=['retail','airline']);p.add_argument('--count',type=int,default=2);p.add_argument('--start-index',type=int,default=0);p.add_argument('--ranks',nargs='+',type=int,default=[1,2]);p.add_argument('--max-steps',type=int,default=60);p.add_argument('--output',type=Path,default=ROOT/'results/experiments/new_benchmarks/tau2');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);backend=Backend(args.cache_dir or args.output/'cache',args.env_file,args.model);user_module.generate=backend.user
    if args.guard_user_stop:
        from scripts.tau2_user_stop_guard import UserStopGuard
        guard=UserStopGuard(backend,Backend((args.cache_dir.parent if args.cache_dir else args.output)/'stop_guard_cache',args.env_file,'qwen3.8-max'))
        user_module.generate=guard.user
    selector=json.loads(args.selector.read_text()) if args.selector else None
    if selector is not None and selector.get('feature_code_sha256'):
        actual=hashlib.sha256((Path(__file__).parent/'tau2_pair_selector.py').read_bytes()).hexdigest()
        if actual!=selector['feature_code_sha256']:
            raise ValueError('Selector feature code differs from frozen training artifact')
    selector_hash=hashlib.sha256(args.selector.read_bytes()).hexdigest() if args.selector else ('validity-filter-v1' if args.validity_filter else None)
    if args.selector and args.validity_filter:raise ValueError('Separate learned and validity-filter arms')
    if selector is not None and args.ranks!=[1]:raise ValueError('Learned selector is evaluated as one arm with --ranks 1')
    judge=Backend((args.cache_dir.parent if args.cache_dir else args.output)/'judge_cache',args.env_file,args.judge_model);judge_module.generate=judge.judge
    for domain in args.domains:
        module=importlib.import_module(f'tau2.domains.{domain}.environment')
        tasks=module.get_tasks(task_split_name=args.task_split)[args.start_index:args.count]
        if args.task_ids:tasks=[t for t in tasks if t.id in args.task_ids]
        for task in tasks:
            lock_dir=args.output/'task_locks';lock_dir.mkdir(exist_ok=True)
            lock=(lock_dir/f'{domain}_{task.id}.lock').open('a')
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close();continue
            for rank in args.ranks:
                path=args.output/f'{domain}_{task.id}_rank{rank}.json'
                previous={}
                if path.exists():
                    previous=json.loads(path.read_text())
                    if previous.get('user_stop_protocol','upstream')!=('public-authorization-guard-v1' if args.guard_user_stop else 'upstream'):raise ValueError('Simulator protocol mismatch')
                    if previous.get('action_protocol_version','candidate-local-validation-v2')!=('candidate-local-validation-v3-json-repair' if args.repair_json else 'candidate-local-validation-v2'):raise ValueError('Parser protocol mismatch')
                    if previous.get('selector_sha256')!=selector_hash:raise ValueError('Output folder contains a different selector regime')
                    needs_nl=task.evaluation_criteria and RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis
                    was_timeout=previous.get('simulation',{}).get('termination_reason')=='timeout'
                    if previous.get('status')=='completed' and not was_timeout and (not needs_nl or previous.get('grading_version')=='strict-nl-v1'): continue
                    archive=args.output/'errors';archive.mkdir(exist_ok=True)
                    path.replace(archive/(path.stem+f'.{time.time_ns()}.json'))
                    if was_timeout:previous={}
                records=previous.get('decisions',[]) if previous.get('simulation') else [];env=module.get_environment();agent=CandidateAgent(backend,rank,records,selector=selector,validity_filter=args.validity_filter,repair_json=args.repair_json,tools=env.get_tools(),domain_policy=env.get_policy(),llm=args.model)
                user=UserSimulator(llm=args.model,instructions=str(task.user_scenario))
                orch=Orchestrator(domain=domain,agent=agent,user=user,environment=env,task=task,max_steps=args.max_steps,seed=0,timeout=1800)
                result={'domain':domain,'task_id':task.id,'rank':rank,'model':args.model,'user_model':args.model,'user_stop_protocol':('public-authorization-guard-v1' if args.guard_user_stop else 'upstream'),'max_steps':args.max_steps,'judge_model':args.judge_model,'selector_sha256':selector_hash,'action_protocol_version':('candidate-local-validation-v3-json-repair' if args.repair_json else 'candidate-local-validation-v2'),'status':'error'}
                print(json.dumps({'event':'start',**{**result,'status':'running'}}),flush=True)
                try:
                    sim=SimulationRun.model_validate(previous['simulation']) if previous.get('simulation') else orch.run()
                    if sim.termination_reason==TerminationReason.TIMEOUT:
                        result['interrupted_simulation']=sim.model_dump(mode='json')
                        raise TimeoutError('Infrastructure wall-clock guard; resume cached trajectory to the unchanged step budget')
                    result['simulation']=sim.model_dump(mode='json');reward=evaluate_simulation(sim,task,EvaluationType.ALL,False,domain)
                    result.update(status='completed',reward=reward.model_dump(mode='json'),simulation=sim.model_dump(mode='json'),grading_version='strict-nl-v1')
                except Exception as e:
                    import traceback;traceback.print_exc();result['error']=repr(e)
                result['decisions']=records;tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path);print(json.dumps({'event':'result',**{k:v for k,v in result.items() if k not in ['simulation','decisions']}}),flush=True)
            lock.close()
if __name__=='__main__': main()
