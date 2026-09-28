"""Replay logged candidates with ordinary model continuations, never gold action paths."""
import argparse, importlib, json, time
from pathlib import Path
from scripts.run_tau2_candidates import (ROOT, Backend, CandidateAgent, UserSimulator,
    user_module, judge_module, Orchestrator, SimulationRun, evaluate_simulation, EvaluationType,
    AssistantMessage, MultiToolMessage, ToolCall, candidate_error, CandidateFormatError)
from tau2.data_model.tasks import InitialState
from tau2.orchestrator.orchestrator import Role
from tau2.utils.utils import get_now

class ForcedAgent(CandidateAgent):
    def __init__(self, forced, **kwargs):
        super().__init__(**kwargs); self.forced=forced
    def _generate_next_message(self,message,state):
        if self.forced is None: return super()._generate_next_message(message,state)
        state.messages.extend(message.tool_messages if isinstance(message,MultiToolMessage) else [message])
        record=self.forced;self.forced=None;self.records.append(record)
        a=record['candidates'][record['selected']]
        error=candidate_error(a,self.tools)
        if error:
            record['format_error']=error
            raise CandidateFormatError(error)
        if a['kind']=='message':return AssistantMessage(role='assistant',content=a['content'],cost=0)
        return AssistantMessage(role='assistant',tool_calls=[ToolCall(id=f'call_{len(self.records)}',name=a['name'],arguments=a['arguments'])],cost=0)

def run_branch(original, source, decision, choice, backend, judge, domain):
    module=importlib.import_module(f'tau2.domains.{domain}.environment')
    sim=SimulationRun.model_validate(source['simulation']);messages=sim.messages
    # Current baseline emits one assistant action per record; reject ambiguous mapping.
    positions=[i for i,m in enumerate(messages) if isinstance(m,AssistantMessage)]
    assert len(positions)==sum('candidates' in r and 'format_error' not in r for r in source['decisions'])+1, 'Nonstandard initial history: mapping requires explicit audit'
    logged=source['decisions'][decision]
    if 'format_error' in logged:
        assert decision==len(source['decisions'])-1, 'Malformed action must be the terminal attempt'
        pos=len(messages)
    else:
        pos=positions[decision+1]
        a=logged['candidates'][logged['selected']];m=messages[pos]
        assert ((a['kind']=='message' and m.content==a['content']) or (a['kind']=='tool' and m.tool_calls and m.tool_calls[0].name==a['name'] and m.tool_calls[0].arguments==a['arguments']))
    prefix=messages[:pos]
    copied=original.model_copy(deep=True)
    copied.initial_state=(original.initial_state.model_copy(deep=True) if original.initial_state else InitialState())
    copied.initial_state.message_history=prefix
    records=source['decisions'][:decision].copy();forced={**logged,'selected':choice,'forced_logged_candidate':True}
    forced.pop('format_error',None)
    env=module.get_environment()
    agent=ForcedAgent(forced=forced,backend=backend,rank=1,records=records,tools=env.get_tools(),domain_policy=env.get_policy(),llm=backend.model)
    user=UserSimulator(llm=backend.model,instructions=str(original.user_scenario))
    orch=Orchestrator(domain=domain,agent=agent,user=user,environment=env,task=copied,max_steps=source['max_steps'],seed=0,timeout=1800)
    orch._run_start_time=get_now();orch._run_start_perf=time.perf_counter();orch.initialize()
    check=module.get_environment();init=original.initial_state
    check.set_state(initialization_data=init.initialization_data if init else None,initialization_actions=init.initialization_actions if init else None,message_history=prefix)
    prefix_hash=env.get_db_hash()
    assert prefix_hash==check.get_db_hash(),'Prefix database restoration mismatch'
    assert orch.to_role==Role.AGENT,'Branch prefix is not an agent decision'
    # One tool per action means each orchestrator step adds one trajectory message.
    assert all(not getattr(x,'tool_calls',None) or len(x.tool_calls)==1 for x in prefix)
    orch.step_count=len(prefix)-1
    while not orch.done:
        orch.step();orch._check_termination()
    result=orch._finalize()
    out={'domain':domain,'task_id':original.id,'decision':decision,'candidate_index':choice,'prefix_messages':len(prefix),'prefix_db_hash':prefix_hash,'prefix_restoration_verified':True,'suffix_agent_decisions':len(records)-decision,'simulation':result.model_dump(mode='json'),'decisions':records,'status':'ungraded'}
    if result.termination_reason.value=='timeout':
        out.update(status='interrupted_timeout',error='Infrastructure wall-clock guard; replay cached branch to the unchanged step budget')
        return out
    try:
        reward=evaluate_simulation(result,original,EvaluationType.ALL,False,domain)
        out.update(status='completed',reward=reward.model_dump(mode='json'),grading_version='strict-nl-v1')
    except Exception as e:out['grading_error']=repr(e)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',required=True);p.add_argument('--split',default='train',choices=['train','test']);p.add_argument('--limit',type=int,default=4);p.add_argument('--source',type=Path,default=ROOT/'results/experiments/new_benchmarks/tau2_aliyun');p.add_argument('--output',type=Path,default=ROOT/'results/experiments/new_benchmarks/tau2_branches');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    backend=Backend(args.source/'cache',ROOT/'.env.aliyun','qwen3.8-flash');judge=Backend(args.source/'judge_cache',ROOT/'.env.aliyun','qwen3.8-max');user_module.generate=backend.user;judge_module.generate=judge.judge
    module=importlib.import_module(f'tau2.domains.{args.domain}.environment');tasks=module.get_tasks(task_split_name=args.split)
    n=0
    for task in tasks:
        src=args.source/f'{args.domain}_{task.id}_rank1.json'
        if not src.exists():continue
        source=json.loads(src.read_text())
        if source.get('status')!='completed' or source['reward']['reward']==1:continue
        if n>=args.limit:break
        n+=1
        eligible=[i for i,r in enumerate(source['decisions']) if 'format_error' not in r and len(r.get('candidates',[]))>1 and any(a['kind']=='tool' for a in r['candidates'])][-2:]
        for decision in eligible:
            for choice in [0,1]:
                path=args.output/f'{args.domain}_{task.id}_d{decision}_c{choice}.json'
                if path.exists():continue
                print(json.dumps({'event':'branch_start','task':task.id,'decision':decision,'choice':choice}),flush=True)
                try:out=run_branch(task,source,decision,choice,backend,judge,args.domain)
                except Exception as e:
                    import traceback;traceback.print_exc();out={'status':'error','error':repr(e),'task_id':task.id,'decision':decision,'candidate_index':choice}
                out['split']=args.split;out['raw_reward']=source['reward']['reward'];out['raw_suffix_agent_decisions']=len(source['decisions'])-decision
                if choice==0 and out.get('status')=='completed':out['raw_outcome_reproduced']=out['reward']['reward']==source['reward']['reward']
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
                print(json.dumps({'event':'branch_result','task':task.id,'decision':decision,'choice':choice,'status':out['status'],'reward':out.get('reward',{}).get('reward'),'error':out.get('error')}),flush=True)
if __name__=='__main__':main()
