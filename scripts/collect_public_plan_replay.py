"""Ordinary observable continuation: return via observed room graph, follow public plan."""
import argparse,json,re,hashlib
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.eval_public_instruction_plan import clauses,choose
from scripts.eval_pool_baselines import make_env
ROOT=Path('results/experiments')
OPPOSITE={'north':'south','south':'north','east':'west','west':'east'}
def room(feedback,previous=None):
    m=re.search(r'-=\s*(.*?)\s*=-',feedback)
    return m.group(1) if m else previous

def continuation(task,history,action,budget=20):
    env=make_env(task);suffix=[];calls=0
    try:
        state=env.reset();initial=state.feedback;plan=clauses(initial);start=room(initial);current=start;anchor=start;graph={};done=False;cursor=0;ontrack=True
        for a in list(history)+[action]:
            target=None;target_index=cursor
            if ontrack:
                for index in range(cursor,len(plan)):
                    target=choose(plan[index],state.admissible_commands)
                    if target is not None:target_index=index;break
            previous=current;state,_,done=env.step(a);calls+=1;current=room(state.feedback,current)
            if ontrack and a==target:
                cursor=target_index+1;anchor=current
            elif ontrack and current!=previous:
                ontrack=False;anchor=previous
            elif not ontrack and current==anchor:
                ontrack=True
            if a.startswith('go ') and current!=previous:
                graph.setdefault(previous,{})[current]=a;graph.setdefault(current,{})[previous]='go '+OPPOSITE[a.split()[1]]
            if done:break
        queue=deque([(current,[])]);seen={current};route=None
        while queue:
            node,path=queue.popleft()
            if node==anchor:route=path;break
            for neighbor,a in graph.get(node,{}).items():
                if neighbor not in seen:seen.add(neighbor);queue.append((neighbor,path+[a]))
        if not done and route is not None:
            for a in route:
                if a not in state.admissible_commands or len(suffix)>=budget:break
                state,_,done=env.step(a);calls+=1;suffix.append(a)
                if done:break
            else:
                for clause in plan[cursor:]:
                    if done or len(suffix)>=budget:break
                    a=choose(clause,state.admissible_commands)
                    if a is None:continue
                    state,_,done=env.step(a);calls+=1;suffix.append(a)
        return {'success':bool(state.won),'cost':len(suffix) if state.won else None,'suffix':suffix,'calls':calls,'observed_return_path':route,'plan_cursor':cursor,'anchor_room':anchor}
    finally:env.close()

def task_rows(task,rows,folder):
    path=Path(folder)/'shards'/(Path(task).stem+'.json')
    if path.exists():return json.loads(path.read_text())
    results=[]
    for r in rows:
        trials={a:continuation(task,r['history'],a) for a in r['candidates']}
        results.append({**{k:r[k] for k in ['task_id','seed','step_index','trajectory_id','history','candidates','executed_action','rejected_action']},'candidate_costs':{a:v['cost'] for a,v in trials.items()},'trials':trials})
    path.write_text(json.dumps(results));return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='train');p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    out=ROOT/'public_plan_progress_replay';(out/'shards').mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in open(ROOT/'natural'/f'{args.split}.jsonl')]
    groups={}
    for r in rows:groups.setdefault(r['task_id'],[]).append(r)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(task_rows,t,g,str(out)) for t,g in groups.items()]
        for i,f in enumerate(as_completed(futures),1):results.extend(f.result());print('tasks',i,len(groups),flush=True)
    results.sort(key=lambda r:(r['task_id'],r['step_index']));(out/f'{args.split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    raw=alt=both=strict=0
    for r in results:
        rc=r['candidate_costs'][r['executed_action']];others=[v for a,v in r['candidate_costs'].items() if a!=r['executed_action'] and v is not None]
        raw+=rc is not None;alt+=bool(others);both+=rc is not None and bool(others);strict+=rc is not None and bool(others) and min(others)<rc
    summary={'split':args.split,'tasks':len(groups),'decisions':len(results),'raw_verified':raw,'alternative_verified':alt,'both_verified':both,'strict_finite_cost_improvements':strict,'suffix_budget':20,'policy':'Track matched public-plan progress; return to last matched waypoint through observed graph; follow remaining public instructions','oracle_policy_commands':False,'calls':sum(t['calls'] for r in results for t in r['trials'].values()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parser_sha256':hashlib.sha256(Path('scripts/eval_public_instruction_plan.py').read_bytes()).hexdigest()}
    (out/f'{args.split}_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
