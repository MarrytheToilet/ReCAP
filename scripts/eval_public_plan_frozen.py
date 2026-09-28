"""Frozen public-plan baseline on validation and matched held-out local-policy games."""
import json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.eval_public_instruction_plan import clauses,choose
from scripts.eval_pool_baselines import make_env
from recap.agents.lm_policy_agent import rank_lm_policy_pool
ROOT=Path('results/experiments');OUT=ROOT/'public_plan'

def run(task,limit):
    env=make_env(task);history=[];skipped=[]
    try:
        state=env.reset();initial=state.feedback;plan=clauses(initial)
        for clause in plan:
            if len(history)>=30 or state.won:break
            actions=list(state.admissible_commands)
            if limit:actions=rank_lm_policy_pool(tuple(actions),tuple(history),objective=initial,mode='progress')[:limit]
            action=choose(clause,actions)
            if action is None:skipped.append(clause);continue
            state,_,done=env.step(action);history.append(action)
            if done:break
        return {'task_id':task,'success':bool(state.won),'num_steps':len(history),'actions':history,'skipped':skipped,'plan':plan,'pool_limit':limit}
    finally:env.close()

def main():
    OUT.mkdir(exist_ok=True)
    protocol={'frozen_parser_sha256':hashlib.sha256(Path('scripts/eval_public_instruction_plan.py').read_bytes()).hexdigest(),'development':'First 30 natural-train tasks only; regex rules frozen before validation and held-out evaluation.','inputs':'Full public initial observation, executed history and admissible actions. No policy_commands, facts or hidden goal predicates.','arms':['full admissible support','same progress-ranked pool of 20 as local-LM controls'],'horizon':30,'validation':'66 natural-valid tasks','heldout':'100 hard + 100 xhard matched local-policy evaluation games','limitations':'Benchmark-specific grammar parser; larger public-context budget than truncated LM prompts. No claim of learned ReCAP improvement.'}
    (OUT/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2))
    suites={'valid':list(dict.fromkeys(json.loads(l)['task_id'] for l in open(ROOT/'natural/valid.jsonl')))}
    suites.update({d:Path(f'analysis/localpolicy_eval/ctrl_{d}100_games.txt').read_text().split() for d in ['hard','xhard']})
    summary={}
    for suite,tasks in suites.items():
        for limit in [0,20]:
            name=f'{suite}_pool{limit}';path=OUT/f'{name}.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else [];seen={r['task_id'] for r in rows}
            with ProcessPoolExecutor(max_workers=6) as pool,path.open('a') as f:
                futures=[pool.submit(run,t,limit) for t in tasks if t not in seen]
                for future in as_completed(futures):r=future.result();rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
            summary[name]={'n':len(rows),'success':sum(r['success'] for r in rows),'success_rate':sum(r['success'] for r in rows)/len(rows),'mean_steps':sum(r['num_steps'] for r in rows)/len(rows)}
            (OUT/'frozen_results.json').write_text(json.dumps(summary,indent=2));print(name,summary[name],flush=True)
if __name__=='__main__':main()
