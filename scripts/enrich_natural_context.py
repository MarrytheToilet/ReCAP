"""Restore observable TextWorld context without requesting privileged policy paths."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.eval_pool_baselines import make_env
ROOT=Path('results/experiments');OUT=ROOT/'natural_context'

def collect(task,rows):
    dest=OUT/'shards'/(Path(task).stem+'.json')
    if dest.exists():return json.loads(dest.read_text())
    env=make_env(task);result=[]
    try:
        state=env.reset();initial=state.feedback;history=[];feedback=[]
        for r in sorted(rows,key=lambda x:x['step_index']):
            assert history==r['history']
            result.append({**r,'initial_observation':initial,'observation':state.feedback,'observation_history':feedback.copy(),'admissible_actions':list(state.admissible_commands)})
            feedback.append(state.feedback);state,_,_=env.step(r['executed_action']);history.append(r['executed_action'])
    finally:env.close()
    dest.write_text(json.dumps(result));return result

def main():
    (OUT/'shards').mkdir(parents=True,exist_ok=True)
    for split in ['train','valid','test']:
        rows=[json.loads(l) for l in (ROOT/'natural'/f'{split}.jsonl').read_text().splitlines()];groups={}
        for r in rows:groups.setdefault(r['task_id'],[]).append(r)
        lookup={}
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(collect,t,g) for t,g in groups.items()]
            for i,f in enumerate(as_completed(futures),1):
                for r in f.result():lookup[(r['task_id'],r['step_index'])]=r
                if i%50==0:print(split,i,len(groups),flush=True)
        (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(lookup[(r['task_id'],r['step_index'])])+'\n' for r in rows))
        print('completed',split,len(rows),flush=True)
if __name__=='__main__':main()
