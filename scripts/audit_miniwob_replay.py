"""Replay the original action and suffix on12 predeclared training instances."""
import argparse,json,hashlib
from pathlib import Path
from scripts.collect_miniwob_branches import branch,bench
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);args=p.parse_args();out=bench.OUT/'raw_replay_audit';out.mkdir(exist_ok=True)
jobs=[(t,s) for t in bench.TASKS for s in [0,1]]
for index,(task,seed) in enumerate(jobs):
 if index%2!=args.worker:continue
 src=bench.OUT/'train'/f'{task}_{seed}_raw_rank1.json';dest=out/f'{task}_{seed}.json'
 if dest.exists():continue
 raw=json.loads(src.read_text());decision=next(i for i,r in enumerate(raw['records']) if r.get('candidates'))
 try:
  repeated=branch(task,seed,raw,decision,0);d={'task':task,'seed':seed,'status':repeated['status'],'prefix_match_verified':repeated['prefix_match_verified'],'outcome_reproduced':repeated['reward']==raw['reward'],'raw_reward':raw['reward'],'repeated_reward':repeated['reward'],'raw_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'recorded_steps':len(raw['records']),'repeated_steps':len(repeated['records'])}
 except Exception as e:d={'task':task,'seed':seed,'status':'error','error':repr(e)}
 dest.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
