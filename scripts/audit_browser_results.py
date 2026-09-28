"""Read-only reproducibility audit of frozen browser rescue experiments."""
import ast, collections, hashlib, json
from pathlib import Path
R=Path('results/benchmarks'); W=R/'miniwob'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(W/'protocol.json'); splits=[set(p[k]) for k in ['train_seeds','validation_seeds','test_seeds']]
assert all(not a&b for i,a in enumerate(splits) for b in splits[i+1:])
a=read(W/'selected_selector.json'); freeze=read(W/'selector_freeze.json')
assert sha(W/'selected_selector.json')==freeze['sha256']
assert sha(Path('scripts/browser_pair_features.py'))==a['feature_code_sha256']
assert sha(Path('scripts/run_miniwob_recap.py'))==p['source_sha256']
for source,digest in a['source_sha256'].items():assert sha(Path(source))==digest
labels=collections.Counter(); branches=0
for task in p['tasks']:
 for seed in p['train_seeds']:
  stem=f'{task}_{seed}'; rawpath=W/'train'/f'{stem}_raw_rank1.json';raw=read(rawpath);s=read(W/'branches'/f'{stem}_summary.json')
  assert s['complete'] and s['raw_sha256']==sha(rawpath)
  eligible=[i for i,r in enumerate(raw['records']) if len(r.get('candidates',[]))>1]
  chosen=sorted(sorted(eligible,key=lambda i:hashlib.sha256(f'miniwob-recap-v1:{task}:{seed}:{i}'.encode()).hexdigest())[:1 if raw['reward'] else 3])
  assert s['points']==chosen
  expected={f'{stem}_d{i}_c{j}.json' for i in chosen for j in range(1,len(raw['records'][i]['candidates']))}
  assert set(s['outputs'])==expected
  for name in expected:
   b=read(W/'branches'/name);assert b['status']=='completed' and b['prefix_match_verified'];assert b['raw_sha256']==sha(rawpath)
   for i in range(b['decision']+1):
    assert b['records'][i]['state']==raw['records'][i]['state']
   pool=raw['records'][b['decision']]['candidates']
   assert ast.dump(ast.parse(pool[0]))!=ast.dump(ast.parse(pool[b['choice']]))
   assert b['raw_suffix_steps']==len(raw['records'])-b['decision']
   assert b['alternative_suffix_steps']==len(b['records'])-b['decision']
   label=int(b['reward']-raw['reward'])
   if not label and b['reward']:label=int(b['alternative_suffix_steps']<b['raw_suffix_steps'])-int(b['alternative_suffix_steps']>b['raw_suffix_steps'])
   labels[label]+=1;branches+=1
assert dict(map(lambda kv:(int(kv[0]),kv[1]),a['labels'].items()))==dict(labels)
replays=list((W/'raw_replay_audit').glob('*.json'));assert len(replays)==12
for f in replays:
 d=read(f);assert d['status']=='completed' and d['prefix_match_verified'] and d['outcome_reproduced'];assert d['recorded_steps']==d['repeated_steps']
final=read(R/'FINAL_RESULTS.json');assert all(v['complete'] and not v['errors'] for v in final['web']['arms'].values())
for f in (W/'test').glob('*.json'):
 assert read(f)['protocol_sha256']==sha(W/'protocol.json')
matched_retrievals=0
for f in (W/'memory_validation').glob('*.json'):
 original=read(f);shuffled=read(W/'shuffled_validation'/f.name)
 assert original['status']==shuffled['status']=='completed'
 if original['retrievals'] and shuffled['retrievals']:
  assert original['retrievals'][0]==shuffled['retrievals'][0];matched_retrievals+=1
for policy in ['reasoned','heuristic']:
 proto=W/f'recovery_{policy}_protocol.json';assert read(proto)['source_sha256']==sha(Path('scripts/run_browser_recovery.py'))
 for f in (W/f'recovery_{policy}_test').glob('*.json'):
  d=read(f);assert d['protocol_sha256']==sha(proto) and len(d['recovery'])<=4
out={'status':'PASS','disjoint_seed_splits':[len(x)*len(p['tasks']) for x in splits],'training_branches':branches,'labels':dict(labels),'original_action_replays':len(replays),'complete_final_runs':420,'frozen_selector_sha256':freeze['sha256'],'selected_threshold':a['threshold'],'checks':['source and feature hashes','training source lineage','sampling plan and complete denominator','strict pairwise labels','distinct alternative actions','exact replay outcomes and action counts','all final arms complete without infrastructure errors','recovery protocol and intervention budget']}
out['matched_first_decision_retrievals']=matched_retrievals
(R/'AUDIT_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
