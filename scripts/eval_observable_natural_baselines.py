"""Fixed public-state rules on the same frozen logged support and all decisions."""
import datetime,json
from pathlib import Path
import numpy as np
from scripts.run_certificate_objectives import read,metrics,bootstrap
from recap.agents.lm_policy_agent import rank_lm_policy_pool
ROOT=Path('results/experiments');OUT=ROOT/'observable_natural_baselines'
def pick(row,rule):
 candidates=row['candidates'];raw=row['executed_action'];admissible=[a for a in candidates if a in row['admissible_actions']]
 if rule=='rank2':return candidates[min(1,len(candidates)-1)]
 if rule=='first_admissible':return raw if raw in admissible else next(iter(admissible),raw)
 if rule=='avoid_immediate_repeat':
  if row['history'] and raw==row['history'][-1]:return next((a for a in admissible if a!=raw),raw)
  return raw
 if rule=='progress_ranker':
  if not admissible:return raw
  return rank_lm_policy_pool(tuple(admissible),tuple(row['history']),objective=row['initial_observation'],mode='progress')[0]
 raise ValueError(rule)
def main():
 OUT.mkdir(exist_ok=True);rules=['rank2','first_admissible','avoid_immediate_repeat','progress_ranker']
 protocol={'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rules':rules,'scope':'Same frozen candidate support, current public admissibility, public objective and executed history. No training or threshold search. Evaluated as additional diagnostic baselines after the learned-selector results were known.'}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));result={}
 for split in ['valid','test']:
  rows=read(ROOT/'natural_context'/f'{split}.jsonl');result[split]={}
  for rule in rules:
   scores=np.zeros((len(rows),max(len(r['candidates']) for r in rows)))
   for i,r in enumerate(rows):scores[i,r['candidates'].index(pick(r,rule))]=1.
   m,details=metrics(rows,scores,.5);m['net_rate_ci95']=bootstrap(details);result[split][rule]=m
   (OUT/f'{split}_{rule}.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in details))
 (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['test'],indent=2))
if __name__=='__main__':main()
