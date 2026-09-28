"""Task-cluster uncertainty for selective intervention precision, recall and harm."""
import json
from pathlib import Path
import numpy as np
root=Path('results/experiments/context_pairs');rows=[json.loads(l) for l in (root/'gated.jsonl').read_text().splitlines()];tasks=sorted({r['task_id'] for r in rows});stats=[]
for task in tasks:
 group=[r for r in rows if r['task_id']==task];stats.append([len(group),sum(r['intervention'] for r in group),sum(r['improved'] for r in group),sum(r['harmed'] for r in group),sum(r['opportunity'] for r in group)])
s=np.array(stats);rng=np.random.default_rng(20260928);b=s[rng.integers(len(s),size=(10000,len(s)))].sum(1)
def rates(a):
 n,intervened,good,harm,opportunities=a.T
 return {'intervention_precision':np.divide(good,intervened,out=np.full_like(good,np.nan,dtype=float),where=intervened>0),'repair_recall':good/opportunities,'harm_rate':harm/n,'net_improvement_rate':(good-harm)/n}
point=rates(s.sum(0)[None,:]);out={'tasks':len(tasks),'decisions':len(rows),'bootstrap_replicates':10000,'unit':'task','metrics':{name:{'estimate':float(point[name][0]),'ci95':np.nanquantile(values,[.025,.975]).tolist()} for name,values in rates(b).items()}}
(root/'uncertainty.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
