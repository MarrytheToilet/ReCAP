"""Three prespecified shuffled-label controls for the selected observable classifier."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.train_context_pairs import extra_features
from scripts.train_selective_pairs import features,scores
from scripts.run_certificate_objectives import read,metrics,bootstrap
ROOT=Path('results/experiments');OUT=ROOT/'context_label_placebo'
def prep(rows):
 x,loc,y,w=features(rows);return np.c_[x,extra_features(rows,loc)],loc,y,w
def main():
 OUT.mkdir(exist_ok=True);rows={s:read(ROOT/'natural_context'/f'{s}.jsonl') for s in ['train','valid']};x,_,y,w=prep(rows['train']);vx,vl,_,_=prep(rows['valid']);known=y>=0;results={};frozen=[]
 (OUT/'protocol.json').write_text(json.dumps({'architecture':'Selected structured_tree,15leaves,200iterations,min_leaf30,l2=1','controls':'Shuffle finite training pair labels globally with seeds0,1,2; retain inputs, weights, class counts. Do not select the best placebo seed.','selection':'Same validation penalty choices1,2,5 and101quantile gates; <=1% harm and no unresolved intervention. Test untouched until all three gates are frozen.'},indent=2))
 for seed in [0,1,2]:
  shuffled=np.random.default_rng(seed).permutation(y[known]);model=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=200,min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=0);model.fit(x[known],shuffled,sample_weight=w[known]);best=None
  for penalty in [1,2,5]:
   z=scores(rows['valid'],vl,model.predict_proba(vx),penalty)
   for t in np.unique(np.r_[0,np.quantile(np.maximum(z.max(1),0),np.linspace(0,1,101)),1.01]):
    m,_=metrics(rows['valid'],z,float(t))
    if m['harm_rate']>.01 or m['unresolved_interventions']:continue
    value=(m['net_improvements'],-m['harmed'],-m['interventions'])
    if best is None or value>best[0]:best=(value,penalty,float(t),m)
  frozen.append((seed,model,best));print('frozen',seed,best[0],flush=True)
 test=read(ROOT/'natural_context/test.jsonl');tx,tl,_,_=prep(test)
 for seed,model,(_,penalty,threshold,valid) in frozen:
  m,detail=metrics(test,scores(test,tl,model.predict_proba(tx),penalty),threshold);m['net_rate_ci95']=bootstrap(detail);results[seed]={'penalty':penalty,'threshold':threshold,'validation':valid,'test':m};(OUT/f'seed{seed}.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in detail))
 (OUT/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
