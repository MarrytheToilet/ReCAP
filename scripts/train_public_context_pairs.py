"""Fit selective reranker entirely from public-plan continuation certificates."""
import json,pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.train_selective_pairs import features,scores
from scripts.train_context_pairs import extra_features
from scripts.run_certificate_objectives import read,metrics,bootstrap
ROOT=Path('results/experiments');OUT=ROOT/'public_context_pairs'
def load(split):
 rows=read(ROOT/'natural_context'/f'{split}.jsonl');costs={(r['task_id'],r['step_index']):r for r in read(ROOT/'public_plan_progress_replay'/f'{split}.jsonl')}
 return [{**r,'candidate_costs':costs[(r['task_id'],r['step_index'])]['candidate_costs']} for r in rows]
def prepare(rows):
 x,loc,y,w=features(rows);return np.c_[x,extra_features(rows,loc)],loc,y,w

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 train=load('train');valid=load('valid');x,_,y,w=prepare(train);vx,vl,_,_=prepare(valid);known=y>=0
 protocol={'labels':'Only observable public-plan continuation costs, budget20, no policy_commands or hidden facts in data collection; finite comparisons only','features':'Same structured public context as context_pairs; no pretrained embedding needed','selection':'Natural validation net improvement with <=1% harm and zero unresolved interventions','evaluation':'Chosen policy evaluated against both public-continuation costs and original privileged-verifier costs, neither test used for tuning','train_pairs':int(known.sum()),'train_class_counts':np.bincount(y[known],minlength=3).tolist()}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));best=None;search=[]
 for leaves in [7,15,31]:
  for l2 in [1,10]:
   model=HistGradientBoostingClassifier(max_leaf_nodes=leaves,l2_regularization=l2,max_iter=200,min_samples_leaf=30,early_stopping=False,random_state=0);model.fit(x[known],y[known],sample_weight=w[known]);assert list(model.classes_)==[0,1,2]
   prob=model.predict_proba(vx)
   for penalty in [1,2,5]:
    z=scores(valid,vl,prob,penalty)
    for threshold in np.unique(np.r_[0,np.quantile(np.maximum(z.max(1),0),np.linspace(0,1,101)),1.01]):
     m,_=metrics(valid,z,float(threshold))
     if m['harm_rate']>.01 or m['unresolved_interventions']:continue
     value=(m['net_improvements'],-m['harmed'],-m['interventions'])
     if best is None or value>best[0]:best=(value,model,penalty,float(threshold),{'leaves':leaves,'l2':l2},m)
   search.append({'leaves':leaves,'l2':l2,'best_validation':best[5]});print('fit',leaves,l2,best[0],flush=True)
 _,model,penalty,threshold,hyper,vm=best
 with (OUT/'model.pkl').open('wb') as f:pickle.dump({'model':model,'penalty':penalty,'threshold':threshold,'hyperparameters':hyper},f)
 (OUT/'validation_search.json').write_text(json.dumps(search,indent=2))
 test=load('test');tx,tl,_,_=prepare(test);z=scores(test,tl,model.predict_proba(tx),penalty)
 original=read(ROOT/'natural_context/test.jsonl');assert [(r['task_id'],r['step_index'],r['candidates']) for r in test]==[(r['task_id'],r['step_index'],r['candidates']) for r in original]
 result={'hyperparameters':hyper,'penalty':penalty,'threshold':threshold,'validation':vm}
 for verifier,rows in [('public',test),('privileged',original)]:
  for gate,t in [('gated',threshold),('ungated',0.)]:
   name=verifier+'_'+gate;m,detail=metrics(rows,z,t);m['net_improvement_rate_ci95']=bootstrap(detail);result[name]=m;(OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in detail))
 (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
