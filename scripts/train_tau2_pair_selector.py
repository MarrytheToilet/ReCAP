"""Fit pair-supervised, one-shot selection; calibrate only on training-task holdout."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scripts.tau2_pair_selector import features,probabilities,select
ROOT=Path('results/experiments/new_benchmarks');OUT=ROOT/'tau2_learned_selector'

def main():
 OUT.mkdir(exist_ok=True);folder=ROOT/'tau2_training_branches_uniform';rows=[];groups=[]
 for plan_path in sorted(folder.glob('*_plan.json')):
  plan=json.loads(plan_path.read_text());assert plan['official_split']=='train'
  for point in plan['points']:
   group={'domain':plan['domain'],'task_id':plan['task_id'],'decision':point['decision'],'fit_split':plan['fit_split'],'record':point['record'],'weight':point['sampling_weight'],'labels':{}}
   for choice in point['choices']:
    p=folder/f"{plan['domain']}_{plan['task_id']}_d{point['decision']}_c{choice}.json"
    if not p.exists():continue
    d=json.loads(p.read_text())
    if d.get('status')!='completed' or d.get('simulation',{}).get('termination_reason')=='timeout':continue
    label=int(d['reward']['reward'])-int(plan['raw_reward']);group['labels'][choice]=label
    rows.append({'domain':plan['domain'],'task_id':plan['task_id'],'decision':point['decision'],'fit_split':plan['fit_split'],'label':label,'weight':point['sampling_weight']/len(point['choices']),'record':{**point['record'],'alternative_index':choice},'source':p.name})
   groups.append(group)
 train=[r for r in rows if r['fit_split']=='train'];valid=[g for g in groups if g['fit_split']=='valid']
 if not train or not valid:raise RuntimeError('Need both train and validation task groups; collect more branches')
 assert not {(r['domain'],r['task_id']) for r in train}&{(r['domain'],r['task_id']) for r in valid}
 vector=DictVectorizer();x=vector.fit_transform([features(r['record']) for r in train]);y=np.array([r['label'] for r in train]);w=np.array([r['weight'] for r in train]);w=w/w.mean()
 best=None;search=[]
 for family,param in [("logistic",c) for c in [.1,1,10]]+[("tree",depth) for depth in [2,3,4]]:
  c=param if family=="logistic" else None
  if len(set(y))==1:artifact={'constant':int(y[0])}
  else:
   model=LogisticRegression(C=c,max_iter=1000,solver='lbfgs') if family=='logistic' else DecisionTreeClassifier(max_depth=param,min_samples_leaf=8,random_state=0)
   model.fit(x,y,sample_weight=w)
   artifact={'constant':None,'vocabulary':vector.vocabulary_,'classes':model.classes_.tolist()}
   if family=='logistic':artifact.update(coef=model.coef_.tolist(),intercept=model.intercept_.tolist())
   else:
    t=model.tree_;artifact['tree']={'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,:].tolist()}
   for row,expected in zip(train[:20],model.predict_proba(x[:20])):
    actual=probabilities(artifact,row['record']);assert np.allclose([actual.get(int(k),0) for k in model.classes_],expected)
  for risk in [1,3,10]:
   predictions=[]
   for group in valid:
    options=[]
    for j in range(1,len(group['record']['candidates'])):
     p=probabilities(artifact,{**group['record'],'alternative_index':j});options.append((p.get(1,0)-risk*p.get(-1,0),-j))
    margin,negative_choice=max(options);predictions.append((margin,-negative_choice))
   margins=np.array([m for m,j in predictions])
   for threshold in np.unique(np.r_[0,margins[margins>=0],1.01]):
    gain=harm=unknown=0.;count=0
    for group,(margin,choice) in zip(valid,predictions):
     if margin<=threshold:continue
     count+=1;label=group['labels'].get(choice)
     if label is None:unknown+=group['weight']
     elif label>0:gain+=group['weight']
     elif label<0:harm+=group['weight']
    if harm>0 or unknown>0:continue
    value=(gain,-count)
    if best is None or value>best[0]:best=(value,{**artifact,'risk_penalty':risk,'threshold':float(threshold),'C':c,'model_family':family,'parameter':param},{'weighted_rescues':gain,'weighted_harms':harm,'unresolved_interventions':unknown,'interventions':count})
  search.append({'model_family':family,'parameter':param,'best_valid':best[2]})
 artifact=best[1];artifact.update(name='learned_one_shot_pair',intervention_budget=1,protocol='Official training tasks only, hash-split by task80/20. Uniform hash-sampled prefixes:3 on failed tasks,1 on successful; all alternative candidates verified independently. Inverse prefix inclusion weighting. Gate maximizes weighted validation rescues with zero observed harms and zero unresolved interventions. Labels are budgeted task outcome differences, not listwise dominance.',train_pairs=len(train),valid_prefixes=len(valid),train_labels={str(i):int((y==i).sum()) for i in [-1,0,1]},valid_outcome=best[2],source_files=[r['source'] for r in rows],feature_code_sha256=hashlib.sha256(Path('scripts/tau2_pair_selector.py').read_bytes()).hexdigest())
 artifact['source_sha256']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in artifact['source_files']}
 artifact['plan_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*_plan.json'))}
 artifact['training_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (OUT/'selector.json').write_text(json.dumps(artifact,indent=2));(OUT/'validation_search.json').write_text(json.dumps(search,indent=2));print(json.dumps({k:artifact[k] for k in ['train_pairs','valid_prefixes','train_labels','valid_outcome','model_family','parameter','C','risk_penalty','threshold']}),flush=True)
if __name__=='__main__':main()
