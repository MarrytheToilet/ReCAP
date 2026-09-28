"""Train a default-preserving pair classifier on natural decisions, select on validation."""
import json, pickle, hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from recap.models.policy_reranker import candidate_matrix
from scripts.run_certificate_objectives import read, metrics, bootstrap

ROOT=Path('results/experiments');OUT=ROOT/'selective_pairs'

def features(rows):
    values=[];locations=[];labels=[];weights=[]
    for i,r in enumerate(rows):
        safe={**r,'rejected_action':r['executed_action']}
        candidates,m=candidate_matrix(safe);raw=candidates.index(r['executed_action']);rc=r['candidate_costs'].get(candidates[raw])
        known=sum(r['candidate_costs'].get(a) is not None for j,a in enumerate(candidates) if j!=raw)
        for j,a in enumerate(candidates):
            if j==raw:continue
            values.append(np.r_[m[j],m[raw],m[j]-m[raw],min(r['step_index'],100)/30])
            locations.append((i,j));c=r['candidate_costs'].get(a)
            labels.append(-1 if rc is None or c is None else (2 if c<rc else 0 if c>rc else 1))
            weights.append(1/max(known,1))
    return np.asarray(values),locations,np.asarray(labels),np.asarray(weights)

def scores(rows,locations,prob,penalty):
    width=max(len(r['candidates']) for r in rows);out=np.full((len(rows),width),-1e9)
    for i,r in enumerate(rows):out[i,r['candidates'].index(r['executed_action'])]=0
    for (i,j),pr in zip(locations,prob):out[i,j]=pr[2]-penalty*pr[0]
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows={s:read(ROOT/'natural'/f'{s}.jsonl') for s in ['train','valid']}
    assert not {r['task_id'] for r in rows['train']} & {r['task_id'] for r in rows['valid']}
    x,loc,y,w=features(rows['train']);vx,vl,_,_=features(rows['valid']);known=y>=0
    protocol={'features':'candidate + executed features + difference, clipped step index; rejected_action always actual executed action','targets':'finite certified cost: better/tie/worse versus executed; unknown censored','models':{'max_leaf_nodes':[7,15,31],'l2_regularization':[1,10],'max_iter':150,'learning_rate':.05,'min_samples_leaf':30},'gate':'validation net correction, <=1% harm and zero unresolved interventions','risk_penalty':[1,2,5],'known_train_pairs':int(known.sum()),'training_class_counts':np.bincount(y[known],minlength=3).tolist(),'additional_verifier_information':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    best=None;leaderboard=[]
    for leaves in [7,15,31]:
        for l2 in [1,10]:
            model=HistGradientBoostingClassifier(max_leaf_nodes=leaves,l2_regularization=l2,max_iter=150,learning_rate=.05,min_samples_leaf=30,early_stopping=False,random_state=0)
            model.fit(x[known],y[known],sample_weight=w[known]);assert list(model.classes_)==[0,1,2]
            prob=model.predict_proba(vx)
            for penalty in [1,2,5]:
                z=scores(rows['valid'],vl,prob,penalty)
                thresholds=np.unique(np.r_[0,np.quantile(np.maximum(z.max(1),0),np.linspace(0,1,101)),1.01])
                for threshold in thresholds:
                    m,_=metrics(rows['valid'],z,float(threshold))
                    if m['harm_rate']>.01 or m['unresolved_interventions']:continue
                    value=(m['net_improvements'],-m['harmed'],-m['interventions'])
                    if best is None or value>best[0]:best=(value,model,penalty,float(threshold),{'leaves':leaves,'l2':l2},m)
                leaderboard.append({'leaves':leaves,'l2':l2,'penalty':penalty,'best_validation_so_far':best[5]})
            print('fit',leaves,l2,'best',best[0],flush=True)
    _,model,penalty,threshold,hyper,valid=best
    with (OUT/'model.pkl').open('wb') as f:pickle.dump({'model':model,'penalty':penalty,'threshold':threshold,'hyperparameters':hyper},f)
    (OUT/'validation_search.json').write_text(json.dumps(leaderboard,indent=2))
    # Load test only after model, objective and intervention threshold are frozen.
    test=read(ROOT/'natural/test.jsonl');assert not {r['task_id'] for r in test}&{r['task_id'] for split in rows.values() for r in split}
    tx,tl,_,_=features(test);z=scores(test,tl,model.predict_proba(tx),penalty)
    result={'hyperparameters':hyper,'penalty':penalty,'threshold':threshold,'valid':valid}
    for name,t in [('gated',threshold),('ungated',0.)]:
        m,details=metrics(test,z,t);m['net_improvement_rate_ci95']=bootstrap(details);result[name]=m
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
