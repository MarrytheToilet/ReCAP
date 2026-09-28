"""State-aware selective reranking: public observations plus frozen text embeddings."""
import json,pickle,hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scripts.train_selective_pairs import features,scores
from scripts.run_certificate_objectives import read,metrics,bootstrap
from recap.models.lm_candidate_policy import compact_text
from recap.agents.lm_policy_agent import objective_action_overlap,is_inverse_navigation,is_semantic_undo,pool_priority
ROOT=Path('results/experiments');OUT=ROOT/'context_pairs'

def extra_features(rows,loc):
    out=[]
    def feat(r,a):
        history=tuple(r['history']);obs=r['observation'];initial=r['initial_observation'];past=r['observation_history']
        return [objective_action_overlap(a,initial),objective_action_overlap(a,obs),pool_priority(a,history,initial),float(bool(history) and is_inverse_navigation(history[-1],a)),float(any(is_semantic_undo(h,a) for h in history[-3:])),float(a in r['admissible_actions']),float(len(past)>0 and obs==past[-1])]
    for i,j in loc:
        r=rows[i];a=np.array(feat(r,r['candidates'][j]));raw=np.array(feat(r,r['executed_action']));out.append(np.r_[a,raw,a-raw])
    return np.asarray(out)

def embeddings(rows,split,encoder):
    texts=[];locations=[]
    for i,r in enumerate(rows):
        initial=compact_text(r['initial_observation'].split('-=')[0],1200);obs=compact_text(r['observation'],650);history=' | '.join(r['history'][-4:])
        for j,a in enumerate(r['candidates']):
            texts.append(f'Candidate action: {a}\nOriginally executed: {r["executed_action"]}\nTask: {initial}\nRecent actions: {history}\nState: {obs}');locations.append((i,j))
    digest=hashlib.sha256(json.dumps(texts).encode()).hexdigest();path=OUT/f'{split}_{digest[:12]}.npz'
    if path.exists():z=np.load(path)['embeddings']
    else:
        z=encoder.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False);np.savez_compressed(path,embeddings=z)
    return {key:row for key,row in zip(locations,z)}

def prepare(rows,split,encoder):
    x,loc,y,w=features(rows);extras=extra_features(rows,loc);emb=embeddings(rows,split,encoder);em=[]
    for i,j in loc:
        raw=rows[i]['candidates'].index(rows[i]['executed_action']);em.append(np.r_[emb[(i,j)],emb[(i,j)]-emb[(i,raw)]])
    return np.c_[x,extras],np.c_[x,extras,np.asarray(em)],loc,y,w

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(6)
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',local_files_only=True);encoder.max_seq_length=512
    rows={s:read(ROOT/'natural_context'/f'{s}.jsonl') for s in ['train','valid']}
    tr,te,_,y,w=prepare(rows['train'],'train',encoder);vr,ve,vl,_,_=prepare(rows['valid'],'valid',encoder);known=y>=0
    protocol={'model':'all-MiniLM-L6-v2 frozen CPU embeddings','inputs':'Public initial objective, current observation, last 4 actions, candidate text; raw/candidate structural differences. No facts, policy commands, costs or task IDs in features.','supervision':'Finite verifier-cost comparisons on train only; censored pairs excluded.','max_seq_length':512,'selection':'validation net gain under <=1% harm and zero unresolved; 101 quantile gates','limitations':'Extra observed context and all-candidate cost supervision; not an objective-only ablation'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));best=None;search=[]
    configurations=[('structured_tree',l) for l in [7,15,31]]+[('embedding_logistic',c) for c in [.01,.1,1,10]]
    for family,param in configurations:
        if family=='structured_tree':
            model=HistGradientBoostingClassifier(max_leaf_nodes=param,max_iter=200,min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=0);x=tr;vx=vr;model.fit(x[known],y[known],sample_weight=w[known])
        else:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=param,max_iter=600,solver='lbfgs'));x=te;vx=ve;model.fit(x[known],y[known],logisticregression__sample_weight=w[known])
        prob=model.predict_proba(vx);local=None
        for penalty in [1,2,5]:
            z=scores(rows['valid'],vl,prob,penalty)
            for threshold in np.unique(np.r_[0,np.quantile(np.maximum(z.max(1),0),np.linspace(0,1,101)),1.01]):
                m,_=metrics(rows['valid'],z,float(threshold))
                if m['harm_rate']>.01 or m['unresolved_interventions']:continue
                value=(m['net_improvements'],-m['harmed'],-m['interventions'])
                if local is None or value>local[0]:local=(value,penalty,float(threshold),m)
                if best is None or value>best[0]:best=(value,model,family,param,penalty,float(threshold),m)
        search.append({'family':family,'parameter':param,'validation':local[3],'penalty':local[1],'threshold':local[2]});print(json.dumps(search[-1]),flush=True)
        (OUT/'validation_search.json').write_text(json.dumps(search,indent=2))
    _,model,family,param,penalty,threshold,valid=best
    with (OUT/'model.pkl').open('wb') as f:pickle.dump({'model':model,'family':family,'parameter':param,'penalty':penalty,'threshold':threshold},f)
    test=read(ROOT/'natural_context/test.jsonl');xr,xe,tl,_,_=prepare(test,'test',encoder);z=scores(test,tl,model.predict_proba(xr if family=='structured_tree' else xe),penalty)
    result={'family':family,'parameter':param,'penalty':penalty,'threshold':threshold,'validation':valid}
    for name,t in [('gated',threshold),('ungated',0.)]:
        m,details=metrics(test,z,t);m['net_improvement_rate_ci95']=bootstrap(details);result[name]=m;(OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
