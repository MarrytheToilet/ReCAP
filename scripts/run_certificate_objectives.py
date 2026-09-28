"""Frozen-support objectives and validation-gated natural-distribution evaluation.

All-cost targets require extra candidate verifications. Null costs are censored;
training comparisons use only finite costs, never assumed failures.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from recap.models.policy_reranker import CandidatePolicy, candidate_matrix
from recap.models.reranker_dataset import FEATURE_NAMES

BASE='analysis/recap_xhard_700_mimo25_t1_top5_'

def read(path): return [json.loads(l) for l in open(path) if l.strip()]
def key(r): return r['task_id'],int(r.get('seed',0)),int(r['step_index'])

def tensors(rows):
    width=max(len(r['candidates']) for r in rows)
    x=torch.zeros(len(rows),width,len(FEATURE_NAMES)); mask=torch.zeros(len(rows),width,dtype=torch.bool)
    costs=torch.full((len(rows),width),float('inf')); pos=torch.zeros(len(rows),dtype=torch.long); neg=pos.clone()
    for i,r in enumerate(rows):
        c,m=candidate_matrix(r); x[i,:len(c)]=torch.from_numpy(m);mask[i,:len(c)]=True
        for j,a in enumerate(c):
            v=r['candidate_costs'].get(a)
            if v is not None: costs[i,j]=float(v)
        pos[i]=c.index(r.get('preferred_action',c[0]));neg[i]=c.index(r.get('executed_action',r['rejected_action']))
    return dict(x=x,mask=mask,costs=costs,pos=pos,neg=neg)

def objective(logits,b,kind):
    z=logits.masked_fill(~b['mask'],-1e9)
    if kind=='listwise': return F.cross_entropy(z,b['pos'])
    if kind=='certified_pair':
        i=torch.arange(len(z))
        pc=b['costs'][i,b['pos']]; nc=b['costs'][i,b['neg']]
        strict=torch.isfinite(pc) & torch.isfinite(nc) & (pc<nc)
        if not strict.any(): return z.sum()*0
        return F.softplus(z[i,b['neg']]-z[i,b['pos']])[strict].mean()
    finite=torch.isfinite(b['costs']) & b['mask']
    if kind=='minima':
        minima=finite & (b['costs']==b['costs'].min(1,keepdim=True).values)
        # Uniform target over all certified minima, with no emitted-label boost.
        target=minima.float()/minima.sum(1,keepdim=True).clamp_min(1)
        usable=minima.any(1)
        # Unknown suffix costs are censored, not certified negatives.
        observed_logits=z.masked_fill(~finite,-1e9)
        return (-(target*F.log_softmax(observed_logits,dim=1)).sum(1))[usable].mean()
    if kind=='cost_pairs':
        pairs=(b['costs'][:,:,None]<b['costs'][:,None,:]) & finite[:,:,None] & finite[:,None,:]
        terms=F.softplus(z[:,None,:]-z[:,:,None])
        count=pairs.sum((1,2));usable=count>0
        if not usable.any(): return z.sum()*0
        return ((terms*pairs).sum((1,2))/count.clamp_min(1))[usable].mean()
    raise ValueError(kind)

def metrics(rows,scores,threshold=0.0):
    outcomes=[];label_rr=[];label_top=[]
    for r,s in zip(rows,scores):
        c=r['candidates'];raw=c.index(r['executed_action']);j=int(np.argmax(s[:len(c)]))
        if s[j]-s[raw]<threshold: j=raw
        rc=r['candidate_costs'].get(c[raw]);sc=r['candidate_costs'].get(c[j]);finite=[v for v in r['candidate_costs'].values() if v is not None]
        known=rc is not None and sc is not None
        good=known and sc<rc; bad=known and sc>rc; intervention=j!=raw
        opportunity=rc is not None and bool(finite) and min(finite)<rc
        outcomes.append(dict(task_id=r['task_id'],step_index=r['step_index'],selected=c[j],intervention=intervention,
            improved=bool(good),harmed=bool(bad),comparable=known,opportunity=opportunity,
            gain=rc-sc if known else None,min_cost=bool(sc is not None and finite and sc==min(finite)),
            regret=sc-min(finite) if sc is not None and finite else None))
        if 'preferred_action' in r:
            order=sorted(range(len(c)),key=lambda k:(-s[k],k));rank=order.index(c.index(r['preferred_action']))+1
            label_rr.append(1/rank);label_top.append(rank==1)
    n=len(rows); ni=sum(o['intervention'] for o in outcomes);good=sum(o['improved'] for o in outcomes);bad=sum(o['harmed'] for o in outcomes)
    opp=sum(o['opportunity'] for o in outcomes);comparable=sum(o['comparable'] for o in outcomes)
    gains=[o['gain'] for o in outcomes if o['gain'] is not None];regrets=[o['regret'] for o in outcomes if o['regret'] is not None]
    m=dict(n=n,interventions=ni,improved=good,harmed=bad,net_improvements=good-bad,
        intervention_precision=good/ni if ni else None,repair_recall=good/opp if opp else None,
        opportunities=opp,comparable=comparable,unresolved_interventions=sum(o['intervention'] and not o['comparable'] for o in outcomes),
        harm_rate=bad/n,min_cost_top1=sum(o['min_cost'] for o in outcomes)/n,
        mean_regret=float(np.mean(regrets)) if regrets else None,mean_cost_gain=float(np.mean(gains)) if gains else None,
        label_mrr=float(np.mean(label_rr)) if label_rr else None,label_top1=float(np.mean(label_top)) if label_top else None)
    return m,outcomes

def select_gate(rows,scores):
    margins=[]
    for r,s in zip(rows,scores):
        raw=r['candidates'].index(r['executed_action']);margins.append(float(max(s[:len(r['candidates'])])-s[raw]))
    thresholds=sorted(set([0.0]+margins+[max(margins)+1]))
    options=[]
    for t in thresholds:
        m,_=metrics(rows,scores,t)
        # Fixed 1% validation harm budget; no test information used.
        if m['harm_rate']<=.01 and m['unresolved_interventions']==0:
            options.append((m['net_improvements'],-m['harmed'],-m['interventions'],t))
    return max(options)[-1]

def score(model,b):
    model.eval()
    with torch.no_grad():return model(b['x']).masked_fill(~b['mask'],-1e9).numpy()

def bootstrap(outcomes,seed=42):
    groups={}
    for o in outcomes:groups.setdefault(o['task_id'],[]).append(o)
    stats=np.array([[len(g),sum(x['improved']-x['harmed'] for x in g)] for g in groups.values()])
    rng=np.random.default_rng(seed);ix=rng.integers(len(stats),size=(2000,len(stats)));s=stats[ix].sum(1)
    return np.quantile(s[:,1]/s[:,0],[.025,.975]).tolist()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',default='results/experiments/objectives')
    p.add_argument('--natural-dir');p.add_argument('--epochs',type=int,default=400);p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--natural-training',action='store_true');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    costs={key(r):dict(zip(r['successful_actions'],r['successful_suffix_lens'])) for r in read(BASE+'repair_multiplicity_rows.jsonl')}
    splits={}
    for split in ('train','valid','test'):
        splits[split]=[]
        for r in read(BASE+'splits_t30/'+split+'.jsonl'):
            r['candidate_costs']=costs[key(r)]
            r['executed_action']=r['rejected_action']
            splits[split].append(r)
    natural={s:read(Path(a.natural_dir)/(s+'.jsonl')) for s in splits} if a.natural_dir else None
    for s,t in [('train','valid'),('train','test'),('valid','test')]:
        assert not ({r['task_id'] for r in splits[s]} & {r['task_id'] for r in splits[t]})
        if natural:assert not ({r['task_id'] for r in natural[s]} & {r['task_id'] for r in natural[t]})
    if a.natural_training:
        assert natural
        fit={s:[r for r in natural[s] if any(v is not None for v in r['candidate_costs'].values())] for s in ('train','valid')}
        kinds=('minima','cost_pairs')
    else:fit=splits;kinds=('listwise','certified_pair','minima','cost_pairs')
    train=tensors(fit['train']);valid=tensors(fit['valid']);test=tensors(splits['test'])
    nb={s:tensors(natural[s]) for s in ('valid','test')} if natural else None
    results={};protocol=dict(args=vars(a),counts={s:len(fit[s]) for s in ('train','valid')},test_labels=len(splits['test']),
        features=list(FEATURE_NAMES),gate='validation net improvement subject to <=1% harm, zero unresolved intervention',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        supervision='all-candidate finite verifier costs for minima/cost_pairs; emitted label for listwise/certified_pair')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for seed in map(int,a.seeds.split(',')):
        for kind in kinds:
            name=f'{kind}_seed{seed}';torch.manual_seed(seed);model=CandidatePolicy(len(FEATURE_NAMES),128,2,.05)
            opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.0001);best=float('inf');beststate=None;bestepoch=None
            for epoch in range(a.epochs):
                model.train();z=model(train['x']);loss=objective(z,train,kind)
                probs=torch.softmax(z.masked_fill(~train['mask'],-1e9),1)
                loss=loss+.005*(probs*probs.clamp_min(1e-8).log()).sum(1).mean()
                opt.zero_grad();loss.backward();opt.step();model.eval()
                with torch.no_grad():v=float(objective(model(valid['x']),valid,kind))
                if v<best:best=v;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch+1
            model.load_state_dict(beststate)
            payload=dict(state_dict=beststate,feature_names=FEATURE_NAMES,hyperparameters=dict(hidden_dim=128,num_layers=2,dropout=.05,objective=kind,seed=seed))
            torch.save(payload,out/(name+'.pt'))
            m,rows=metrics(splits['test'],score(model,test));result=dict(positive_test=m,best_epoch=bestepoch)
            if natural:
                vs=score(model,nb['valid']);ts=score(model,nb['test']);t=select_gate(natural['valid'],vs)
                for tag,threshold in [('ungated',0.),('gated',t)]:
                    nm,detail=metrics(natural['test'],ts,threshold);nm['net_improvement_rate_ci95']=bootstrap(detail)
                    result[tag]=nm
                    (out/(name+'_'+tag+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in detail))
                result['threshold']=t
            results[name]=result;(out/'results.json').write_text(json.dumps(results,indent=2));print(name,json.dumps(result),flush=True)
    if natural:
        for name,rank in [('raw',0),('rank2',1)]:
            s=np.full(nb['test']['mask'].shape,-1.)
            for i,r in enumerate(natural['test']):s[i,min(rank,len(r['candidates'])-1)]=1
            m,detail=metrics(natural['test'],s);m['net_improvement_rate_ci95']=bootstrap(detail);results[name]=m
        (out/'results.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
