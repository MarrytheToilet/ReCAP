"""Fixed-regularization pair classifier from replay outcomes; no held-out task reads."""
import json,hashlib,collections
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from scripts.browser_pair_features import features,margin
ROOT=Path('results/benchmarks/miniwob');folder=ROOT/'branches'
plans=list(folder.glob('*_summary.json'));assert len(plans)==120 and all(json.loads(p.read_text())['complete'] for p in plans),'Wait for all120 sampled task ledgers'
rows=[];ledger=collections.Counter()
for p in sorted(folder.glob('*_d*_c*.json')):
 d=json.loads(p.read_text());ledger[d['status']]+=1
 if d['status']!='completed':continue
 src=Path(d['raw_source']);assert hashlib.sha256(src.read_bytes()).hexdigest()==d['raw_sha256'];raw=json.loads(src.read_text());record=raw['records'][d['decision']];delta=d['reward']-raw['reward']
 cost_label=int(delta)
 if not delta and d['reward']:
  cost_label=int(d['alternative_suffix_steps']<d['raw_suffix_steps'])-int(d['alternative_suffix_steps']>d['raw_suffix_steps'])
 rows.append({'fit_split':d['fit_split'],'task':d['task'],'seed':d['seed'],'features':features(record['state'],record['candidates'][0],record['candidates'][d['choice']]),'label':cost_label,'terminal_delta':delta,'weight':d['sampling_weight']/max(len(record['candidates'])-1,1),'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
train=[r for r in rows if r['fit_split']=='fit'];assert train
vector=DictVectorizer();x=vector.fit_transform([r['features'] for r in train]);y=np.array([r['label'] for r in train]);w=np.array([r['weight'] for r in train]);w/=w.mean()
if len(set(y))==1:artifact={'constant':int(y[0])}
else:
 model=LogisticRegression(C=.3,max_iter=2000,solver='lbfgs');model.fit(x,y,sample_weight=w);artifact={'constant':None,'vocabulary':vector.vocabulary_,'classes':model.classes_.tolist(),'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist()}
if artifact.get('constant') is None:
 for r,expected in zip(train[:20],model.predict_proba(x[:20])):
  branch=json.loads(Path(r['source']).read_text());raw=json.loads(Path(branch['raw_source']).read_text());rec=raw['records'][branch['decision']]
  expected=dict(zip(model.classes_,expected));actual=margin(artifact,rec['state'],rec['candidates'][0],rec['candidates'][branch['choice']])
  assert abs(actual-(expected.get(1,0)-3*expected.get(-1,0)))<1e-8
artifact.update(threshold=.1,objective='Pairwise cost-order classification; task failure is worse than success; among successful suffixes fewer actions preferred; unsuccessful/unsuccessful is tie. No listwise dominance labels.',C=.3,risk_penalty=3,train_seeds=list(range(20)),validation_seeds=list(range(100,110)),train_pairs=len(train),labels=dict(collections.Counter(map(int,y))),feature_code_sha256=hashlib.sha256(Path('scripts/browser_pair_features.py').read_bytes()).hexdigest(),source_sha256={r['source']:r['source_sha256'] for r in rows})
(ROOT/'trained_selector.json').write_text(json.dumps(artifact,indent=2));(ROOT/'training_pairs.json').write_text(json.dumps({'rows':rows,'ledger':dict(ledger)},indent=2))
print(json.dumps({k:artifact[k] for k in ['train_pairs','labels','threshold','C']}))
