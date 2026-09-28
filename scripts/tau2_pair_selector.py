"""Portable pair selector: public logged history/candidates in, one logged candidate out."""
import json,math,re
from collections import Counter
READ_PREFIX=('get_','find_','list_','search_','calculate','check_')
def canonical(action):return json.dumps(action,sort_keys=True,ensure_ascii=False)
def features(record):
 c=record['candidates'];alternative_index=record.get('alternative_index',min(1,len(c)-1));raw=c[0] if isinstance(c[0],dict) else {'kind':'invalid'};second=c[alternative_index];alt=second if isinstance(second,dict) else {'kind':'invalid'};history=record.get('history',[]);past=[];users=[]
 for h in history:
  if h.get('role')=='assistant':
   for t in h.get('tool_calls',[]):past.append({'kind':'tool','name':t['name'],'arguments':t['arguments']})
  if h.get('role')=='user':users.append(h.get('content') or '')
 counts=Counter(map(canonical,past));f={}
 user_tokens=set(re.findall(r'[a-z0-9]+',' '.join(users)[-4000:].lower()))
 tool_text=' '.join(str(h.get('content') or '') for h in history if h.get('role')=='tool')[-4000:]
 tool_tokens=set(re.findall(r'[a-z0-9]+',tool_text.lower()))
 for index,(name,a) in enumerate([('raw',raw),('alt',alt)]):
  tool=a.get('kind')=='tool';fn=str(a.get('name') or 'message');content=str(a.get('content') or '')
  f[name+'_tool']=float(tool);f[name+'_fn='+fn]=1.
  action_tokens=set(re.findall(r'[a-z0-9]+',json.dumps(a.get('arguments',{}),sort_keys=True).lower() if tool else content.lower()))
  fn_tokens=set(fn.split('_'))-{'get','find','by','id','details','all'}
  f[name+'_user_overlap']=len(action_tokens&user_tokens)/max(len(action_tokens),1)
  f[name+'_observed_overlap']=len(action_tokens&tool_tokens)/max(len(action_tokens),1)
  f[name+'_intent_overlap']=len(fn_tokens&user_tokens)/max(len(fn_tokens),1)
  errors=record.get('candidate_errors')
  error_index=0 if index==0 else alternative_index
  f[name+'_invalid']=float(bool(errors and errors[min(error_index,len(errors)-1)]))
  f[name+'_read']=float(tool and fn.startswith(READ_PREFIX));f[name+'_repeat']=min(counts[canonical(a)],5)/5
  f[name+'_length']=min(len(canonical(a)),3000)/3000
  f[name+'_confirmation']=float(bool(re.search(r'\bconfirm|\byes\b|permission',content,re.I)))
  f[name+'_finish']=float(bool(re.search(r'completed|processed|done|anything else',content,re.I)))
 f['same_kind']=float(raw.get('kind')==alt.get('kind'));f['same_fn']=float(raw.get('name')==alt.get('name'))
 f['same_arguments']=float(raw.get('arguments')==alt.get('arguments'))
 f['n_tools']=min(len(past),40)/40;f['n_users']=min(len(users),20)/20
 f['n_candidates']=len(c)/3
 f['alt_rank_reciprocal']=1/(alternative_index+1)
 f['user_confirmed']=float(bool(users and re.search(r'\byes\b|confirm|proceed|go ahead',users[-1],re.I)))
 f['repeat_delta']=f['alt_repeat']-f['raw_repeat'];f['read_delta']=f['alt_read']-f['raw_read']
 return f

def probabilities(artifact,record):
 if artifact.get('constant') is not None:return {int(artifact['constant']):1.}
 f=features(record);vocab=artifact['vocabulary'];x={vocab[k]:v for k,v in f.items() if k in vocab}
 if 'tree' in artifact:
  tree=artifact['tree'];node=0
  while tree['left'][node]!=-1:
   node=tree['left'][node] if x.get(tree['feature'][node],0)<=tree['threshold'][node] else tree['right'][node]
  values=tree['value'][node];total=sum(values)
  return {c:v/total for c,v in zip(artifact['classes'],values)}
 logits=[b+sum(row[j]*v for j,v in x.items()) for row,b in zip(artifact['coef'],artifact['intercept'])]
 classes=artifact['classes']
 if len(classes)==2:
  p=1/(1+math.exp(-max(-700,min(700,logits[0]))));return {classes[0]:1-p,classes[1]:p}
 m=max(logits);e=[math.exp(z-m) for z in logits];return {c:p/sum(e) for c,p in zip(classes,e)}

def select(artifact,record):
 if len(record['candidates'])<2:return 0
 chosen=0;best=artifact['threshold']
 for j in range(1,len(record['candidates'])):
  p=probabilities(artifact,{**record,'alternative_index':j});margin=p.get(1,0)-artifact['risk_penalty']*p.get(-1,0)
  if margin>best:chosen=j;best=margin
 return chosen
