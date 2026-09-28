"""Public browser affordance features; no seed, task ID, reward or oracle state."""
import ast,re,collections

def tokens(x):return set(re.findall(r'[a-z0-9]+',str(x).lower()))
def action_features(state,action):
 try:
  node=ast.parse(action,mode='eval').body;fn=node.func.id;args=[ast.literal_eval(x) for x in node.args]
 except Exception:return {'invalid':1.}
 goal=tokens(state['goal']);elements={str(e['bid']):e for e in state.get('visible_dom_elements',[])};bid=str(args[0]) if args else '';target=elements.get(bid,{})
 label=tokens(target.get('text',''));argument=tokens(args[1] if len(args)>1 else '')
 previous=state.get('previous_actions',[]);quotes=re.findall(r'"([^"]+)"',state['goal'])
 f={'fn='+fn:1.,'tag='+target.get('tag','unlisted'):1.,'readonly':float(bool(target.get('readonly'))),'disabled':float(bool(target.get('disabled'))),'fill_readonly':float(fn=='fill' and bool(target.get('readonly'))),'missing_target':float(fn!='scroll' and not target),'repeat':min(previous.count(action),5)/5,'last_same':float(bool(previous) and action==previous[-1]),'repeat_after_error':float(bool(previous) and action==previous[-1] and bool(state.get('last_action_error'))),'goal_label_overlap':len(goal&label)/max(len(label),1),'goal_arg_overlap':len(goal&argument)/max(len(argument),1),'arg_in_goal':float(bool(len(args)>1 and str(args[1]).lower() in state['goal'].lower())),'quoted_target_match':float(any(q.lower()==str(target.get('text','')).strip().lower() for q in quotes)),'has_value':float(bool(target.get('value'))),'already_filled':float(fn=='fill' and len(args)>1 and str(args[1])==str(target.get('value',''))),'label_length':min(len(str(target.get('text',''))),240)/240,'progress':min(len(previous),12)/12,'prior_error':float(bool(state.get('last_action_error')))}
 return f

def features(state,raw,alt):
 a=action_features(state,raw);b=action_features(state,alt);return {'delta_'+k:b.get(k,0)-a.get(k,0) for k in set(a)|set(b)}

def margin(artifact,state,raw,alt):
 import math
 f=features(state,raw,alt);vocab=artifact['vocabulary'];x={vocab[k]:v for k,v in f.items() if k in vocab}
 if artifact.get('constant') is not None:return float(artifact['constant'])
 logits=[bias+sum(row[j]*v for j,v in x.items()) for row,bias in zip(artifact['coef'],artifact['intercept'])];classes=artifact['classes']
 if len(classes)==2:
  p=1/(1+math.exp(-max(-700,min(700,logits[0]))));prob=dict(zip(classes,[1-p,p]))
 else:
  largest=max(logits);ex=[math.exp(x-largest) for x in logits];prob=dict(zip(classes,[x/sum(ex) for x in ex]))
 return prob.get(1,0)-3*prob.get(-1,0)

def select(artifact,state,candidates):
 j=0;best=artifact['threshold']
 for i in range(1,len(candidates)):
  score=margin(artifact,state,candidates[0],candidates[i])
  if score>best:best=score;j=i
 return j
