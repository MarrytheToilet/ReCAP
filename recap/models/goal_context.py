"""Preserve the public task objective separately from the current observation."""
from typing import Mapping,Any
from recap.models.lm_candidate_policy import CandidateBatch,compact_text

def goal_context(record:Mapping[str,Any],max_history=12,max_observation_chars=360):
 # Room descriptions follow this public TextWorld delimiter; no game metadata is read.
 initial=str(record.get('initial_observation','')).split('-=')[0]
 objective=compact_text(initial,1600)
 observation=compact_text(str(record.get('observation','')),max_observation_chars)
 history=[str(a) for a in record.get('history',())][-max_history:]
 lines=['You are reranking logged actions for a text-game agent.','Choose only among the logged candidates.']
 if objective:lines.append('Task/state excerpt: '+objective)
 if observation:lines.append('Current observation: '+observation)
 lines.append('Recent actions: '+(' | '.join(history) if history else 'none'))
 candidates=[str(a) for a in record.get('candidates',())]
 if candidates and len(candidates)<=12:
  lines.append('Logged candidates:');lines.extend(f'{i+1}. {a}' for i,a in enumerate(candidates))
 elif candidates:lines.append(f'Logged candidate pool size: {len(candidates)}')
 return '\n'.join(lines)

def encode_goal_batch(tokenizer,record,max_length=1024,max_history=12,max_observation_chars=360,device=None):
 candidates=tuple(str(a) for a in record.get('candidates',()));context=goal_context(record,max_history,max_observation_chars);ids=[]
 for action in candidates:
  suffix=f'\n\nCandidate action: {action}\nShould this candidate be ranked first for the current decision? Answer yes or no.\nAnswer:'
  full=tokenizer.encode(context+suffix,add_special_tokens=True)
  if len(full)>max_length:
   ending=tokenizer.encode(suffix,add_special_tokens=False)
   if len(ending)>=max_length:raise ValueError('Candidate suffix alone exceeds token budget')
   full=tokenizer.encode(context,add_special_tokens=True)[:max_length-len(ending)]+ending
  ids.append(full)
 encoded=tokenizer.pad({'input_ids':ids},padding=True,return_tensors='pt')
 if device is not None:encoded={k:v.to(device) for k,v in encoded.items()}
 return CandidateBatch(candidates=candidates,input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'])
