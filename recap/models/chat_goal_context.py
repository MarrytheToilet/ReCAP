"""Goal-preserving input using the base tokenizer's chat template and no thinking."""
from recap.models.goal_context import goal_context
from recap.models.lm_candidate_policy import CandidateBatch

def encode_chat_goal_batch(tokenizer,record,max_length=1024,max_history=12,max_observation_chars=360,device=None):
 candidates=tuple(str(a) for a in record.get('candidates',()));context=goal_context(record,max_history,max_observation_chars);ids=[]
 for action in candidates:
  suffix=f'\n\nCandidate action: {action}\nShould this candidate be ranked first for the current decision? Answer yes or no.'
  text=tokenizer.apply_chat_template([{'role':'user','content':context+suffix}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
  full=tokenizer.encode(text,add_special_tokens=False)
  if len(full)>max_length:
   position=text.rfind('\n\nCandidate action:');assert position>=0
   ending=tokenizer.encode(text[position:],add_special_tokens=False)
   if len(ending)>=max_length:raise ValueError('Candidate/chat suffix alone exceeds token budget')
   full=tokenizer.encode(text[:position],add_special_tokens=False)[:max_length-len(ending)]+ending
  ids.append(full)
 encoded=tokenizer.pad({'input_ids':ids},padding=True,return_tensors='pt')
 if device is not None:encoded={k:v.to(device) for k,v in encoded.items()}
 return CandidateBatch(candidates=candidates,input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'])
