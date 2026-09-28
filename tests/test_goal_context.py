from recap.models.goal_context import goal_context,encode_goal_batch
class CharTokenizer:
 def encode(self,text,add_special_tokens=True):return list(map(ord,text))
 def pad(self,data,**kwargs):
  import torch
  n=max(map(len,data['input_ids']));return {'input_ids':torch.tensor([[0]*(n-len(x))+x for x in data['input_ids']]),'attention_mask':torch.tensor([[0]*(n-len(x))+[1]*len(x) for x in data['input_ids']])}

def test_objective_keeps_late_instruction_without_room_dump():
 record={'initial_observation':'Your task: '+('Walk carefully. '*30)+'Finally eat apple.\n-= Room =-\nRoom dump','observation':'Now here','history':[],'candidates':['eat apple']}
 context=goal_context(record,max_observation_chars=10)
 assert 'Finally eat apple.' in context and 'Room dump' not in context

def test_long_context_cannot_delete_candidate_suffix():
 record={'initial_observation':'Your task: '+'x'*1600,'candidates':['eat apple','open door']}
 batch=encode_goal_batch(CharTokenizer(),record,max_length=220)
 texts=[''.join(chr(i) for i in row.tolist() if i) for row in batch.input_ids]
 assert texts[0].endswith('Answer:') and texts[1].endswith('Answer:')
 assert 'Candidate action: eat apple' in texts[0] and 'Candidate action: open door' in texts[1]
 assert batch.input_ids.shape[1]==220

class MockChatTokenizer(CharTokenizer):
 def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
  assert tokenize is False and add_generation_prompt is True and enable_thinking is False
  return '<USER>'+messages[0]['content']+'</USER><ASSISTANT>'

def test_chat_encoder_preserves_role_end_and_action_when_cropped():
 from recap.models.chat_goal_context import encode_chat_goal_batch
 record={'initial_observation':'Your task: '+'x'*1600,'candidates':['eat apple','open door']}
 batch=encode_chat_goal_batch(MockChatTokenizer(),record,max_length=220)
 texts=[''.join(chr(i) for i in row.tolist() if i) for row in batch.input_ids]
 assert all(t.startswith('<USER>') and t.endswith('</USER><ASSISTANT>Answer:') for t in texts)
 assert 'eat apple' in texts[0] and 'open door' in texts[1]
