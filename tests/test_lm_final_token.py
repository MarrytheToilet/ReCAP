from types import SimpleNamespace
import torch
from recap.models.lm_candidate_policy import final_token_logits
class LegacyModel:
 def __call__(self,input_ids,attention_mask):return SimpleNamespace(logits=input_ids.float().unsqueeze(-1))
def test_fallback_handles_left_and_right_padding_positions():
 ids=torch.tensor([[0,0,21,22],[31,32,0,0]]);mask=torch.tensor([[0,0,1,1],[1,1,0,0]])
 assert final_token_logits(LegacyModel(),ids,mask).squeeze(-1).tolist()==[22,32]

def test_fallback_rejects_all_padding():
 import pytest
 with pytest.raises(ValueError,match='entirely padded'):
  final_token_logits(LegacyModel(),torch.zeros((1,3),dtype=torch.long),torch.zeros((1,3),dtype=torch.long))
