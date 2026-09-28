"""Same1800-context softmax training; only the goal encoder's chat template differs."""
from recap.models import train_lm_candidate_policy as trainer
from recap.models.chat_goal_context import encode_chat_goal_batch
trainer.encode_candidate_batch=encode_chat_goal_batch
if __name__=='__main__':trainer.main()
