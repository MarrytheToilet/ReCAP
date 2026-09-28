"""Same local-policy trainer and rewards, with an explicitly separate goal encoder."""
from recap.models import train_lm_candidate_policy as trainer
from recap.models.goal_context import encode_goal_batch
trainer.encode_candidate_batch=encode_goal_batch
if __name__=='__main__':trainer.main()
