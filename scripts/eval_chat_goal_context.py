"""Same sharded evaluation as the plain goal-context arm, with chat encoding."""
import sys
from scripts import eval_goal_context as runner
from recap.models.chat_goal_context import encode_chat_goal_batch
runner.agent_module.encode_candidate_batch=encode_chat_goal_batch
if __name__=='__main__':
 if '--num-shards' in sys.argv:runner.evaluator.main()
 else:runner.parallel('scripts.eval_chat_goal_context')
