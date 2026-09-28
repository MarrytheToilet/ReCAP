"""Offline smoke check: malformed model actions score as failure, network errors do not."""
from scripts.run_tau2_candidates import (CandidateAgent,Orchestrator,UserSimulator,user_module,
 AssistantMessage,evaluate_simulation,EvaluationType)
from tau2.domains.retail.environment import get_environment,get_tasks
class Fake:
 def __init__(self,network=False):self.network=network
 def call(self,messages):
  if self.network:raise ConnectionError('synthetic transport failure')
  return '{"candidates": []}'
def episode(network=False):
 env=get_environment();agent=CandidateAgent(Fake(network),1,[],tools=env.get_tools(),domain_policy=env.get_policy(),llm='mock')
 user=UserSimulator(llm='mock',instructions='Mock user')
 task=get_tasks()[0];orch=Orchestrator(domain='retail',agent=agent,user=user,environment=env,task=task,max_steps=10)
 return orch,task
user_module.generate=lambda **kwargs:AssistantMessage(role='assistant',content='Hello',cost=0)
orch,task=episode();sim=orch.run();assert sim.termination_reason.value=='agent_error'
reward=evaluate_simulation(sim,task,EvaluationType.ALL,False,'retail');assert reward.reward==0
orch,_=episode(True)
try:orch.run()
except ConnectionError:pass
else:raise AssertionError('Network outage was incorrectly scored as model failure')
print('PASS: malformed model output -> agent_error / reward0; transport error stays unscored')
from tau2.data_model.message import UserMessage
class PartiallyMalformed:
 def call(self,messages):return '{"candidates":[{"kind":"message","content":"Hello"},{"kind":"message","content":"How can I help?"},{"kind":"invalid"}]}'
env=get_environment();agent=CandidateAgent(PartiallyMalformed(),1,[],tools=env.get_tools(),domain_policy=env.get_policy(),llm='mock')
message=agent._generate_next_message(UserMessage(role='user',content='Hello'),agent.get_init_state())
assert message.content=='Hello' and agent.records[-1]['candidate_errors'][2]
print('PASS: malformed unselected candidate does not invalidate a legal selected action')
class TwoValid:
 def call(self,messages):return '{"candidates":[{"kind":"message","content":"First"},{"kind":"message","content":"Second"}]}'
agent=CandidateAgent(TwoValid(),1,[],selector={'constant':1,'risk_penalty':1,'threshold':0},tools=env.get_tools(),domain_policy=env.get_policy(),llm='mock')
state=agent.get_init_state()
a=agent._generate_next_message(UserMessage(role='user',content='One'),state)
b=agent._generate_next_message(UserMessage(role='user',content='Two'),state)
assert (a.content,b.content)==('Second','First')
print('PASS: learned/fixed selector changes at most one decision per episode')
