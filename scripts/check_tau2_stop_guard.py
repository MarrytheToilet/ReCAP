"""Check that simulator repair only removes a premature termination marker."""
import json
from tau2.data_model.message import SystemMessage,UserMessage
from scripts.tau2_user_stop_guard import UserStopGuard
class Backend:
 def __init__(self,response):self.response=response;self.requests=[]
 def call(self,messages):self.requests.append(messages);return self.response
messages=[SystemMessage(role='system',content='PRIVATE_SCENARIO'),UserMessage(role='user',content='Please confirm before I cancel the order.')]
base=Backend('Yes, please cancel. ###STOP###');critic=Backend(json.dumps({'pending_authorization':True,'evidence':'Customer authorizes cancellation; agent has not performed it.'}))
g=UserStopGuard(base,critic);assert g.user(messages).content=='Yes, please cancel.'
assert 'PRIVATE_SCENARIO' not in json.dumps(critic.requests)
base=Backend('Thanks, goodbye. ###STOP###');critic=Backend(json.dumps({'pending_authorization':False,'evidence':'Farewell after completion.'}))
assert UserStopGuard(base,critic).user(messages).content==base.response
base=Backend('Yes, please cancel.');critic=Backend('unusable')
assert UserStopGuard(base,critic).user(messages).content==base.response and not critic.requests
base=Backend('Yes, please cancel. ###STOP###');critic=Backend('{"pending_authorization":"yes","evidence":"x"}')
try:UserStopGuard(base,critic).user(messages)
except AssertionError:pass
else:raise AssertionError('Invalid guard response must stay unscored, not silently alter termination')
print('PASS: exact marker-only edit; no private scenario to guard; preserve legitimate stop; bypass ordinary turns; reject malformed verdict')
