import json
import pytest
from scripts.tau2_candidate_json import parse_candidates

def test_preserves_actions_and_delimiters_in_strings():
 actions=[{'kind':'tool','name':'x','arguments':{'s':'literal } ] \\"'}},{'kind':'message','content':'do not alter } punctuation'}]
 good=json.dumps({'candidates':actions})
 assert parse_candidates(good)==actions
 bad=good.replace('punctuation"}', 'punctuation"}}')
 assert parse_candidates(bad)==actions

def test_single_action_envelope():
 action={'kind':'message','content':'Ask for identity'}
 assert parse_candidates(json.dumps(action))==[action]

@pytest.mark.parametrize('bad',['{"candidates":[{"kind":"message","content":"truncated', '{"candidates":[]}', '{"x":1}', '{"candidates":[{"kind":"tool"]}'])
def test_ambiguous_or_empty_rejected(bad):
 with pytest.raises(ValueError):parse_candidates(bad)
