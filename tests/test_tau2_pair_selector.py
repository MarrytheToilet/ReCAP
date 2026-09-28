"""Check candidate coverage and portable classifier semantics used in experiments."""
from scripts.tau2_pair_selector import probabilities,select

def record():
 return {'candidates':[{'kind':'message','content':str(i)} for i in range(3)],'history':[],'candidate_errors':[None,None,None]}

def test_third_candidate_can_be_selected():
 artifact={'classes':[-1,1],'vocabulary':{'alt_rank_reciprocal':0},'coef':[[-3.]],'intercept':[2.],'risk_penalty':1,'threshold':0}
 assert select(artifact,record())==2

def test_no_positive_evidence_abstains():
 assert select({'constant':0,'risk_penalty':1,'threshold':0},record())==0
 assert select({'constant':-1,'risk_penalty':1,'threshold':0},record())==0

def test_ties_prefer_earliest_logged_alternative():
 assert select({'constant':1,'risk_penalty':1,'threshold':0},record())==1

def test_missing_rescue_class_is_not_invented():
 artifact={'classes':[-1,0],'vocabulary':{},'coef':[[]],'intercept':[0.],'risk_penalty':1,'threshold':0}
 p=probabilities(artifact,record())
 assert p=={-1:.5,0:.5} and select(artifact,record())==0

def test_portable_tree_uses_the_selected_alternative_features():
 artifact={'classes':[-1,1],'vocabulary':{'alt_rank_reciprocal':0},'tree':{'left':[1,-1,-1],'right':[2,-1,-1],'feature':[0,-2,-2],'threshold':[.4,-2,-2],'value':[[5,5],[0,10],[10,0]]},'risk_penalty':1,'threshold':0}
 assert select(artifact,record())==2

def test_private_outcomes_are_not_features():
 from scripts.tau2_pair_selector import features
 a=record();b={**a,'reward':1,'gold_action':'secret','candidate_costs':{'secret':0},'user_scenario':'PRIVATE'}
 assert features(a)==features(b)
