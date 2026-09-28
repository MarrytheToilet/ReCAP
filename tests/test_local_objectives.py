import torch
from scripts.run_certificate_objectives import objective,metrics,select_gate

def batch(costs):
    return dict(mask=torch.ones(1,len(costs),dtype=torch.bool),costs=torch.tensor([costs]),pos=torch.tensor([1]),neg=torch.tensor([0]))

def test_pairwise_certificate_does_not_label_uncompared_candidate():
    z=torch.tensor([[1.,2.,3.]],requires_grad=True)
    objective(z,batch([3.,1.,0.]),'certified_pair').backward()
    assert z.grad[0,2]==0
    assert z.grad[0,0]>0 and z.grad[0,1]<0

def test_cost_pairs_ignore_unknown_and_ties():
    z=torch.tensor([[0.,0.,0.]],requires_grad=True)
    objective(z,batch([1.,1.,float('inf')]),'cost_pairs').backward()
    assert torch.equal(z.grad,torch.zeros_like(z))

def test_minima_does_not_force_emitted_label():
    z=torch.tensor([[0.,0.,0.]],requires_grad=True)
    objective(z,batch([1.,2.,1.]),'minima').backward()
    assert z.grad[0,0]<0 and z.grad[0,2]<0 and z.grad[0,1]>0

def test_natural_metrics_count_harm_and_unresolved():
    rows=[dict(task_id=str(i),step_index=0,candidates=['a','b'],executed_action='a',candidate_costs=c)
          for i,c in enumerate([{'a':3,'b':1},{'a':1,'b':2},{'a':1,'b':None}])]
    scores=[[0.,1.]]*3;m,_=metrics(rows,scores)
    assert m['improved']==1 and m['harmed']==1 and m['unresolved_interventions']==1
    assert m['intervention_precision']==1/3
    t=select_gate(rows,scores);g,_=metrics(rows,scores,t)
    assert g['interventions']==0

def test_local_pairwise_reward_ties_and_order():
    from recap.models.lm_candidate_policy import pairwise_reward_loss
    prior=torch.ones(3)/3
    z=torch.zeros(3,requires_grad=True)
    loss,m=pairwise_reward_loss(z,torch.tensor([1.,1.,1.]),prior)
    loss.backward();assert torch.equal(z.grad,torch.zeros_like(z))
    z=torch.zeros(3,requires_grad=True)
    loss,m=pairwise_reward_loss(z,torch.tensor([2.,1.,1.]),prior)
    loss.backward();assert z.grad[0]<0 and z.grad[1]>0 and z.grad[2]>0
    assert m['preference_comparison_count']==2

def test_minima_does_not_treat_censored_action_as_negative():
    z=torch.tensor([[0.,0.,0.]],requires_grad=True)
    objective(z,batch([1.,2.,float('inf')]),'minima').backward()
    assert z.grad[0,2]==0 and z.grad[0,0]<0 and z.grad[0,1]>0
