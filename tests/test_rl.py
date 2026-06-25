from __future__ import annotations

import random

from recap.agents import AgentContext
from recap.controllers import PriorController
from recap.envs.toy_adapter import ToyAdapter
from recap.rl.tabular_q import QLearningConfig, choose_action


def test_recap_hard_prior_filters_noop_from_q_action_choice() -> None:
    adapter = ToyAdapter()
    reset = adapter.reset("toy-default", seed=0)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation=reset.observation,
        admissible_actions=("look", "open door"),
        history=(),
        state_signature=adapter.signature(reset.state),
        seen_signatures=(adapter.signature(reset.state),),
        initial_observation=reset.observation,
    )
    controller = PriorController(adapter, env_name="toy")

    choice = choose_action(
        q_values={},
        state_signature=adapter.signature(reset.state),
        actions=("look", "open door"),
        context=context,
        config=QLearningConfig(prior="prior-hard", epsilon=0.0),
        rng=random.Random(0),
        controller=controller,
        epsilon=0.0,
    )

    assert choice.action == "open door"
    assert "look" in choice.blocked_actions
    assert "noop" in choice.blocked_actions["look"]
