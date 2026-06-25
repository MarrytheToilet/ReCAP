from __future__ import annotations

from acta.agents import AgentContext, CandidateAction
from acta.controllers import (
    ActAController,
    ReplayRepairController,
    ReplayVerifiedProposalController,
)
from acta.envs.toy_adapter import ToyAdapter
from acta.eval.agent_loop import run_episode


class FixedCandidateAgent:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        return tuple(CandidateAction(action=action) for action in self.actions)


class RerankedCandidateAgent:
    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        return (
            CandidateAction(
                "open door",
                metadata={
                    "recap_raw_rank": 2,
                    "recap_reranker_score": 2.0,
                    "recap_reranker_intervened": True,
                    "recap_reranker_abstained": False,
                },
            ),
            CandidateAction(
                "look",
                metadata={
                    "recap_raw_rank": 1,
                    "recap_reranker_score": 1.0,
                    "recap_reranker_intervened": False,
                    "recap_reranker_abstained": False,
                },
            ),
        )


class BadRerankedCandidateAgent:
    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        return (
            CandidateAction(
                "go kitchen",
                metadata={
                    "recap_raw_rank": 2,
                    "recap_reranker_score": 2.0,
                    "recap_reranker_intervened": True,
                    "recap_reranker_abstained": False,
                },
            ),
            CandidateAction(
                "look",
                metadata={
                    "recap_raw_rank": 1,
                    "recap_reranker_score": 1.0,
                    "recap_reranker_intervened": False,
                    "recap_reranker_abstained": False,
                },
            ),
        )


class GoldToyAdapter(ToyAdapter):
    def policy_commands(self, state: object) -> tuple[str, ...]:
        if getattr(state, "door_open", False):
            return ()
        return ("open door",)


class DoorGoalToyAdapter(GoldToyAdapter):
    def step(self, action: str):
        result = super().step(action)
        return type(result)(
            state=result.state,
            observation=result.observation,
            reward=result.reward,
            done=getattr(result.state, "door_open", False),
            valid=result.valid,
            info=result.info,
        )


def test_acta_controller_reranks_noop_below_effectful_action() -> None:
    adapter = ToyAdapter()
    reset = adapter.reset("toy-default", seed=0)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation=reset.observation,
        admissible_actions=tuple(adapter.admissible_actions(reset.state)),
        history=(),
        state_signature=adapter.signature(reset.state),
        seen_signatures=(adapter.signature(reset.state),),
    )
    controller = ActAController(adapter, env_name="toy")

    decision = controller.rerank(
        context,
        [CandidateAction("look"), CandidateAction("open door")],
    )

    assert decision.selected.action == "open door"
    assert "noop" in decision.reasons["look"]


def test_acta_controller_cache_preserves_decision() -> None:
    adapter = ToyAdapter()
    reset = adapter.reset("toy-default", seed=0)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation=reset.observation,
        admissible_actions=tuple(adapter.admissible_actions(reset.state)),
        history=(),
        state_signature=adapter.signature(reset.state),
        seen_signatures=(adapter.signature(reset.state),),
    )
    controller = ActAController(adapter, env_name="toy")
    candidates = [CandidateAction("look"), CandidateAction("open door")]

    first = controller.rerank(context, candidates)
    misses_after_first = dict(controller.cache_misses)
    second = controller.rerank(context, candidates)

    assert [candidate.action for candidate in first.candidates] == [
        candidate.action for candidate in second.candidates
    ]
    assert controller.cache_misses == misses_after_first
    assert controller.cache_hits["replay"] >= 2


def test_agent_loop_restores_state_after_controller_probe() -> None:
    adapter = ToyAdapter()
    agent = FixedCandidateAgent(["look", "open door"])
    controller = ActAController(adapter, env_name="toy")

    result = run_episode(
        adapter=adapter,
        agent=agent,
        task_id="toy-default",
        seed=0,
        max_steps=1,
        controller=controller,
    )

    assert result.steps[0].action == "open door"
    assert result.steps[0].valid
    assert result.metrics["steps"] == 1.0


def test_agent_loop_without_controller_uses_first_candidate() -> None:
    adapter = ToyAdapter()
    agent = FixedCandidateAgent(["look", "open door"])

    result = run_episode(
        adapter=adapter,
        agent=agent,
        task_id="toy-default",
        seed=0,
        max_steps=1,
        controller=None,
    )

    assert result.steps[0].action == "look"


def test_agent_loop_logs_gold_recovery_diagnostics() -> None:
    adapter = GoldToyAdapter()
    agent = FixedCandidateAgent(["look", "open door"])
    controller = ActAController(adapter, env_name="toy")

    result = run_episode(
        adapter=adapter,
        agent=agent,
        task_id="toy-default",
        seed=0,
        max_steps=1,
        controller=controller,
    )

    step = result.steps[0]
    assert step.action == "open door"
    assert step.candidates_before == ("look", "open door")
    assert step.candidates[0] == "open door"
    assert step.gold_action == "open door"
    assert step.gold_rank_before == 2
    assert step.gold_rank_after == 1
    assert step.selected_is_gold
    assert step.acta_recovered_gold
    assert not step.acta_demoted_gold
    assert step.top1_action == "look"
    assert "noop" in step.top1_structural_reasons
    assert "seen_state" in step.top1_structural_reasons
    assert step.top1_structural_bad
    assert step.acta_blocked_bad_top1
    assert result.metrics["selected_gold_rate"] == 1.0
    assert result.metrics["acta_recovered_gold"] == 1.0


def test_recap_replay_controller_promotes_verified_success_candidate() -> None:
    adapter = DoorGoalToyAdapter()
    reset = adapter.reset("toy-default", seed=0)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation=reset.observation,
        admissible_actions=tuple(adapter.admissible_actions(reset.state)),
        history=(),
        state_signature=adapter.signature(reset.state),
        seen_signatures=(adapter.signature(reset.state),),
    )
    controller = ReplayRepairController(adapter)

    decision = controller.rerank(
        context,
        [CandidateAction("look"), CandidateAction("open door")],
    )

    assert decision.selected.action == "open door"
    assert decision.reasons["open door"] == ("replay_success_suffix_0",)


def test_verified_proposal_controller_executes_certified_learned_proposal() -> None:
    adapter = DoorGoalToyAdapter()
    agent = RerankedCandidateAgent()
    controller = ReplayVerifiedProposalController(adapter)

    result = run_episode(
        adapter=adapter,
        agent=agent,
        task_id="toy-default",
        seed=0,
        max_steps=1,
        controller=controller,
    )

    step = result.steps[0]
    assert step.action == "open door"
    assert step.candidates_before == ("look", "open door")
    assert step.reranker_intervened
    assert step.controller_reasons == ("verified_proposal_suffix_0",)


def test_verified_proposal_controller_falls_back_when_proposal_uncertified() -> None:
    adapter = DoorGoalToyAdapter()
    agent = BadRerankedCandidateAgent()
    controller = ReplayVerifiedProposalController(adapter)

    result = run_episode(
        adapter=adapter,
        agent=agent,
        task_id="toy-default",
        seed=0,
        max_steps=1,
        controller=controller,
    )

    step = result.steps[0]
    assert step.action == "look"
    assert step.candidates_before == ("look", "go kitchen")
    assert not step.reranker_intervened
    assert step.controller_reasons == ("fallback_raw",)


def test_agent_loop_recovers_raw_order_from_reranker_metadata() -> None:
    adapter = GoldToyAdapter()
    agent = RerankedCandidateAgent()

    result = run_episode(
        adapter=adapter,
        agent=agent,
        task_id="toy-default",
        seed=0,
        max_steps=1,
        controller=None,
    )

    step = result.steps[0]
    assert step.action == "open door"
    assert step.candidates_before == ("look", "open door")
    assert step.candidates == ("open door", "look")
    assert step.gold_rank_before == 2
    assert step.gold_rank_after == 1
    assert step.reranker_used
    assert step.reranker_intervened
    assert result.metrics["reranker_interventions"] == 1.0
    assert result.metrics["reranker_recovered_gold"] == 1.0
