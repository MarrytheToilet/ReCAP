from __future__ import annotations

import json

from recap.agents import AgentContext, CandidateAction
from recap.agents.preference_agent import (
    ActionPreference,
    PreferenceRerankAgent,
    load_action_preferences,
)
from recap.agents.learned_reranker_agent import LearnedRerankerAgent, load_reranker_model
from recap.models.policy_reranker import train_policy_reranker


class FixedAgent:
    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        return (
            CandidateAction("look", score=2.0),
            CandidateAction("open door", score=1.0),
        )


def make_context(history: tuple[str, ...] = ()) -> AgentContext:
    return AgentContext(
        task_id="/tmp/game.z8",
        seed=0,
        step_index=len(history),
        observation="obs",
        admissible_actions=("look", "open door"),
        history=history,
        state_signature=(),
        seen_signatures=(),
        initial_observation="initial",
    )


def test_preference_reranker_promotes_preferred_action() -> None:
    agent = PreferenceRerankAgent(
        FixedAgent(),
        (
            ActionPreference(
                task_id="game.z8",
                seed=0,
                history=(),
                preferred_action="open door",
                rejected_action="look",
                source="policy_repair_suffix",
            ),
        ),
        bonus=10.0,
    )

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["open door", "look"]
    assert candidates[0].metadata["preference_sources"] == ("policy_repair_suffix",)
    assert candidates[1].metadata["preference_rejected"] is True


def test_preference_reranker_ignores_history_mismatch() -> None:
    agent = PreferenceRerankAgent(
        FixedAgent(),
        (
            ActionPreference(
                task_id="game.z8",
                seed=0,
                history=("go north",),
                preferred_action="open door",
                rejected_action="look",
            ),
        ),
    )

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["look", "open door"]


def test_load_action_preferences(tmp_path) -> None:
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "game.z8",
                "seed": 0,
                "history": ["go north"],
                "preferred_action": "open door",
                "rejected_action": "look",
                "source": "policy_repair_suffix",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    preferences = load_action_preferences(path)

    assert preferences == (
        ActionPreference(
            task_id="game.z8",
            seed=0,
            history=("go north",),
            preferred_action="open door",
            rejected_action="look",
            source="policy_repair_suffix",
        ),
    )


def test_learned_reranker_promotes_high_scoring_candidate() -> None:
    model = {"weights": {"is_manipulation": 10.0, "is_raw_top1": -1.0}}
    agent = LearnedRerankerAgent(FixedAgent(), model, abstain_margin=0.0)

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["open door", "look"]
    assert candidates[0].metadata["recap_reranker_abstained"] is False


def test_learned_reranker_abstains_when_margin_is_small() -> None:
    model = {"weights": {"is_manipulation": 1.0, "is_raw_top1": 0.0}}
    agent = LearnedRerankerAgent(FixedAgent(), model, abstain_margin=5.0)

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["look", "open door"]
    assert candidates[0].metadata["recap_reranker_abstained"] is True


class ConstantGateEstimator:
    classes_ = (0, 1)

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, _x):
        return [[1.0 - self.positive_probability, self.positive_probability]]


def test_learned_reranker_uses_learned_intervention_gate() -> None:
    model = {"weights": {"is_manipulation": 10.0, "is_raw_top1": -1.0}}
    gate_model = {
        "model_type": "recap_intervention_gate",
        "estimator": ConstantGateEstimator(0.1),
        "threshold": 0.5,
    }
    agent = LearnedRerankerAgent(
        FixedAgent(),
        model,
        abstain_margin=0.0,
        gate_model=gate_model,
    )

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["look", "open door"]
    assert candidates[0].metadata["recap_reranker_abstained"] is True
    assert candidates[0].metadata["recap_reranker_safety_reason"] == "learned_intervention_gate"


def test_learned_reranker_gate_allows_high_probability_intervention() -> None:
    model = {"weights": {"is_manipulation": 10.0, "is_raw_top1": -1.0}}
    gate_model = {
        "model_type": "recap_intervention_gate",
        "estimator": ConstantGateEstimator(0.9),
        "threshold": 0.5,
    }
    agent = LearnedRerankerAgent(
        FixedAgent(),
        model,
        abstain_margin=0.0,
        gate_model=gate_model,
    )

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["open door", "look"]
    assert candidates[0].metadata["recap_reranker_abstained"] is False
    assert candidates[0].metadata["recap_intervention_gate_model_type"] == "recap_intervention_gate"


class StaticVerifier:
    def __init__(self, allow: bool) -> None:
        self.allow_value = allow

    def allow(self, context, raw_action, proposed_action, candidates):
        return self.allow_value, "test"


def test_learned_reranker_verifier_can_reject_intervention() -> None:
    model = {"weights": {"is_manipulation": 10.0, "is_raw_top1": -1.0}}
    agent = LearnedRerankerAgent(
        FixedAgent(),
        model,
        abstain_margin=0.0,
        intervention_verifier=StaticVerifier(False),
    )

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["look", "open door"]
    assert candidates[0].metadata["recap_reranker_abstained"] is True
    assert candidates[0].metadata["recap_reranker_safety_reason"] == "llm_verifier:test"


def test_learned_reranker_verifier_can_allow_intervention() -> None:
    model = {"weights": {"is_manipulation": 10.0, "is_raw_top1": -1.0}}
    agent = LearnedRerankerAgent(
        FixedAgent(),
        model,
        abstain_margin=0.0,
        intervention_verifier=StaticVerifier(True),
    )

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["open door", "look"]
    assert candidates[0].metadata["recap_reranker_abstained"] is False


class ManipulationFirstAgent:
    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        return (
            CandidateAction("open door", score=2.0),
            CandidateAction("go west", score=1.0),
        )


class NavigationRepeatAgent:
    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        return (
            CandidateAction("go south", score=2.0),
            CandidateAction("go west", score=1.0),
        )


def test_learned_reranker_conservative_gate_protects_raw_manipulation() -> None:
    model = {"weights": {"is_navigation": 10.0, "is_raw_top1": -1.0}}
    agent = LearnedRerankerAgent(
        ManipulationFirstAgent(),
        model,
        abstain_margin=0.0,
        safety_mode="conservative",
    )

    candidates = agent.candidates(make_context())

    assert [candidate.action for candidate in candidates] == ["open door", "go west"]
    assert candidates[0].metadata["recap_reranker_abstained"] is True
    assert candidates[0].metadata["recap_reranker_safety_reason"] == "would_demote_manipulation"


def test_learned_reranker_loop_safe_gate_protects_immediate_navigation_repeat() -> None:
    model = {"weights": {"is_raw_top1": -10.0, "raw_rank_from_bottom": 5.0}}
    agent = LearnedRerankerAgent(
        NavigationRepeatAgent(),
        model,
        abstain_margin=0.0,
        safety_mode="loop-safe",
    )

    candidates = agent.candidates(make_context(history=("go south",)))

    assert [candidate.action for candidate in candidates] == ["go south", "go west"]
    assert candidates[0].metadata["recap_reranker_abstained"] is True
    assert (
        candidates[0].metadata["recap_reranker_safety_reason"]
        == "protect_immediate_navigation_repeat"
    )


def test_load_policy_reranker_model_supports_online_agent(tmp_path) -> None:
    record = {
        "task_id": "game.z8",
        "seed": 0,
        "step_index": 0,
        "history": (),
        "candidates": ["look", "open door"],
        "preferred_action": "open door",
        "rejected_action": "look",
        "certificate_level": "C3_failure_repair",
        "repair_suffix_len": 1,
    }
    model = train_policy_reranker(
        (record,),
        hidden_dim=8,
        epochs=20,
        learning_rate=0.01,
        entropy_coef=0.0,
        seed=0,
    )
    path = tmp_path / "policy.pt"

    import torch

    torch.save(model, path)
    loaded = load_reranker_model(path)
    agent = LearnedRerankerAgent(FixedAgent(), loaded, abstain_margin=0.0)

    candidates = agent.candidates(make_context())

    assert loaded["model_type"] == "recap_support_constrained_policy"
    assert [candidate.action for candidate in candidates] == ["open door", "look"]
