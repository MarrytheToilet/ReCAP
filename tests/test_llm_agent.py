from __future__ import annotations

from acta.agents import AgentContext
from acta.agents.llm_agent import (
    LLMConfig,
    MockLLMAgent,
    load_response_cache,
    parse_action_response,
    response_cache_key,
    save_response_cache,
)
from acta.agents.lm_policy_agent import (
    is_semantic_undo,
    rank_lm_policy_pool,
    structural_penalty,
)
from acta.agents.noisy_agent import NoisyCandidateAgent


def test_parse_action_response_filters_to_admissible_actions() -> None:
    actions = parse_action_response(
        '{"actions": ["open door", "fly", "look", "open door"]}',
        ("look", "open door"),
    )

    assert actions == ["open door", "look"]


def test_mock_llm_agent_returns_top_k_candidates() -> None:
    agent = MockLLMAgent(max_candidates=2)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation="",
        admissible_actions=("look", "open door", "inventory", "read note"),
        history=(),
        state_signature=(),
        seen_signatures=(),
    )

    candidates = agent.candidates(context)

    assert [candidate.action for candidate in candidates] == ["open door", "read note"]
    assert all(candidate.source == "mock-llm" for candidate in candidates)


def test_noisy_candidate_agent_frontloads_structural_distractors() -> None:
    base = MockLLMAgent(max_candidates=2)
    agent = NoisyCandidateAgent(base, max_candidates=4)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation="",
        admissible_actions=("look", "open door", "inventory", "read note"),
        history=(),
        state_signature=(),
        seen_signatures=(),
    )

    candidates = agent.candidates(context)

    assert [candidate.action for candidate in candidates][:2] == ["inventory", "look"]
    assert any(candidate.action == "open door" for candidate in candidates)


def test_noisy_candidate_agent_can_reorder_without_injection() -> None:
    base = MockLLMAgent(max_candidates=4)
    agent = NoisyCandidateAgent(base, mode="frontload-existing-structural", max_candidates=4)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation="",
        admissible_actions=("open door", "look", "inventory", "read note"),
        history=(),
        state_signature=(),
        seen_signatures=(),
    )

    candidates = agent.candidates(context)

    assert {candidate.action for candidate in candidates} == {"open door", "look", "inventory", "read note"}
    assert [candidate.action for candidate in candidates][:2] == ["inventory", "look"]


def test_llm_response_cache_key_is_state_specific(tmp_path) -> None:
    config = LLMConfig(model="test-model", max_candidates=2, temperature=1.0)
    context = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=0,
        observation="room",
        admissible_actions=("look", "open door"),
        history=(),
        state_signature=(),
        seen_signatures=(),
    )
    changed = AgentContext(
        task_id="toy-default",
        seed=0,
        step_index=1,
        observation="room",
        admissible_actions=("look", "open door"),
        history=("look",),
        state_signature=(),
        seen_signatures=(),
    )

    key = response_cache_key(context, config)
    assert key != response_cache_key(changed, config)

    path = tmp_path / "cache.json"
    save_response_cache(path, {key: '{"actions": ["open door"]}'})

    assert load_response_cache(path)[key] == '{"actions": ["open door"]}'


def test_lm_policy_progress_pool_prioritizes_goal_relevant_actions() -> None:
    ranked = rank_lm_policy_pool(
        (
            "examine shelf",
            "drop keycard",
            "go west",
            "look",
            "open chest",
        ),
        history=(),
        objective="First, travel west. Then open the chest.",
        mode="progress",
    )

    assert ranked[:2] == ["go west", "open chest"]
    assert ranked[-1] == "look"


def test_lm_policy_structural_penalty_detects_semantic_undo() -> None:
    assert is_semantic_undo("take keycard from box", "insert keycard into box")
    assert is_semantic_undo("open chest", "close chest")
    penalty = structural_penalty(
        "insert keycard into box",
        ("take keycard from box",),
        semantic_undo_penalty=3.0,
    )

    assert penalty == 3.0
