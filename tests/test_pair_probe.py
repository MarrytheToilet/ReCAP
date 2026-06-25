from __future__ import annotations

import json

from acta.envs.toy_adapter import ToyAdapter
from acta.probe import PairProbe, ProbeConfig


def make_probe(mode: str = "full") -> PairProbe:
    return PairProbe(ToyAdapter(), env_name="toy", config=ProbeConfig(equivalence_mode=mode))


def test_replay_is_deterministic() -> None:
    adapter = ToyAdapter()
    first = adapter.replay("toy-default", ["open door", "go kitchen", "take apple"], seed=7)
    second = adapter.replay("toy-default", ["open door", "go kitchen", "take apple"], seed=7)

    assert first.valid
    assert second.valid
    assert adapter.signature(first.state) == adapter.signature(second.state)


def test_commute_for_independent_idempotent_actions() -> None:
    record = make_probe().probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=[],
        action_a="look",
        action_b="read note",
    )

    assert record.relations["commute"] is True
    assert record.relations["idempotent_a"] is True
    assert record.relations["idempotent_b"] is True
    assert record.relations["inverse_ab"] is False


def test_idempotent_open_door() -> None:
    record = make_probe().probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=[],
        action_a="open door",
        action_b="toggle lamp",
    )

    assert record.relations["idempotent_a"] is True
    assert record.relations["idempotent_b"] is False


def test_invalid_repetition_is_unknown_not_idempotent() -> None:
    record = make_probe().probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=["open door", "go kitchen"],
        action_a="take apple",
        action_b="look",
    )

    assert record.validities["a"] is True
    assert record.validities["aa"] is False
    assert record.relations["idempotent_a"] is None


def test_dependency_detects_enabled_action() -> None:
    record = make_probe().probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=[],
        action_a="open door",
        action_b="go kitchen",
    )

    assert record.validities["b"] is False
    assert record.validities["ab"] is True
    assert record.relations["dependency_a_then_b"] is True


def test_invalid_pairs_are_not_mislabeled_as_commuting() -> None:
    record = make_probe().probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=[],
        action_a="go kitchen",
        action_b="take apple",
    )

    assert record.validities["ab"] is False
    assert record.validities["ba"] is False
    assert record.relations["commute"] is None


def test_goal_equivalence_can_ignore_irrelevant_facts() -> None:
    full_record = make_probe("full").probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=[],
        action_a="toggle lamp",
        action_b="read note",
    )
    goal_record = make_probe("goal").probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=[],
        action_a="toggle lamp",
        action_b="read note",
    )

    assert full_record.relations["idempotent_a"] is False
    assert goal_record.relations["idempotent_a"] is True


def test_relation_record_serializes_to_json() -> None:
    record = make_probe().probe_pair(
        task_id="toy-default",
        seed=0,
        prefix_actions=[],
        action_a="look",
        action_b="read note",
    )

    payload = json.loads(record.to_json())
    assert payload["env"] == "toy"
    assert payload["relations"]["commute"] is True
    assert payload["signatures"]["s"]
