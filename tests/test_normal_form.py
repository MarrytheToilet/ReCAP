from __future__ import annotations

from recap.envs.toy_adapter import ToyAdapter
from recap.rewrite import NormalizerConfig, ReplayNormalizer


def make_normalizer(**kwargs: object) -> ReplayNormalizer:
    return ReplayNormalizer(
        ToyAdapter(),
        env_name="toy",
        config=NormalizerConfig(**kwargs),
    )


def test_normalizer_removes_noops_and_absorbed_repetition() -> None:
    normalizer = make_normalizer()
    result = normalizer.normalize(
        "toy-default",
        ["look", "open door", "open door", "read note", "read note"],
        seed=0,
    )

    assert result.state_preserved
    assert result.normalized_valid
    assert result.normalized_actions == ("open door", "read note")
    assert [step.rule for step in result.steps].count("remove_noop") == 1
    assert [step.rule for step in result.steps].count("remove_absorbed_second") == 2


def test_normalizer_removes_inverse_pair() -> None:
    normalizer = make_normalizer()
    result = normalizer.normalize(
        "toy-default",
        ["toggle lamp", "toggle lamp", "open door"],
        seed=0,
    )

    assert result.state_preserved
    assert result.normalized_actions == ("open door",)
    assert any(step.rule == "remove_inverse_pair" for step in result.steps)


def test_normalizer_swaps_commuting_actions_to_canonical_order() -> None:
    normalizer = make_normalizer(remove_noops=False, remove_absorbed=False)
    result = normalizer.normalize(
        "toy-default",
        ["read note", "open door"],
        seed=0,
    )

    assert result.state_preserved
    assert result.normalized_actions == ("open door", "read note")
    assert [step.rule for step in result.steps] == ["swap_commuting_pair"]


def test_normalizer_does_not_swap_noncommuting_actions() -> None:
    normalizer = make_normalizer(remove_noops=False, remove_absorbed=False)
    result = normalizer.normalize(
        "toy-default",
        ["go kitchen", "open door"],
        seed=0,
    )

    assert result.state_preserved
    assert result.normalized_actions == ("go kitchen", "open door")
    assert not result.steps
