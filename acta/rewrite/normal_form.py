from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

from acta.envs.base import EnvAdapter
from acta.probe import PairProbe, ProbeConfig


@dataclass(frozen=True)
class NormalizerConfig:
    """Conservative replay-backed normal form settings."""

    equivalence_mode: str = "full"
    max_passes: int = 20
    remove_noops: bool = True
    remove_absorbed: bool = True
    remove_inverse_pairs: bool = True
    canonicalize_commuting: bool = True


@dataclass(frozen=True)
class RewriteStep:
    """One local rewrite applied to a trajectory."""

    rule: str
    index: int
    before: tuple[str, ...]
    after: tuple[str, ...]
    prefix: tuple[str, ...]


@dataclass(frozen=True)
class NormalFormResult:
    """Result of normalizing and verifying a trajectory."""

    original_actions: tuple[str, ...]
    normalized_actions: tuple[str, ...]
    steps: tuple[RewriteStep, ...]
    original_signature: Hashable
    normalized_signature: Hashable
    original_valid: bool
    normalized_valid: bool
    state_preserved: bool
    metadata: dict[str, object] = field(default_factory=dict)


class ReplayNormalizer:
    """Normalize traces using local relations verified by environment replay."""

    def __init__(
        self,
        adapter: EnvAdapter,
        env_name: str,
        config: NormalizerConfig | None = None,
        action_key: Callable[[str], str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.env_name = env_name
        self.config = config or NormalizerConfig()
        self.action_key = action_key or (lambda action: action)
        self.probe = PairProbe(
            adapter=adapter,
            env_name=env_name,
            config=ProbeConfig(equivalence_mode=self.config.equivalence_mode),
        )

    def normalize(
        self,
        task_id: str,
        actions: Sequence[str],
        seed: int = 0,
    ) -> NormalFormResult:
        original = tuple(actions)
        current = list(original)
        steps: list[RewriteStep] = []

        for pass_id in range(self.config.max_passes):
            changed = False
            index = 0
            while index < len(current):
                rewritten = self._rewrite_at(task_id, seed, current, index)
                if rewritten is None:
                    index += 1
                    continue

                current = list(rewritten.after)
                steps.append(rewritten)
                changed = True
                index = max(rewritten.index - 1, 0)

            if not changed:
                break
        else:
            pass_id = self.config.max_passes - 1

        original_replay = self.adapter.replay(task_id, original, seed)
        normalized = tuple(current)
        normalized_replay = self.adapter.replay(task_id, normalized, seed)
        original_signature = self.adapter.signature(
            original_replay.state,
            mode=self.config.equivalence_mode,
        )
        normalized_signature = self.adapter.signature(
            normalized_replay.state,
            mode=self.config.equivalence_mode,
        )
        state_preserved = self.adapter.is_equivalent(
            original_signature,
            normalized_signature,
            mode=self.config.equivalence_mode,
        )

        return NormalFormResult(
            original_actions=original,
            normalized_actions=normalized,
            steps=tuple(steps),
            original_signature=original_signature,
            normalized_signature=normalized_signature,
            original_valid=original_replay.valid,
            normalized_valid=normalized_replay.valid,
            state_preserved=state_preserved,
            metadata={"passes": pass_id + 1, "env": self.env_name},
        )

    def _rewrite_at(
        self,
        task_id: str,
        seed: int,
        actions: list[str],
        index: int,
    ) -> RewriteStep | None:
        prefix = tuple(actions[:index])
        action = actions[index]

        if self.config.remove_noops and self._is_noop(task_id, seed, prefix, action):
            after = tuple(actions[:index] + actions[index + 1 :])
            return RewriteStep(
                rule="remove_noop",
                index=index,
                before=tuple(actions),
                after=after,
                prefix=prefix,
            )

        if index + 1 >= len(actions):
            return None

        next_action = actions[index + 1]
        record = self.probe.probe_pair(
            task_id=task_id,
            seed=seed,
            prefix_actions=prefix,
            action_a=action,
            action_b=next_action,
        )

        if self.config.remove_inverse_pairs and record.relations["inverse_ab"] is True:
            after = tuple(actions[:index] + actions[index + 2 :])
            return RewriteStep(
                rule="remove_inverse_pair",
                index=index,
                before=tuple(actions),
                after=after,
                prefix=prefix,
            )

        if self.config.remove_absorbed and record.relations["absorb_ab"] is True:
            after = tuple(actions[: index + 1] + actions[index + 2 :])
            return RewriteStep(
                rule="remove_absorbed_second",
                index=index + 1,
                before=tuple(actions),
                after=after,
                prefix=prefix,
            )

        if (
            self.config.canonicalize_commuting
            and record.relations["commute"] is True
            and self.action_key(action) > self.action_key(next_action)
        ):
            swapped = actions.copy()
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            return RewriteStep(
                rule="swap_commuting_pair",
                index=index,
                before=tuple(actions),
                after=tuple(swapped),
                prefix=prefix,
            )

        return None

    def _is_noop(
        self,
        task_id: str,
        seed: int,
        prefix: tuple[str, ...],
        action: str,
    ) -> bool:
        before = self.adapter.replay(task_id, prefix, seed)
        after = self.adapter.replay(task_id, prefix + (action,), seed)
        if not after.valid:
            return False
        sig_before = self.adapter.signature(before.state, mode=self.config.equivalence_mode)
        sig_after = self.adapter.signature(after.state, mode=self.config.equivalence_mode)
        return self.adapter.is_equivalent(sig_before, sig_after, mode=self.config.equivalence_mode)

