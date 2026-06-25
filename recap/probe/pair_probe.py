from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

from recap.envs.base import EnvAdapter, ReplayResult


Jsonable = Any


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for pair probing."""

    equivalence_mode: str = "full"


@dataclass(frozen=True)
class SequenceOutcome:
    """Replay outcome for one tested suffix."""

    actions: tuple[str, ...]
    signature: Hashable
    valid: bool
    replay: ReplayResult


@dataclass(frozen=True)
class RelationRecord:
    """Serializable relation probe record for one `(s, a, b)` tuple."""

    env: str
    task_id: str
    seed: int
    prefix_actions: tuple[str, ...]
    observation: str
    admissible_actions: tuple[str, ...]
    action_a: str
    action_b: str
    equivalence_mode: str
    signatures: Mapping[str, Hashable]
    validities: Mapping[str, bool]
    relations: Mapping[str, bool | None]
    metadata: Mapping[str, Jsonable] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def to_dict(self) -> dict[str, Jsonable]:
        return {
            "env": self.env,
            "task_id": self.task_id,
            "seed": self.seed,
            "prefix_actions": list(self.prefix_actions),
            "observation": self.observation,
            "admissible_actions": list(self.admissible_actions),
            "action_a": self.action_a,
            "action_b": self.action_b,
            "equivalence_mode": self.equivalence_mode,
            "signatures": _jsonable(self.signatures),
            "validities": dict(self.validities),
            "relations": dict(self.relations),
            "metadata": _jsonable(self.metadata),
        }


class PairProbe:
    """Probe contextual relations between two actions with replay."""

    def __init__(self, adapter: EnvAdapter, env_name: str, config: ProbeConfig | None = None) -> None:
        self.adapter = adapter
        self.env_name = env_name
        self.config = config or ProbeConfig()

    def probe_pair(
        self,
        task_id: str,
        seed: int,
        prefix_actions: Sequence[str],
        action_a: str,
        action_b: str,
    ) -> RelationRecord:
        prefix = tuple(prefix_actions)
        prefix_replay = self.adapter.replay(task_id=task_id, prefix_actions=prefix, seed=seed)
        mode = self.config.equivalence_mode
        sig_s = self.adapter.signature(prefix_replay.state, mode=mode)

        admissible = tuple(self.adapter.admissible_actions(prefix_replay.state))

        outcomes = {
            "a": self._run_suffix(task_id, seed, prefix, (action_a,)),
            "b": self._run_suffix(task_id, seed, prefix, (action_b,)),
            "ab": self._run_suffix(task_id, seed, prefix, (action_a, action_b)),
            "ba": self._run_suffix(task_id, seed, prefix, (action_b, action_a)),
            "aa": self._run_suffix(task_id, seed, prefix, (action_a, action_a)),
            "bb": self._run_suffix(task_id, seed, prefix, (action_b, action_b)),
        }

        signatures = {"s": sig_s} | {name: outcome.signature for name, outcome in outcomes.items()}
        validities = {"prefix": prefix_replay.valid} | {
            name: outcome.valid for name, outcome in outcomes.items()
        }

        relations: dict[str, bool | None] = {
            "commute": self._commute_relation(outcomes, signatures),
            "idempotent_a": self._idempotent_relation(outcomes["a"], outcomes["aa"], signatures["a"], signatures["aa"]),
            "idempotent_b": self._idempotent_relation(outcomes["b"], outcomes["bb"], signatures["b"], signatures["bb"]),
            "inverse_ab": self._inverse_relation(outcomes["a"], outcomes["ab"], signatures["s"], signatures["a"], signatures["ab"]),
            "inverse_ba": self._inverse_relation(outcomes["b"], outcomes["ba"], signatures["s"], signatures["b"], signatures["ba"]),
            "absorb_ab": self._absorb_relation(outcomes["a"], outcomes["ab"], signatures["a"], signatures["ab"]),
            "absorb_ba": self._absorb_relation(outcomes["b"], outcomes["ba"], signatures["b"], signatures["ba"]),
            "dependency_a_then_b": (not outcomes["b"].valid) and outcomes["ab"].valid,
            "dependency_b_then_a": (not outcomes["a"].valid) and outcomes["ba"].valid,
        }

        return RelationRecord(
            env=self.env_name,
            task_id=task_id,
            seed=seed,
            prefix_actions=prefix,
            observation=prefix_replay.observation,
            admissible_actions=admissible,
            action_a=action_a,
            action_b=action_b,
            equivalence_mode=mode,
            signatures=signatures,
            validities=validities,
            relations=relations,
            metadata={"implemented_core_relations": ["commute", "idempotent_a", "idempotent_b"]},
        )

    def _run_suffix(
        self,
        task_id: str,
        seed: int,
        prefix_actions: tuple[str, ...],
        suffix_actions: tuple[str, ...],
    ) -> SequenceOutcome:
        actions = prefix_actions + suffix_actions
        replay = self.adapter.replay(task_id=task_id, prefix_actions=actions, seed=seed)
        suffix_steps = replay.steps[len(prefix_actions) :]
        suffix_valid = replay.valid and len(suffix_steps) == len(suffix_actions)
        signature = self.adapter.signature(replay.state, mode=self.config.equivalence_mode)
        return SequenceOutcome(
            actions=suffix_actions,
            signature=signature,
            valid=suffix_valid,
            replay=replay,
        )

    def _equivalent(self, sig1: Hashable, sig2: Hashable) -> bool:
        return self.adapter.is_equivalent(sig1, sig2, mode=self.config.equivalence_mode)

    def _commute_relation(
        self,
        outcomes: Mapping[str, SequenceOutcome],
        signatures: Mapping[str, Hashable],
    ) -> bool | None:
        if not outcomes["ab"].valid or not outcomes["ba"].valid:
            return None
        return self._equivalent(signatures["ab"], signatures["ba"])

    def _idempotent_relation(
        self,
        single: SequenceOutcome,
        double: SequenceOutcome,
        sig_single: Hashable,
        sig_double: Hashable,
    ) -> bool | None:
        if not single.valid or not double.valid:
            return None
        return self._equivalent(sig_single, sig_double)

    def _inverse_relation(
        self,
        first: SequenceOutcome,
        outcome: SequenceOutcome,
        sig_start: Hashable,
        sig_first: Hashable,
        sig_pair: Hashable,
    ) -> bool | None:
        if not first.valid or not outcome.valid:
            return None
        if self._equivalent(sig_start, sig_first):
            return False
        return self._equivalent(sig_start, sig_pair)

    def _absorb_relation(
        self,
        first: SequenceOutcome,
        pair: SequenceOutcome,
        sig_first: Hashable,
        sig_pair: Hashable,
    ) -> bool | None:
        if not first.valid or not pair.valid:
            return None
        return self._equivalent(sig_first, sig_pair)


def _jsonable(value: Any) -> Jsonable:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_jsonable(item) for item in value)
    return value
