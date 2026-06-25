from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

from recap.envs.base import EnvAdapter


@dataclass(frozen=True)
class TraceReplayOutcome:
    actions: tuple[str, ...]
    valid: bool
    success: bool
    done: bool
    total_reward: float
    final_signature: Hashable


@dataclass(frozen=True)
class TraceEditRecord:
    edit_type: str
    index: int
    action: str
    labels: tuple[str, ...]
    edited_actions: tuple[str, ...]
    valid: bool
    success: bool
    total_reward: float
    final_equivalent_to_original: bool
    replacement: str | None = None
    repair_suffix: tuple[str, ...] = ()
    second_index: int | None = None
    second_action: str | None = None


@dataclass(frozen=True)
class TraceActionSummary:
    index: int
    action: str
    labels: tuple[str, ...]
    repair_replacements: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceEditReport:
    task_id: str
    seed: int
    original_actions: tuple[str, ...]
    original_success: bool
    replay_success: bool
    replay_valid: bool
    total_reward: float
    final_signature: Hashable
    action_summaries: tuple[TraceActionSummary, ...]
    edits: tuple[TraceEditRecord, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEditConfig:
    equivalence_mode: str = "full"
    probe_deletions: bool = True
    probe_swaps: bool = True
    probe_replacements: bool = True
    probe_policy_repairs: bool = True
    probe_suffix_match_repairs: bool = False
    probe_candidate_policy_repairs: bool = False
    max_replacements_per_step: int = 5
    candidate_branch_limit: int = 10
    suffix_match_limit_per_step: int = 3


class TraceEditProbe:
    """Compile an executed trace into counterfactual edit diagnostics."""

    def __init__(
        self,
        adapter: EnvAdapter,
        config: TraceEditConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or TraceEditConfig()

    def probe_episode(self, episode: Mapping[str, Any]) -> TraceEditReport:
        task_id = str(episode["task_id"])
        seed = int(episode.get("seed", 0))
        steps = tuple(episode.get("steps", ()))
        actions = tuple(str(step["action"]) for step in steps)
        original_success = bool(episode.get("success", False))
        original = self.replay(task_id=task_id, seed=seed, actions=actions)

        edits: list[TraceEditRecord] = []
        if self.config.probe_deletions:
            edits.extend(
                self._probe_deletions(
                    task_id=task_id,
                    seed=seed,
                    actions=actions,
                    original=original,
                    original_success=original_success,
                )
            )
        if self.config.probe_swaps:
            edits.extend(
                self._probe_swaps(
                    task_id=task_id,
                    seed=seed,
                    actions=actions,
                    original=original,
                    original_success=original_success,
                )
            )
        if self.config.probe_replacements:
            edits.extend(
                self._probe_replacements(
                    task_id=task_id,
                    seed=seed,
                    steps=steps,
                    actions=actions,
                    original=original,
                    original_success=original_success,
                )
            )
        if self.config.probe_policy_repairs:
            edits.extend(
                self._probe_policy_repairs(
                    task_id=task_id,
                    seed=seed,
                    actions=actions,
                    original=original,
                    original_success=original_success,
                )
            )
        if self.config.probe_suffix_match_repairs:
            edits.extend(
                self._probe_suffix_match_repairs(
                    task_id=task_id,
                    seed=seed,
                    steps=steps,
                    actions=actions,
                    original=original,
                    original_success=original_success,
                )
            )
        if self.config.probe_candidate_policy_repairs:
            edits.extend(
                self._probe_candidate_policy_repairs(
                    task_id=task_id,
                    seed=seed,
                    steps=steps,
                    actions=actions,
                    original=original,
                    original_success=original_success,
                )
            )

        action_summaries = summarize_actions(actions, tuple(edits))
        return TraceEditReport(
            task_id=task_id,
            seed=seed,
            original_actions=actions,
            original_success=original_success,
            replay_success=original.success,
            replay_valid=original.valid,
            total_reward=original.total_reward,
            final_signature=original.final_signature,
            action_summaries=action_summaries,
            edits=tuple(edits),
            metrics=trace_metrics(tuple(edits), len(actions)),
        )

    def replay(self, task_id: str, seed: int, actions: Sequence[str]) -> TraceReplayOutcome:
        replay = self.adapter.replay(task_id=task_id, prefix_actions=tuple(actions), seed=seed)
        final_signature = self.adapter.signature(replay.state, mode=self.config.equivalence_mode)
        return TraceReplayOutcome(
            actions=tuple(actions),
            valid=replay.valid,
            success=replay_success(replay.state, replay.done),
            done=replay.done,
            total_reward=sum(step.reward for step in replay.steps),
            final_signature=final_signature,
        )

    def _probe_deletions(
        self,
        task_id: str,
        seed: int,
        actions: tuple[str, ...],
        original: TraceReplayOutcome,
        original_success: bool,
    ) -> tuple[TraceEditRecord, ...]:
        records: list[TraceEditRecord] = []
        for index, action in enumerate(actions):
            edited_actions = actions[:index] + actions[index + 1 :]
            outcome = self.replay(task_id=task_id, seed=seed, actions=edited_actions)
            equivalent = self._equivalent(outcome.final_signature, original.final_signature)
            labels = deletion_labels(
                original_success=original_success,
                outcome_success=outcome.success,
                final_equivalent=equivalent,
            )
            records.append(
                TraceEditRecord(
                    edit_type="delete",
                    index=index,
                    action=action,
                    labels=labels,
                    edited_actions=edited_actions,
                    valid=outcome.valid,
                    success=outcome.success,
                    total_reward=outcome.total_reward,
                    final_equivalent_to_original=equivalent,
                )
            )
        return tuple(records)

    def _probe_swaps(
        self,
        task_id: str,
        seed: int,
        actions: tuple[str, ...],
        original: TraceReplayOutcome,
        original_success: bool,
    ) -> tuple[TraceEditRecord, ...]:
        records: list[TraceEditRecord] = []
        for index in range(len(actions) - 1):
            action = actions[index]
            second_action = actions[index + 1]
            edited = list(actions)
            edited[index], edited[index + 1] = edited[index + 1], edited[index]
            edited_actions = tuple(edited)
            outcome = self.replay(task_id=task_id, seed=seed, actions=edited_actions)
            equivalent = self._equivalent(outcome.final_signature, original.final_signature)
            labels = swap_labels(
                original_success=original_success,
                outcome_success=outcome.success,
                final_equivalent=equivalent,
            )
            records.append(
                TraceEditRecord(
                    edit_type="swap_adjacent",
                    index=index,
                    action=action,
                    labels=labels,
                    edited_actions=edited_actions,
                    valid=outcome.valid,
                    success=outcome.success,
                    total_reward=outcome.total_reward,
                    final_equivalent_to_original=equivalent,
                    second_index=index + 1,
                    second_action=second_action,
                )
            )
        return tuple(records)

    def _probe_replacements(
        self,
        task_id: str,
        seed: int,
        steps: tuple[Mapping[str, Any], ...],
        actions: tuple[str, ...],
        original: TraceReplayOutcome,
        original_success: bool,
    ) -> tuple[TraceEditRecord, ...]:
        records: list[TraceEditRecord] = []
        for index, step in enumerate(steps):
            action = actions[index]
            candidates = replacement_candidates(
                step=step,
                original_action=action,
                limit=self.config.max_replacements_per_step,
            )
            for replacement in candidates:
                edited_actions = actions[:index] + (replacement,) + actions[index + 1 :]
                outcome = self.replay(task_id=task_id, seed=seed, actions=edited_actions)
                equivalent = self._equivalent(outcome.final_signature, original.final_signature)
                labels = replacement_labels(
                    original_success=original_success,
                    outcome_success=outcome.success,
                    final_equivalent=equivalent,
                )
                records.append(
                    TraceEditRecord(
                        edit_type="replace",
                        index=index,
                        action=action,
                        labels=labels,
                        replacement=replacement,
                        edited_actions=edited_actions,
                        valid=outcome.valid,
                        success=outcome.success,
                        total_reward=outcome.total_reward,
                        final_equivalent_to_original=equivalent,
                    )
                )
        return tuple(records)

    def _probe_policy_repairs(
        self,
        task_id: str,
        seed: int,
        actions: tuple[str, ...],
        original: TraceReplayOutcome,
        original_success: bool,
    ) -> tuple[TraceEditRecord, ...]:
        records: list[TraceEditRecord] = []
        for prefix_len in range(len(actions) + 1):
            prefix = actions[:prefix_len]
            prefix_replay = self.adapter.replay(task_id=task_id, prefix_actions=prefix, seed=seed)
            suffix = tuple(str(action) for action in self.adapter.policy_commands(prefix_replay.state))
            if not suffix and not prefix_replay.done:
                continue
            edited_actions = prefix + suffix
            outcome = self.replay(task_id=task_id, seed=seed, actions=edited_actions)
            equivalent = self._equivalent(outcome.final_signature, original.final_signature)
            labels = policy_repair_labels(
                original_success=original_success,
                outcome_success=outcome.success,
                prefix_len=prefix_len,
                original_len=len(actions),
                suffix_len=len(suffix),
                final_equivalent=equivalent,
            )
            if not labels:
                continue
            action_index = prefix_len - 1
            records.append(
                TraceEditRecord(
                    edit_type="policy_suffix",
                    index=action_index,
                    action=actions[action_index] if action_index >= 0 else "<start>",
                    labels=labels,
                    repair_suffix=suffix,
                    edited_actions=edited_actions,
                    valid=outcome.valid,
                    success=outcome.success,
                    total_reward=outcome.total_reward,
                    final_equivalent_to_original=equivalent,
                )
            )
        return tuple(records)

    def _probe_suffix_match_repairs(
        self,
        task_id: str,
        seed: int,
        steps: tuple[Mapping[str, Any], ...],
        actions: tuple[str, ...],
        original: TraceReplayOutcome,
        original_success: bool,
    ) -> tuple[TraceEditRecord, ...]:
        records: list[TraceEditRecord] = []
        for index, step in enumerate(steps):
            action = actions[index]
            prefix = actions[:index]
            prefix_replay = self.adapter.replay(task_id=task_id, prefix_actions=prefix, seed=seed)
            suffix = tuple(str(action) for action in self.adapter.policy_commands(prefix_replay.state))
            if not suffix:
                continue
            candidates = replacement_candidates(
                step=step,
                original_action=action,
                limit=self.config.candidate_branch_limit,
            )
            matches: list[tuple[int, str]] = []
            for replacement in candidates:
                try:
                    suffix_index = suffix.index(replacement)
                except ValueError:
                    continue
                matches.append((suffix_index, replacement))
            matches.sort(reverse=True)
            for suffix_index, replacement in matches[: self.config.suffix_match_limit_per_step]:
                repair_suffix = suffix[suffix_index:]
                edited_actions = prefix + repair_suffix
                outcome = self.replay(task_id=task_id, seed=seed, actions=edited_actions)
                equivalent = self._equivalent(outcome.final_signature, original.final_signature)
                labels = suffix_match_repair_labels(
                    original_success=original_success,
                    outcome_success=outcome.success,
                    suffix_index=suffix_index,
                    final_equivalent=equivalent,
                )
                if not labels:
                    continue
                records.append(
                    TraceEditRecord(
                        edit_type="suffix_match_policy",
                        index=index,
                        action=action,
                        labels=labels,
                        replacement=replacement,
                        repair_suffix=repair_suffix[1:],
                        edited_actions=edited_actions,
                        valid=outcome.valid,
                        success=outcome.success,
                        total_reward=outcome.total_reward,
                        final_equivalent_to_original=equivalent,
                    )
                )
        return tuple(records)

    def _probe_candidate_policy_repairs(
        self,
        task_id: str,
        seed: int,
        steps: tuple[Mapping[str, Any], ...],
        actions: tuple[str, ...],
        original: TraceReplayOutcome,
        original_success: bool,
    ) -> tuple[TraceEditRecord, ...]:
        records: list[TraceEditRecord] = []
        for index, step in enumerate(steps):
            action = actions[index]
            prefix = actions[:index]
            candidates = replacement_candidates(
                step=step,
                original_action=action,
                limit=self.config.candidate_branch_limit,
            )
            for replacement in candidates:
                branch_prefix = prefix + (replacement,)
                branch_replay = self.adapter.replay(
                    task_id=task_id,
                    prefix_actions=branch_prefix,
                    seed=seed,
                )
                if not branch_replay.valid:
                    continue
                suffix = tuple(
                    str(action)
                    for action in self.adapter.policy_commands(branch_replay.state)
                )
                if not suffix and not branch_replay.done:
                    continue
                edited_actions = branch_prefix + suffix
                outcome = self._continue_from_replay(
                    replay=branch_replay,
                    actions=edited_actions,
                    suffix=suffix,
                )
                equivalent = self._equivalent(outcome.final_signature, original.final_signature)
                labels = candidate_policy_repair_labels(
                    original_success=original_success,
                    outcome_success=outcome.success,
                    suffix_len=len(suffix),
                    final_equivalent=equivalent,
                )
                if not labels:
                    continue
                records.append(
                    TraceEditRecord(
                        edit_type="candidate_policy_suffix",
                        index=index,
                        action=action,
                        labels=labels,
                        replacement=replacement,
                        repair_suffix=suffix,
                        edited_actions=edited_actions,
                        valid=outcome.valid,
                        success=outcome.success,
                        total_reward=outcome.total_reward,
                        final_equivalent_to_original=equivalent,
                    )
                )
        return tuple(records)

    def _continue_from_replay(
        self,
        replay: Any,
        actions: tuple[str, ...],
        suffix: tuple[str, ...],
    ) -> TraceReplayOutcome:
        steps = list(getattr(replay, "steps", ()) or ())
        current_state = replay.state
        done = bool(replay.done)
        valid = bool(replay.valid)
        for action in suffix:
            if done:
                break
            step = self.adapter.step(action)
            steps.append(step)
            current_state = step.state
            done = bool(step.done)
            valid = valid and bool(step.valid)
        return TraceReplayOutcome(
            actions=actions,
            valid=valid,
            success=replay_success(current_state, done),
            done=done,
            total_reward=sum(step.reward for step in steps),
            final_signature=self.adapter.signature(
                current_state,
                mode=self.config.equivalence_mode,
            ),
        )

    def _equivalent(self, sig1: Hashable, sig2: Hashable) -> bool:
        return self.adapter.is_equivalent(sig1, sig2, mode=self.config.equivalence_mode)


def replay_success(state: Any, done: bool) -> bool:
    won = getattr(state, "won", None)
    if won is not None:
        return bool(won)
    return bool(done)


def deletion_labels(
    original_success: bool,
    outcome_success: bool,
    final_equivalent: bool,
) -> tuple[str, ...]:
    if original_success:
        if outcome_success and final_equivalent:
            return ("redundant",)
        if outcome_success:
            return ("success_preserving_state_change",)
        return ("necessary",)
    if outcome_success:
        return ("harmful", "delete_repairs_failure")
    return ("no_repair",)


def swap_labels(
    original_success: bool,
    outcome_success: bool,
    final_equivalent: bool,
) -> tuple[str, ...]:
    if outcome_success and final_equivalent:
        return ("order_invariant",)
    if original_success and not outcome_success:
        return ("order_critical",)
    if original_success:
        return ("order_sensitive",)
    if outcome_success:
        return ("repairing_swap",)
    return ("swap_no_repair",)


def replacement_labels(
    original_success: bool,
    outcome_success: bool,
    final_equivalent: bool,
) -> tuple[str, ...]:
    if original_success:
        if outcome_success and final_equivalent:
            return ("equivalent_replacement",)
        if outcome_success:
            return ("success_preserving_replacement",)
        return ("replacement_breaks_success",)
    if outcome_success:
        return ("repair_candidate",)
    return ("replacement_no_repair",)


def policy_repair_labels(
    original_success: bool,
    outcome_success: bool,
    prefix_len: int,
    original_len: int,
    suffix_len: int,
    final_equivalent: bool,
) -> tuple[str, ...]:
    if not outcome_success:
        return ()
    if original_success:
        if prefix_len == original_len and suffix_len == 0:
            return ()
        if prefix_len + suffix_len < original_len and final_equivalent:
            return ("shorter_policy_completion",)
        return ("policy_completion",)
    return ("policy_repair_candidate",)


def candidate_policy_repair_labels(
    original_success: bool,
    outcome_success: bool,
    suffix_len: int,
    final_equivalent: bool,
) -> tuple[str, ...]:
    if not outcome_success:
        return ()
    if original_success:
        if suffix_len == 0 and final_equivalent:
            return ("candidate_shorter_policy_completion",)
        return ("candidate_policy_completion",)
    return ("candidate_policy_repair",)


def suffix_match_repair_labels(
    original_success: bool,
    outcome_success: bool,
    suffix_index: int,
    final_equivalent: bool,
) -> tuple[str, ...]:
    if not outcome_success:
        return ()
    if original_success:
        if suffix_index == 0 and final_equivalent:
            return ("suffix_match_completion",)
        return ("suffix_match_success_preserving",)
    return ("suffix_match_repair",)


def replacement_candidates(
    step: Mapping[str, Any],
    original_action: str,
    limit: int,
) -> tuple[str, ...]:
    raw = step.get("candidates_before") or step.get("candidates") or ()
    candidates: list[str] = []
    for item in raw:
        action = str(item)
        if action == original_action or action in candidates:
            continue
        candidates.append(action)
        if len(candidates) >= limit:
            break
    return tuple(candidates)


def summarize_actions(
    actions: tuple[str, ...],
    edits: tuple[TraceEditRecord, ...],
) -> tuple[TraceActionSummary, ...]:
    labels_by_index: dict[int, set[str]] = {index: set() for index in range(len(actions))}
    repairs_by_index: dict[int, list[str]] = {index: [] for index in range(len(actions))}
    for edit in edits:
        labels_by_index.setdefault(edit.index, set()).update(edit.labels)
        if edit.second_index is not None:
            labels_by_index.setdefault(edit.second_index, set()).update(edit.labels)
        if "repair_candidate" in edit.labels and edit.replacement is not None:
            repairs_by_index.setdefault(edit.index, []).append(edit.replacement)
        if "policy_repair_candidate" in edit.labels and edit.repair_suffix:
            repairs_by_index.setdefault(edit.index, []).append(" ; ".join(edit.repair_suffix))
        if "candidate_policy_repair" in edit.labels and edit.replacement is not None:
            repairs_by_index.setdefault(edit.index, []).append(edit.replacement)
        if "suffix_match_repair" in edit.labels and edit.replacement is not None:
            repairs_by_index.setdefault(edit.index, []).append(edit.replacement)

    return tuple(
        TraceActionSummary(
            index=index,
            action=action,
            labels=tuple(sorted(labels_by_index.get(index, set()))),
            repair_replacements=tuple(repairs_by_index.get(index, ())),
        )
        for index, action in enumerate(actions)
    )


def trace_metrics(edits: tuple[TraceEditRecord, ...], action_count: int) -> dict[str, float]:
    labels = [label for edit in edits for label in edit.labels]
    return {
        "actions": float(action_count),
        "edits": float(len(edits)),
        "redundant_actions": float(count_delete_label(edits, "redundant")),
        "necessary_actions": float(count_delete_label(edits, "necessary")),
        "harmful_actions": float(count_delete_label(edits, "harmful")),
        "repair_candidates": float(labels.count("repair_candidate")),
        "policy_repair_candidates": float(labels.count("policy_repair_candidate")),
        "candidate_policy_repair_candidates": float(
            labels.count("candidate_policy_repair")
        ),
        "suffix_match_repair_candidates": float(labels.count("suffix_match_repair")),
        "policy_completions": float(labels.count("policy_completion")),
        "shorter_policy_completions": float(labels.count("shorter_policy_completion")),
        "order_invariant_pairs": float(labels.count("order_invariant")),
        "order_critical_pairs": float(labels.count("order_critical")),
    }


def count_delete_label(edits: tuple[TraceEditRecord, ...], label: str) -> int:
    return sum(1 for edit in edits if edit.edit_type == "delete" and label in edit.labels)
