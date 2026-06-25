from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from recap.agents import AgentContext, CandidateAction
from recap.controllers.prior_controller import ControllerDecision
from recap.envs.base import EnvAdapter


@dataclass(frozen=True)
class ReplayRepairConfig:
    max_suffix_steps: int = 12
    success_bonus: float = 1000.0
    suffix_penalty: float = 1.0
    max_proposals_to_verify: int | None = None
    require_raw_failure: bool = False
    min_suffix_improvement: int = 1


class ReplayRepairController:
    """Online ReCAP upper-bound controller using branch replay over logged candidates.

    The controller does not create actions. It tests each logged candidate by
    replaying the current prefix, applying the candidate, and then following the
    environment suffix policy when available. Candidates whose branch reaches
    success are promoted, with shorter verified suffixes preferred.
    """

    def __init__(
        self,
        adapter: EnvAdapter,
        config: ReplayRepairConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or ReplayRepairConfig()
        self.cache_hits: dict[str, int] = {"branch": 0}
        self.cache_misses: dict[str, int] = {"branch": 0}
        self._cache: dict[tuple[str, int, tuple[str, ...], str], tuple[bool, int]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = {"branch": 0}
        self.cache_misses = {"branch": 0}

    def rerank(
        self,
        context: AgentContext,
        candidates: Sequence[CandidateAction],
    ) -> ControllerDecision:
        scored: list[CandidateAction] = []
        reasons: dict[str, tuple[str, ...]] = {}
        for candidate in candidates:
            success, suffix_len = self._branch_success(context, candidate.action)
            score = candidate.score
            candidate_reasons: list[str] = []
            if success:
                score += self.config.success_bonus - self.config.suffix_penalty * suffix_len
                candidate_reasons.append(f"replay_success_suffix_{suffix_len}")
            metadata = dict(candidate.metadata)
            metadata["recap_replay_success"] = success
            metadata["recap_replay_suffix_len"] = suffix_len
            scored.append(
                CandidateAction(
                    action=candidate.action,
                    score=score,
                    source=candidate.source,
                    metadata=metadata,
                )
            )
            reasons[candidate.action] = tuple(candidate_reasons)
        scored.sort(key=lambda item: (-item.score, item.action))
        return ControllerDecision(candidates=tuple(scored), reasons=reasons)

    def _branch_success(self, context: AgentContext, action: str) -> tuple[bool, int]:
        key = (context.task_id, context.seed, tuple(context.history), action)
        if key in self._cache:
            self.cache_hits["branch"] += 1
            return self._cache[key]
        self.cache_misses["branch"] += 1
        replay = self.adapter.replay(
            task_id=context.task_id,
            seed=context.seed,
            prefix_actions=context.history + (action,),
        )
        if not replay.valid:
            result = (False, self.config.max_suffix_steps + 1)
            self._cache[key] = result
            return result
        if replay.done:
            result = (True, 0)
            self._cache[key] = result
            return result
        state = replay.state
        suffix_len = 0
        success = False
        while suffix_len < self.config.max_suffix_steps:
            suffix = tuple(self.adapter.policy_commands(state))
            if not suffix:
                break
            step = self.adapter.step(suffix[0])
            suffix_len += 1
            state = step.state
            if not step.valid:
                break
            if step.done:
                success = True
                break
        result = (success, suffix_len if success else self.config.max_suffix_steps + 1)
        self._cache[key] = result
        return result


class ReplayVerifiedProposalController(ReplayRepairController):
    """Verify learned reranker proposals before changing the raw action.

    The full replay controller evaluates every logged candidate and therefore
    behaves like an online upper bound when a suffix policy is available. This
    controller is the deployment-oriented variant: the learned reranker first
    orders candidates; only the first few non-raw proposals are branch-verified;
    if none is certified, execution falls back to the raw top-1 action.
    """

    def rerank(
        self,
        context: AgentContext,
        candidates: Sequence[CandidateAction],
    ) -> ControllerDecision:
        if not candidates:
            return ControllerDecision(candidates=(), reasons={})

        raw_order = recover_raw_order(tuple(candidates))
        raw_top = raw_order[0]
        raw_success, raw_suffix_len = self._branch_success(context, raw_top.action)

        proposal_budget = self.config.max_proposals_to_verify
        if proposal_budget is None:
            proposal_budget = len(candidates)
        proposals = [
            candidate
            for candidate in candidates
            if candidate.action != raw_top.action
        ][: max(0, proposal_budget)]

        reasons: dict[str, tuple[str, ...]] = {candidate.action: () for candidate in raw_order}
        certified: list[tuple[int, CandidateAction]] = []
        for proposal in proposals:
            success, suffix_len = self._branch_success(context, proposal.action)
            metadata = dict(proposal.metadata)
            metadata["recap_verified_proposal_success"] = success
            metadata["recap_verified_proposal_suffix_len"] = suffix_len
            proposal = CandidateAction(
                action=proposal.action,
                score=proposal.score,
                source=proposal.source,
                metadata=metadata,
            )
            improves_raw = (
                not raw_success
                or suffix_len + self.config.min_suffix_improvement <= raw_suffix_len
            )
            if success and improves_raw and not (self.config.require_raw_failure and raw_success):
                certified.append((suffix_len, proposal))
                reasons[proposal.action] = (f"verified_proposal_suffix_{suffix_len}",)

        if certified:
            certified.sort(key=lambda item: (item[0], -item[1].score, item[1].action))
            selected = certified[0][1]
            remaining = [
                candidate
                for candidate in raw_order
                if candidate.action != selected.action
            ]
            return ControllerDecision(candidates=(selected, *remaining), reasons=reasons)

        fallback_metadata = dict(raw_top.metadata)
        fallback_metadata["recap_verified_proposal_fallback"] = True
        fallback_metadata["recap_verified_raw_success"] = raw_success
        fallback_metadata["recap_verified_raw_suffix_len"] = raw_suffix_len
        fallback = CandidateAction(
            action=raw_top.action,
            score=raw_top.score,
            source=raw_top.source,
            metadata=fallback_metadata,
        )
        remaining = [
            candidate
            for candidate in raw_order[1:]
            if candidate.action != fallback.action
        ]
        reasons[fallback.action] = ("fallback_raw",)
        return ControllerDecision(candidates=(fallback, *remaining), reasons=reasons)


def recover_raw_order(candidates: tuple[CandidateAction, ...]) -> tuple[CandidateAction, ...]:
    if not any("recap_raw_rank" in candidate.metadata for candidate in candidates):
        return candidates
    missing_offset = len(candidates) + 1
    indexed: list[tuple[int, int, CandidateAction]] = []
    for index, candidate in enumerate(candidates):
        rank = candidate.metadata.get("recap_raw_rank")
        raw_rank = int(rank) if isinstance(rank, int | float) else missing_offset + index
        indexed.append((raw_rank, index, candidate))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return tuple(candidate for _rank, _index, candidate in indexed)
