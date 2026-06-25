from __future__ import annotations

from acta.agents.base import Agent, AgentContext, CandidateAction


class NoisyCandidateAgent:
    """Wrap an agent and perturb candidate ordering for robustness tests."""

    def __init__(
        self,
        base_agent: Agent,
        mode: str = "frontload-structural",
        max_candidates: int | None = None,
    ) -> None:
        self.base_agent = base_agent
        self.mode = mode
        self.max_candidates = max_candidates

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        base_candidates = list(self.base_agent.candidates(context))
        if self.mode == "none":
            return tuple(base_candidates)
        if self.mode == "frontload-existing-structural":
            candidates = base_candidates
        elif self.mode == "frontload-structural":
            candidates = inject_structural_distractors(base_candidates, context)
        else:
            raise ValueError(f"unknown noise mode: {self.mode}")

        candidates = sorted(
            candidates,
            key=lambda candidate: (
                -structural_distractor_rank(candidate.action, context),
                candidate.action,
            ),
        )
        if self.max_candidates is not None:
            candidates = candidates[: self.max_candidates]
        return tuple(
            CandidateAction(
                action=candidate.action,
                score=float(len(candidates) - index),
                source=f"noisy:{candidate.source}",
                metadata=dict(candidate.metadata),
            )
            for index, candidate in enumerate(candidates)
        )


def inject_structural_distractors(
    candidates: list[CandidateAction],
    context: AgentContext,
) -> list[CandidateAction]:
    seen = {candidate.action for candidate in candidates}
    injected = list(candidates)
    for action in context.admissible_actions:
        if action in seen:
            continue
        if action in {"look", "inventory"} or action in context.history[-4:]:
            injected.append(CandidateAction(action=action, score=0.0, source="noise"))
            seen.add(action)
    return injected


def structural_distractor_rank(action: str, context: AgentContext) -> int:
    if action in context.history[-4:]:
        return 3
    if action in {"look", "inventory"}:
        return 2
    if action.startswith("examine "):
        return 1
    return 0
