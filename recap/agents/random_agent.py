from __future__ import annotations

import random

from recap.agents.base import AgentContext, CandidateAction


class RandomAgent:
    """Random admissible-action baseline.

    It returns a shuffled candidate list instead of only one action so that the
    same agent can be evaluated with or without an ReCAP controller.
    """

    def __init__(self, seed: int = 0, max_candidates: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.max_candidates = max_candidates

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        actions = list(context.admissible_actions)
        self.rng.shuffle(actions)
        if self.max_candidates is not None:
            actions = actions[: self.max_candidates]
        return tuple(
            CandidateAction(action=action, score=0.0, source="random")
            for action in actions
        )

