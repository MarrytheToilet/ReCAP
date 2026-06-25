from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from acta.agents.base import Agent, AgentContext, CandidateAction


class TraceFeedbackAgent:
    """Inject verified trace feedback into an agent's context."""

    def __init__(
        self,
        base_agent: Agent,
        feedback_by_task: Mapping[str, str],
    ) -> None:
        self.base_agent = base_agent
        self.feedback_by_task = dict(feedback_by_task)

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        feedback = self._feedback_for(context.task_id)
        if feedback:
            context = replace(
                context,
                initial_observation=append_feedback(context.initial_observation, feedback),
            )
        return tuple(self.base_agent.candidates(context))

    def _feedback_for(self, task_id: str) -> str:
        return self.feedback_by_task.get(task_id) or self.feedback_by_task.get(Path(task_id).stem, "")


def append_feedback(initial_observation: str, feedback: str) -> str:
    if not initial_observation:
        return feedback
    return f"{initial_observation}\n\n{feedback}"
