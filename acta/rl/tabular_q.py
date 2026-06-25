from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Hashable, Literal, Mapping

from acta.agents import AgentContext, CandidateAction
from acta.controllers import ActAController
from acta.envs.base import EnvAdapter


ActAPrior = Literal["none", "acta-soft", "acta-hard"]


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 100
    max_steps: int = 30
    alpha: float = 0.5
    gamma: float = 0.95
    epsilon: float = 0.2
    epsilon_decay: float = 1.0
    min_epsilon: float = 0.05
    seed: int = 0
    prior: ActAPrior = "none"
    equivalence_mode: str = "full"


@dataclass(frozen=True)
class StepTrace:
    step_index: int
    state_signature: Hashable
    action: str
    reward: float
    done: bool
    valid: bool
    q_before: float
    q_after: float
    candidates: tuple[str, ...]
    acta_reasons: tuple[str, ...] = ()
    blocked_actions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeTrace:
    task_id: str
    seed: int
    episode_index: int
    success: bool
    total_reward: float
    steps: tuple[StepTrace, ...]
    epsilon: float


def train_q_learning(
    adapter: EnvAdapter,
    task_id: str,
    config: QLearningConfig,
    controller: ActAController | None = None,
) -> tuple[EpisodeTrace, ...]:
    if config.prior != "none" and controller is None:
        raise ValueError("controller is required when using an ActA prior")

    rng = random.Random(config.seed)
    q_values: dict[tuple[Hashable, str], float] = {}
    traces: list[EpisodeTrace] = []
    epsilon = config.epsilon

    for episode_index in range(config.episodes):
        trace = run_training_episode(
            adapter=adapter,
            task_id=task_id,
            episode_index=episode_index,
            seed=config.seed,
            q_values=q_values,
            epsilon=epsilon,
            config=config,
            rng=rng,
            controller=controller,
        )
        traces.append(trace)
        epsilon = max(config.min_epsilon, epsilon * config.epsilon_decay)

    return tuple(traces)


def run_training_episode(
    adapter: EnvAdapter,
    task_id: str,
    episode_index: int,
    seed: int,
    q_values: dict[tuple[Hashable, str], float],
    epsilon: float,
    config: QLearningConfig,
    rng: random.Random,
    controller: ActAController | None,
) -> EpisodeTrace:
    reset = adapter.reset(task_id=task_id, seed=seed)
    state = reset.state
    observation = reset.observation
    history: list[str] = []
    seen_signatures: list[Hashable] = [adapter.signature(state, mode=config.equivalence_mode)]
    total_reward = 0.0
    done = reset.done
    steps: list[StepTrace] = []

    for step_index in range(config.max_steps):
        if done:
            break

        state_signature = adapter.signature(state, mode=config.equivalence_mode)
        admissible = tuple(adapter.admissible_actions(state))
        if not admissible:
            break

        context = AgentContext(
            task_id=task_id,
            seed=seed,
            step_index=step_index,
            observation=observation,
            admissible_actions=admissible,
            history=tuple(history),
            state_signature=state_signature,
            seen_signatures=tuple(seen_signatures),
            initial_observation=reset.observation,
        )
        choice = choose_action(
            q_values=q_values,
            state_signature=state_signature,
            actions=admissible,
            context=context,
            config=config,
            rng=rng,
            controller=controller,
            epsilon=epsilon,
        )

        # Controller probes use replay and mutate the adapter; restore the
        # current training state before applying the selected action.
        restored = adapter.replay(task_id=task_id, prefix_actions=tuple(history), seed=seed)
        state = restored.state
        observation = restored.observation
        q_before = q_values.get((state_signature, choice.action), 0.0)
        step = adapter.step(choice.action)
        next_signature = adapter.signature(step.state, mode=config.equivalence_mode)
        next_actions = tuple(adapter.admissible_actions(step.state))
        next_value = (
            max(q_values.get((next_signature, action), 0.0) for action in next_actions)
            if next_actions and not step.done
            else 0.0
        )
        q_after = q_before + config.alpha * (
            step.reward + config.gamma * next_value - q_before
        )
        q_values[(state_signature, choice.action)] = q_after

        total_reward += step.reward
        history.append(choice.action)
        state = step.state
        observation = step.observation
        done = step.done
        seen_signatures.append(next_signature)
        steps.append(
            StepTrace(
                step_index=step_index,
                state_signature=state_signature,
                action=choice.action,
                reward=step.reward,
                done=step.done,
                valid=step.valid,
                q_before=q_before,
                q_after=q_after,
                candidates=choice.candidates,
                acta_reasons=choice.reasons,
                blocked_actions=choice.blocked_actions,
            )
        )

    return EpisodeTrace(
        task_id=task_id,
        seed=seed,
        episode_index=episode_index,
        success=done,
        total_reward=total_reward,
        steps=tuple(steps),
        epsilon=epsilon,
    )


@dataclass(frozen=True)
class ActionChoice:
    action: str
    candidates: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    blocked_actions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def choose_action(
    q_values: dict[tuple[Hashable, str], float],
    state_signature: Hashable,
    actions: tuple[str, ...],
    context: AgentContext,
    config: QLearningConfig,
    rng: random.Random,
    controller: ActAController | None,
    epsilon: float,
) -> ActionChoice:
    if config.prior == "none":
        action = epsilon_greedy_raw(
            q_values=q_values,
            state_signature=state_signature,
            actions=actions,
            rng=rng,
            epsilon=epsilon,
        )
        return ActionChoice(action=action, candidates=actions)

    assert controller is not None
    candidates = tuple(
        CandidateAction(
            action=action,
            score=q_values.get((state_signature, action), 0.0),
            source="q",
        )
        for action in actions
    )
    decision = controller.rerank(context, candidates)
    blocked = {
        action: reasons
        for action, reasons in decision.reasons.items()
        if reasons
    }

    if config.prior == "acta-soft":
        if rng.random() < epsilon:
            action = rng.choice(actions)
        else:
            best_score = max(candidate.score for candidate in decision.candidates)
            best = [
                candidate.action
                for candidate in decision.candidates
                if candidate.score == best_score
            ]
            action = rng.choice(best)
        return ActionChoice(
            action=action,
            candidates=tuple(candidate.action for candidate in decision.candidates),
            reasons=decision.reasons.get(action, ()),
            blocked_actions=blocked,
        )

    if config.prior == "acta-hard":
        allowed = tuple(action for action in actions if action not in blocked)
        action_pool = allowed or actions
        if rng.random() < epsilon:
            action = rng.choice(action_pool)
        else:
            action = choose_max_q(q_values, state_signature, action_pool, rng)
        return ActionChoice(
            action=action,
            candidates=action_pool,
            reasons=decision.reasons.get(action, ()),
            blocked_actions=blocked,
        )

    raise ValueError(f"unknown prior: {config.prior}")


def epsilon_greedy_raw(
    q_values: dict[tuple[Hashable, str], float],
    state_signature: Hashable,
    actions: tuple[str, ...],
    rng: random.Random,
    epsilon: float,
) -> str:
    if rng.random() < epsilon:
        return rng.choice(actions)
    return choose_max_q(q_values, state_signature, actions, rng)


def choose_max_q(
    q_values: dict[tuple[Hashable, str], float],
    state_signature: Hashable,
    actions: tuple[str, ...],
    rng: random.Random,
) -> str:
    best_value = max(q_values.get((state_signature, action), 0.0) for action in actions)
    best_actions = [
        action for action in actions if q_values.get((state_signature, action), 0.0) == best_value
    ]
    return rng.choice(best_actions)
