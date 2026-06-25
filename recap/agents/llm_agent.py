from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recap.agents.base import AgentContext, CandidateAction


DEFAULT_SYSTEM_PROMPT = """You are a TextWorld agent.
Choose useful next actions from the provided admissible action list.
Prioritize completing the task objective stated in the initial observation.
If the objective mentions a target action such as drop, put, insert, lock,
unlock, open, close, take, eat, or go, include exact admissible actions that
directly satisfy that objective near the top of the list.
Do not prefer undo actions that simply return the world to a previous state
unless the objective explicitly asks for that.
Return JSON only, with the shape {"actions": ["action 1", "action 2", ...]}.
Every returned action must exactly match one admissible action."""


@dataclass(frozen=True)
class LLMConfig:
    model: str
    max_candidates: int = 5
    temperature: float = 0.0
    timeout: float | None = None
    max_retries: int = 4
    retry_base_delay: float = 2.0
    call_delay: float = 0.0
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    cache_path: Path | None = None
    # When True, never call the API: serve cached responses and, on a cache
    # miss, fall back to a heuristic ranking of the environment's admissible
    # actions. Lets cached online runs proceed past divergent states offline.
    cache_only: bool = False


class MockLLMAgent:
    """Local stand-in for the LLM top-k candidate interface."""

    def __init__(self, max_candidates: int = 5) -> None:
        self.max_candidates = max_candidates

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        actions = rank_admissible_actions(context.admissible_actions, context.history)
        return tuple(
            CandidateAction(
                action=action,
                score=float(self.max_candidates - index),
                source="mock-llm",
            )
            for index, action in enumerate(actions[: self.max_candidates])
        )


class OpenAIChatAgent:
    """OpenAI-compatible chat-completions agent that returns top-k candidates."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the `openai` package to use OpenAIChatAgent.") from exc

        api_key = os.environ.get(config.api_key_env)
        if config.cache_only:
            # No live calls: tolerate a missing/expired key.
            self.client = None
        else:
            if not api_key:
                raise RuntimeError(
                    f"{config.api_key_env} is not set. Set it in the environment or an .env file."
                )
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": config.base_url or os.environ.get("OPENAI_BASE_URL"),
            }
            if config.timeout is not None:
                client_kwargs["timeout"] = config.timeout
            self.client = OpenAI(**client_kwargs)
        self._last_call_at: float | None = None
        self.cache_path = config.cache_path
        self._response_cache: dict[str, str] = load_response_cache(config.cache_path)

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": build_candidate_prompt(context, self.config.max_candidates)},
        ]
        cache_key = response_cache_key(context, self.config)
        if cache_key in self._response_cache:
            content = self._response_cache[cache_key]
        elif self.config.cache_only:
            actions = rank_admissible_actions(context.admissible_actions, context.history)
            return tuple(
                CandidateAction(
                    action=action,
                    score=float(self.config.max_candidates - index),
                    source="admissible-fallback",
                )
                for index, action in enumerate(actions[: self.config.max_candidates])
            )
        else:
            self._throttle()
            response = self._create_completion(messages)
            content = response.choices[0].message.content or ""
            self._response_cache[cache_key] = content
            save_response_cache(self.cache_path, self._response_cache)
        actions = parse_action_response(content, context.admissible_actions)
        if not actions:
            actions = rank_admissible_actions(context.admissible_actions, context.history)

        return tuple(
            CandidateAction(
                action=action,
                score=float(self.config.max_candidates - index),
                source="openai-chat",
                metadata={"raw_response": content} if index == 0 else {},
            )
            for index, action in enumerate(actions[: self.config.max_candidates])
        )

    def _throttle(self) -> None:
        if self.config.call_delay <= 0:
            return
        now = time.monotonic()
        if self._last_call_at is not None:
            wait_time = self.config.call_delay - (now - self._last_call_at)
            if wait_time > 0:
                time.sleep(wait_time)
        self._last_call_at = time.monotonic()

    def _create_completion(self, messages: list[dict[str, str]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                if response_format_unsupported(exc):
                    return self.client.chat.completions.create(**kwargs)
                if not should_retry(exc) or attempt >= self.config.max_retries:
                    try:
                        return self.client.chat.completions.create(**kwargs)
                    except Exception as fallback_exc:
                        last_error = fallback_exc
                        break
                sleep_for_retry(attempt, self.config.retry_base_delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("completion failed without an exception")


def build_candidate_prompt(context: AgentContext, max_candidates: int) -> str:
    actions = "\n".join(f"- {action}" for action in context.admissible_actions)
    history = "\n".join(context.history[-12:]) or "<empty>"
    initial = context.initial_observation or context.observation
    return (
        f"Task: {context.task_id}\n"
        f"Step: {context.step_index}\n"
        f"Initial observation and objective:\n{initial}\n\n"
        f"Recent action history:\n{history}\n\n"
        f"Observation:\n{context.observation}\n\n"
        f"Admissible actions:\n{actions}\n\n"
        f"Return up to {max_candidates} candidate actions as JSON."
    )


def parse_action_response(content: str, admissible_actions: tuple[str, ...]) -> list[str]:
    admissible = set(admissible_actions)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = content

    if isinstance(payload, dict):
        raw_actions = payload.get("actions") or payload.get("candidate_actions") or []
    elif isinstance(payload, list):
        raw_actions = payload
    else:
        raw_actions = []

    actions: list[str] = []
    for item in raw_actions:
        action = str(item).strip()
        if action in admissible and action not in actions:
            actions.append(action)
    return actions


def rank_admissible_actions(
    admissible_actions: tuple[str, ...],
    history: tuple[str, ...],
) -> list[str]:
    recent = set(history[-4:])
    preferred = [
        action
        for action in admissible_actions
        if action not in recent and action not in {"look", "inventory"}
    ]
    fallback = [action for action in admissible_actions if action not in preferred]
    return preferred + fallback


def should_retry(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return "rate" in message or "timeout" in message or "temporarily" in message


def response_format_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message and (
        "unsupported" in message or "unknown" in message or "invalid" in message
    )


def sleep_for_retry(attempt: int, base_delay: float) -> None:
    delay = base_delay * (2**attempt)
    time.sleep(delay)


def response_cache_key(context: AgentContext, config: LLMConfig) -> str:
    payload = {
        "task_id": context.task_id,
        "seed": context.seed,
        "step_index": context.step_index,
        "history": list(context.history),
        "observation": context.observation,
        "admissible_actions": list(context.admissible_actions),
        "model": config.model,
        "temperature": config.temperature,
        "max_candidates": config.max_candidates,
        "system_prompt": config.system_prompt,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_response_cache(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def save_response_cache(path: Path | None, cache: dict[str, str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def load_env_file(path: str | Path | None) -> None:
    if path is None:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
