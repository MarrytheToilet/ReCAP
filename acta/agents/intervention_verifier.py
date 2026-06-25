from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from acta.agents.base import AgentContext
from acta.agents.llm_agent import (
    load_response_cache,
    response_format_unsupported,
    save_response_cache,
    should_retry,
    sleep_for_retry,
)


VERIFY_SYSTEM_PROMPT = """You are a conservative TextWorld action selector.
You are given the current task context and two candidate next actions:
RAW is the action the base agent would execute.
PROPOSED is the action selected by a ReCAP reranker trained from
replay-certified repairs.

Choose PROPOSED when RAW looks like a loop, repeated navigation, repeated
inspection, inventory/checking behavior, or another action that is unlikely to
make progress, and PROPOSED offers a plausible new route or task-progressing
action. Choose RAW when RAW is a concrete progress action such as taking a new
needed object, opening/unlocking/locking/inserting with a relevant object, or
when PROPOSED appears to undo recent progress or simply oscillate.

If both actions are equally plausible, prefer the action that better breaks
repetition while preserving task progress.

Return JSON only: {"choice": "raw"} or {"choice": "proposed"}."""


@dataclass(frozen=True)
class VerifierConfig:
    model: str
    temperature: float = 0.0
    timeout: float | None = None
    max_retries: int = 3
    retry_base_delay: float = 2.0
    call_delay: float = 0.0
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    cache_path: Path | None = None
    system_prompt: str = VERIFY_SYSTEM_PROMPT


class LLMInterventionVerifier:
    def __init__(self, config: VerifierConfig) -> None:
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the `openai` package to use LLMInterventionVerifier.") from exc

        api_key = os.environ.get(config.api_key_env)
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

    def allow(
        self,
        context: AgentContext,
        raw_action: str,
        proposed_action: str,
        candidates: tuple[str, ...],
    ) -> tuple[bool, str]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": build_verify_prompt(
                    context=context,
                    raw_action=raw_action,
                    proposed_action=proposed_action,
                    candidates=candidates,
                ),
            },
        ]
        cache_key = verifier_cache_key(context, raw_action, proposed_action, candidates, self.config)
        if cache_key in self._response_cache:
            content = self._response_cache[cache_key]
        else:
            self._throttle()
            response = self._create_completion(messages)
            content = response.choices[0].message.content or ""
            self._response_cache[cache_key] = content
            save_response_cache(self.cache_path, self._response_cache)
        choice = parse_choice(content)
        return choice == "proposed", choice

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
        raise RuntimeError("verifier completion failed without an exception")


class StaticInterventionVerifier:
    def __init__(self, allow: bool, reason: str = "static") -> None:
        self._allow = allow
        self._reason = reason

    def allow(
        self,
        context: AgentContext,
        raw_action: str,
        proposed_action: str,
        candidates: tuple[str, ...],
    ) -> tuple[bool, str]:
        return self._allow, self._reason


def build_verify_prompt(
    context: AgentContext,
    raw_action: str,
    proposed_action: str,
    candidates: tuple[str, ...],
) -> str:
    history = "\n".join(context.history[-12:]) or "<empty>"
    actions = "\n".join(f"- {action}" for action in candidates)
    initial = context.initial_observation or context.observation
    return (
        f"Task: {context.task_id}\n"
        f"Step: {context.step_index}\n"
        f"Initial observation and objective:\n{initial}\n\n"
        f"Recent action history:\n{history}\n\n"
        f"Current observation:\n{context.observation}\n\n"
        f"Logged candidate list:\n{actions}\n\n"
        f"RAW action: {raw_action}\n"
        f"PROPOSED action: {proposed_action}\n\n"
        "Choose whether to keep RAW or execute PROPOSED."
    )


def parse_choice(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = content
    if isinstance(payload, Mapping):
        choice = str(payload.get("choice", "")).strip().lower()
    else:
        choice = str(payload).strip().lower()
    if "proposed" in choice:
        return "proposed"
    return "raw"


def verifier_cache_key(
    context: AgentContext,
    raw_action: str,
    proposed_action: str,
    candidates: tuple[str, ...],
    config: VerifierConfig,
) -> str:
    payload = {
        "task_id": context.task_id,
        "seed": context.seed,
        "step_index": context.step_index,
        "history": list(context.history),
        "observation": context.observation,
        "candidates": list(candidates),
        "raw_action": raw_action,
        "proposed_action": proposed_action,
        "model": config.model,
        "temperature": config.temperature,
        "system_prompt": config.system_prompt,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
