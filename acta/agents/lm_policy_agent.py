from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from acta.agents.base import AgentContext, CandidateAction
from acta.agents.llm_agent import rank_admissible_actions
from acta.models.lm_candidate_policy import (
    candidate_scores,
    encode_candidate_batch,
    yes_no_token_ids,
)


class LocalLMPolicyAgent:
    """Rank admissible actions with a local causal LM policy.

    The model scores each admissible action with the same yes/no candidate
    prompt used by ReCAP-SCPO. A LoRA adapter turns the base LM into the
    reward-optimized policy used for online agent evaluation.
    """

    def __init__(
        self,
        base_model: Path,
        adapter: Path | None = None,
        max_candidates: int = 5,
        candidate_pool_limit: int = 20,
        max_length: int = 384,
        max_history: int = 12,
        max_observation_chars: int = 220,
        candidate_chunk_size: int = 2,
        recent_repeat_penalty: float = 0.0,
        inverse_penalty: float = 0.0,
        static_penalty: float = 0.0,
        semantic_undo_penalty: float = 0.0,
        objective_overlap_bonus: float = 0.0,
        navigation_bonus: float = 0.0,
        nonobjective_manipulation_penalty: float = 0.0,
        pool_ranker: str = "default",
        device: str = "cuda",
        load_in_4bit: bool = True,
        model_type: str = "local_lm_policy",
    ) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_candidates = max_candidates
        self.candidate_pool_limit = candidate_pool_limit
        self.max_length = max_length
        self.max_history = max_history
        self.max_observation_chars = max_observation_chars
        self.candidate_chunk_size = candidate_chunk_size
        self.recent_repeat_penalty = recent_repeat_penalty
        self.inverse_penalty = inverse_penalty
        self.static_penalty = static_penalty
        self.semantic_undo_penalty = semantic_undo_penalty
        self.objective_overlap_bonus = objective_overlap_bonus
        self.navigation_bonus = navigation_bonus
        self.nonobjective_manipulation_penalty = nonobjective_manipulation_penalty
        self.pool_ranker = pool_ranker
        self.device = device
        self.model_type = model_type

        tokenizer_source = adapter or base_model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_source),
                local_files_only=True,
                trust_remote_code=True,
            )
        except (OSError, TypeError, ValueError):
            if adapter is None:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(base_model),
                local_files_only=True,
                trust_remote_code=True,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(str(base_model), **model_kwargs)
        if adapter is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter),
                local_files_only=True,
            )
        if not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.yes_id, self.no_id = yes_no_token_ids(self.tokenizer)

    def candidates(self, context: AgentContext) -> tuple[CandidateAction, ...]:
        if not context.admissible_actions:
            return ()
        pool = tuple(
            rank_lm_policy_pool(
                context.admissible_actions,
                context.history,
                objective=context.initial_observation,
                mode=self.pool_ranker,
            )[
                : self.candidate_pool_limit
            ]
        )
        record = {
            "task_id": context.task_id,
            "seed": context.seed,
            "step_index": context.step_index,
            "history": tuple(context.history),
            "observation": context.observation,
            "initial_observation": context.initial_observation,
            "candidates": pool,
        }
        batch = encode_candidate_batch(
            self.tokenizer,
            record,
            max_length=self.max_length,
            max_history=self.max_history,
            max_observation_chars=self.max_observation_chars,
            device=self.device,
        )
        with torch.no_grad():
            scores_tensor = candidate_scores(
                self.model,
                batch,
                yes_id=self.yes_id,
                no_id=self.no_id,
                chunk_size=self.candidate_chunk_size,
            )
        raw_scores = scores_tensor.detach().float().cpu().tolist()
        adjusted_scores = [
            score - structural_penalty(
                action,
                context.history,
                objective=context.initial_observation,
                recent_repeat_penalty=self.recent_repeat_penalty,
                inverse_penalty=self.inverse_penalty,
                static_penalty=self.static_penalty,
                semantic_undo_penalty=self.semantic_undo_penalty,
                objective_overlap_bonus=self.objective_overlap_bonus,
                navigation_bonus=self.navigation_bonus,
                nonobjective_manipulation_penalty=self.nonobjective_manipulation_penalty,
            )
            for action, score in zip(pool, raw_scores)
        ]
        indexed = list(enumerate(zip(pool, adjusted_scores, raw_scores)))
        indexed.sort(key=lambda item: (-item[1][1], item[0]))
        selected = indexed[: self.max_candidates]
        return tuple(
            CandidateAction(
                action=action,
                score=float(score),
                source=self.model_type,
                metadata={
                    "lm_policy_score": float(score),
                    "lm_policy_raw_score": float(raw_score),
                    "lm_policy_model_type": self.model_type,
                    "lm_policy_pool_rank": original_index + 1,
                },
            )
            for original_index, (action, score, raw_score) in selected
        )


def structural_penalty(
    action: str,
    history: tuple[str, ...],
    objective: str = "",
    recent_repeat_penalty: float = 0.0,
    inverse_penalty: float = 0.0,
    static_penalty: float = 0.0,
    semantic_undo_penalty: float = 0.0,
    objective_overlap_bonus: float = 0.0,
    navigation_bonus: float = 0.0,
    nonobjective_manipulation_penalty: float = 0.0,
) -> float:
    penalty = 0.0
    if recent_repeat_penalty and action in history[-4:]:
        penalty += recent_repeat_penalty
    if inverse_penalty and history and is_inverse_navigation(history[-1], action):
        penalty += inverse_penalty
    if static_penalty and action.split()[:1] in (["look"], ["inventory"], ["examine"]):
        penalty += static_penalty
    if semantic_undo_penalty and any(is_semantic_undo(previous, action) for previous in history[-3:]):
        penalty += semantic_undo_penalty
    overlap = objective_action_overlap(action, objective)
    if nonobjective_manipulation_penalty and is_manipulation(action) and overlap <= 0.0:
        penalty += nonobjective_manipulation_penalty
    if objective_overlap_bonus:
        penalty -= objective_overlap_bonus * overlap
    if navigation_bonus and action.startswith("go "):
        penalty -= navigation_bonus
    return penalty


def rank_lm_policy_pool(
    admissible_actions: tuple[str, ...],
    history: tuple[str, ...],
    objective: str = "",
    mode: str = "default",
) -> list[str]:
    if mode == "default":
        return rank_admissible_actions(admissible_actions, history)
    if mode != "progress":
        raise ValueError(f"unknown LM policy pool ranker: {mode}")
    indexed: list[tuple[float, int, str]] = []
    for index, action in enumerate(admissible_actions):
        score = pool_priority(action, history, objective)
        indexed.append((score, index, action))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [action for _score, _index, action in indexed]


def pool_priority(action: str, history: tuple[str, ...], objective: str = "") -> float:
    verb = action_verb(action)
    priority = {
        "go": 0.0,
        "open": 1.0,
        "unlock": 1.0,
        "take": 1.4,
        "put": 2.0,
        "insert": 2.0,
        "drop": 2.2,
        "eat": 2.2,
        "close": 2.4,
        "lock": 2.4,
        "read": 3.0,
        "look": 5.0,
        "inventory": 5.0,
        "examine": 5.0,
    }.get(verb, 3.2)
    if action in history[-4:]:
        priority += 4.0
    if history and is_inverse_navigation(history[-1], action):
        priority += 3.0
    if any(is_semantic_undo(previous, action) for previous in history[-3:]):
        priority += 3.0
    priority -= objective_action_overlap(action, objective)
    return priority


def objective_action_overlap(action: str, objective: str) -> float:
    if not objective:
        return 0.0
    objective_norm = normalize_text(objective)
    action_tokens = [token for token in normalize_text(action).split() if len(token) > 2]
    if not action_tokens:
        return 0.0
    overlap = sum(1 for token in action_tokens if token in objective_norm)
    bonus = min(overlap / max(len(action_tokens), 1), 1.0)
    if action.startswith("go "):
        direction = action.split(" ", 1)[1]
        direction_aliases = {
            "north": ("north",),
            "south": ("south",),
            "east": ("east",),
            "west": ("west",),
            "up": ("up", "upstairs"),
            "down": ("down", "downstairs"),
        }
        if any(alias in objective_norm for alias in direction_aliases.get(direction, (direction,))):
            bonus = max(bonus, 1.0)
    return bonus


def normalize_text(text: str) -> str:
    lowered = str(text).lower()
    return "".join(char if char.isalnum() or char.isspace() else " " for char in lowered)


def action_verb(action: str) -> str:
    return str(action).split(" ", 1)[0].lower()


def is_manipulation(action: str) -> bool:
    return action_verb(action) in {
        "take",
        "drop",
        "put",
        "open",
        "close",
        "unlock",
        "lock",
        "insert",
        "eat",
    }


def is_inverse_navigation(previous: str, current: str) -> bool:
    inverse = {
        "go north": "go south",
        "go south": "go north",
        "go east": "go west",
        "go west": "go east",
        "go up": "go down",
        "go down": "go up",
    }
    return inverse.get(previous) == current


def is_semantic_undo(previous: str, current: str) -> bool:
    previous = str(previous).lower()
    current = str(current).lower()
    if is_inverse_navigation(previous, current):
        return True
    previous_verb = action_verb(previous)
    current_verb = action_verb(current)
    if previous_verb == "open" and current_verb == "close":
        return same_tail(previous, current)
    if previous_verb == "close" and current_verb == "open":
        return same_tail(previous, current)
    if previous_verb == "unlock" and current_verb == "lock":
        return shared_object_token(previous, current)
    if previous_verb == "lock" and current_verb == "unlock":
        return shared_object_token(previous, current)
    if previous_verb == "take" and current_verb in {"drop", "put", "insert"}:
        return moved_object(previous) and moved_object(previous) == moved_object(current)
    if previous_verb in {"drop", "put", "insert"} and current_verb == "take":
        return moved_object(previous) and moved_object(previous) == moved_object(current)
    return False


def same_tail(previous: str, current: str) -> bool:
    previous_tail = previous.split(" ", 1)[1:] or [""]
    current_tail = current.split(" ", 1)[1:] or [""]
    return normalize_text(previous_tail[0]).strip() == normalize_text(current_tail[0]).strip()


def moved_object(action: str) -> str:
    normalized = normalize_text(action)
    words = normalized.split()
    if not words:
        return ""
    verb = words[0]
    if verb == "take":
        if "from" in words:
            return " ".join(words[1 : words.index("from")])
        return " ".join(words[1:])
    if verb == "drop":
        return " ".join(words[1:])
    if verb in {"put", "insert"}:
        stop_words = {"on", "in", "into", "onto"}
        for index, word in enumerate(words[1:], start=1):
            if word in stop_words:
                return " ".join(words[1:index])
        return " ".join(words[1:])
    return ""


def shared_object_token(previous: str, current: str) -> bool:
    previous_words = set(normalize_text(previous).split()[1:])
    current_words = set(normalize_text(current).split()[1:])
    stop = {"with", "in", "into", "on", "the", "a", "an"}
    return bool((previous_words - stop) & (current_words - stop))
