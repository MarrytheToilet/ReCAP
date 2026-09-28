from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def compact_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    stripped = str(text).strip()
    if not stripped:
        return ""
    for marker in (
        "Here is your task",
        "Your task",
        "Objective:",
        "Get ready",
        "It's time",
        "Here is how",
        "First step",
        "First,",
        "First off",
    ):
        index = stripped.find(marker)
        if index >= 0:
            stripped = stripped[index:]
            break
    lines = []
    for line in stripped.splitlines():
        clean = line.strip()
        if not clean:
            continue
        symbol_chars = sum(1 for char in clean if char in "_$\\|/=-")
        if len(clean) >= 20 and symbol_chars / max(1, len(clean)) > 0.55:
            continue
        lines.append(clean)
    text = " ".join(" ".join(lines).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def decision_context(
    record: Mapping[str, Any],
    max_history: int = 12,
    max_observation_chars: int = 360,
) -> str:
    objective = compact_text(str(record.get("initial_observation", "")), max_observation_chars)
    observation = compact_text(str(record.get("observation", "")), max_observation_chars)
    history = [str(item) for item in record.get("history", ())][-max_history:]
    candidates = [str(item) for item in record.get("candidates", ())]
    lines = [
        "You are reranking logged actions for a text-game agent.",
        "Choose only among the logged candidates.",
    ]
    if objective:
        lines.append(f"Task/state excerpt: {objective}")
    if observation:
        lines.append(f"Current observation: {observation}")
    if history:
        lines.append("Recent actions: " + " | ".join(history))
    else:
        lines.append("Recent actions: none")
    if candidates and len(candidates) <= 12:
        lines.append("Logged candidates:")
        lines.extend(f"{index + 1}. {action}" for index, action in enumerate(candidates))
    elif candidates:
        lines.append(f"Logged candidate pool size: {len(candidates)}")
    return "\n".join(lines)


def candidate_prompt(
    record: Mapping[str, Any],
    action: str,
    max_history: int = 12,
    max_observation_chars: int = 360,
) -> str:
    context = decision_context(
        record,
        max_history=max_history,
        max_observation_chars=max_observation_chars,
    )
    return (
        f"{context}\n\n"
        f"Candidate action: {action}\n"
        "Should this candidate be ranked first for the current decision? Answer yes or no.\n"
        "Answer:"
    )


@dataclass(frozen=True)
class CandidateBatch:
    candidates: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def encode_candidate_batch(
    tokenizer: Any,
    record: Mapping[str, Any],
    max_length: int = 512,
    max_history: int = 12,
    max_observation_chars: int = 360,
    device: str | torch.device | None = None,
) -> CandidateBatch:
    candidates = tuple(str(action) for action in record.get("candidates", ()))
    prompts = [
        candidate_prompt(
            record,
            action,
            max_history=max_history,
            max_observation_chars=max_observation_chars,
        )
        for action in candidates
    ]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return CandidateBatch(
        candidates=candidates,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )


def yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    yes_ids = tokenizer.encode(" yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" no", add_special_tokens=False)
    if not yes_ids:
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    if not no_ids:
        no_ids = tokenizer.encode("no", add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise ValueError("Could not identify yes/no token ids for tokenizer.")
    return int(yes_ids[0]), int(no_ids[0])


def candidate_scores(
    model: Any,
    batch: CandidateBatch,
    yes_id: int,
    no_id: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= batch.input_ids.shape[0]:
        logits = final_token_logits(
            model,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        return logits[:, yes_id] - logits[:, no_id]

    scores: list[torch.Tensor] = []
    for start in range(0, batch.input_ids.shape[0], chunk_size):
        stop = min(start + chunk_size, batch.input_ids.shape[0])
        input_ids = batch.input_ids[start:stop]
        attention_mask = batch.attention_mask[start:stop]
        logits = final_token_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        scores.append(logits[:, yes_id] - logits[:, no_id])
    return torch.cat(scores, dim=0)


def final_token_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
        )
        logits = outputs.logits
        if logits.ndim == 3 and logits.shape[1] == 1:
            return logits[:, 0, :]
        if logits.ndim == 3:
            return logits[:, -1, :]
    except TypeError:
        pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    last_indices = torch.where(attention_mask.bool(), positions, -1).amax(dim=1)
    if bool((last_indices < 0).any()):
        raise ValueError("Cannot score an entirely padded candidate prompt")
    row_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    return outputs.logits[row_indices, last_indices, :]


def raw_rank_prior(
    num_candidates: int,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    ranks = torch.arange(num_candidates, dtype=torch.float32, device=device)
    logits = -ranks / max(temperature, 1e-6)
    return torch.softmax(logits, dim=0)


def scpo_loss(
    scores: torch.Tensor,
    target_index: int,
    raw_prior: torch.Tensor,
    kl_coef: float = 0.03,
    mode: str = "preference",
) -> tuple[torch.Tensor, dict[str, float]]:
    log_probs = torch.log_softmax(scores, dim=0)
    probs = torch.softmax(scores, dim=0)
    rewards = torch.zeros_like(scores)
    rewards[target_index] = 1.0
    expected_reward = torch.sum(probs * rewards)
    kl = torch.sum(probs * (log_probs - torch.log(raw_prior.clamp_min(1e-8))))
    # The NLL term keeps the contextual bandit update data-efficient on small
    # ReCAP datasets; the reward term makes the objective an explicit
    # support-constrained policy improvement objective over logged candidates.
    nll = -log_probs[target_index]
    loss = nll - expected_reward + kl_coef * kl
    metrics = {
        f"{mode}_nll": float(nll.detach().cpu()),
        f"{mode}_reward": float(expected_reward.detach().cpu()),
        f"{mode}_kl": float(kl.detach().cpu()),
    }
    return loss, metrics


def candidate_reward_tensor(
    record: Mapping[str, Any],
    candidates: Sequence[str],
    device: str | torch.device | None = None,
) -> torch.Tensor | None:
    rewards = record.get("candidate_rewards")
    if not isinstance(rewards, Mapping):
        return None
    values = []
    found = False
    for action in candidates:
        value = rewards.get(str(action))
        if value is None:
            values.append(0.0)
        else:
            values.append(float(value))
            found = True
    if not found:
        return None
    return torch.tensor(values, dtype=torch.float32, device=device)


def reward_guided_scpo_loss(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    raw_prior: torch.Tensor,
    kl_coef: float = 0.03,
    reward_temperature: float = 1.0,
    expected_reward_coef: float = 0.25,
    mode: str = "preference",
) -> tuple[torch.Tensor, dict[str, float]]:
    log_probs = torch.log_softmax(scores, dim=0)
    probs = torch.softmax(scores, dim=0)
    centered = rewards - rewards.mean()
    scale = centered.std(unbiased=False).clamp_min(1e-6)
    normalized_rewards = centered / scale
    target = torch.softmax(normalized_rewards / max(reward_temperature, 1e-6), dim=0)
    distill_ce = -torch.sum(target.detach() * log_probs)
    expected_reward = torch.sum(probs * normalized_rewards)
    kl = torch.sum(probs * (log_probs - torch.log(raw_prior.clamp_min(1e-8))))
    loss = distill_ce - expected_reward_coef * expected_reward + kl_coef * kl
    metrics = {
        f"{mode}_reward_ce": float(distill_ce.detach().cpu()),
        f"{mode}_reward": float(expected_reward.detach().cpu()),
        f"{mode}_kl": float(kl.detach().cpu()),
        f"{mode}_reward_span": float((rewards.max() - rewards.min()).detach().cpu()),
    }
    return loss, metrics


def kl_regularized_reward_loss(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    raw_prior: torch.Tensor,
    kl_coef: float = 0.03,
    reward_temperature: float = 1.0,
    expected_reward_coef: float = 0.25,
    pairwise_advantage_coef: float = 0.0,
    advantage_margin: float = 0.0,
    entropy_coef: float = 0.0,
    mode: str = "preference",
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL-regularized support-constrained policy improvement.

    The target is the closed-form KL-regularized improvement distribution

        pi*(a|x,C) proportional to pi_raw(a|x,C) exp(A(a) / tau),

    where A is a normalized replay-derived candidate reward. This is more
    conservative than reward-only distillation and is better suited to small
    online-control datasets: the learner only departs from the raw candidate
    prior when replay reward provides evidence.
    """

    log_probs = torch.log_softmax(scores, dim=0)
    probs = torch.softmax(scores, dim=0)
    centered = rewards - rewards.mean()
    scale = centered.std(unbiased=False).clamp_min(1e-6)
    advantages = centered / scale
    prior_log = torch.log(raw_prior.clamp_min(1e-8))
    target_logits = prior_log + advantages / max(reward_temperature, 1e-6)
    target = torch.softmax(target_logits, dim=0)
    policy_ce = -torch.sum(target.detach() * log_probs)
    expected_reward = torch.sum(probs * advantages)
    kl = torch.sum(probs * (log_probs - prior_log))
    entropy = -torch.sum(probs * log_probs)
    pairwise = advantage_pairwise_loss(
        scores,
        advantages,
        margin=advantage_margin,
    )
    loss = (
        policy_ce
        - expected_reward_coef * expected_reward
        + kl_coef * kl
        + pairwise_advantage_coef * pairwise
        - entropy_coef * entropy
    )
    metrics = {
        f"{mode}_target_ce": float(policy_ce.detach().cpu()),
        f"{mode}_reward": float(expected_reward.detach().cpu()),
        f"{mode}_kl": float(kl.detach().cpu()),
        f"{mode}_entropy": float(entropy.detach().cpu()),
        f"{mode}_adv_pairwise": float(pairwise.detach().cpu()),
        f"{mode}_reward_span": float((rewards.max() - rewards.min()).detach().cpu()),
    }
    return loss, metrics


def advantage_pairwise_loss(
    scores: torch.Tensor,
    advantages: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    diffs = advantages[:, None] - advantages[None, :]
    mask = diffs > margin
    if not bool(mask.any()):
        return scores.new_tensor(0.0)
    score_diffs = scores[:, None] - scores[None, :]
    weights = diffs.clamp_min(0.0).detach()
    losses = torch.nn.functional.softplus(-score_diffs)
    weighted = losses * weights * mask
    return weighted.sum() / (weights * mask).sum().clamp_min(1e-6)


def pairwise_reward_loss(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    raw_prior: torch.Tensor,
    kl_coef: float = 0.0,
    mode: str = "preference",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direct pairwise control: compare unequal rewards, without label boosting.

    Each ordered reward comparison has equal weight. Ties imply no comparison;
    an all-tied list returns a differentiable zero (plus optional prior KL).
    This tests the same reward information with a simpler objective.
    """
    mask = rewards[:, None] > rewards[None, :]
    differences = scores[:, None] - scores[None, :]
    pairwise = (torch.nn.functional.softplus(-differences) * mask).sum() / mask.sum().clamp_min(1)
    log_probs = torch.log_softmax(scores, dim=0)
    kl = (log_probs.exp() * (log_probs - raw_prior.clamp_min(1e-8).log())).sum()
    loss = pairwise + kl_coef * kl
    return loss, {
        f"{mode}_pairwise": float(pairwise.detach()),
        f"{mode}_kl": float(kl.detach()),
        f"{mode}_comparison_count": int(mask.sum()),
    }


def target_index(record: Mapping[str, Any]) -> int | None:
    candidates = tuple(str(action) for action in record.get("candidates", ()))
    preferred = str(record.get("preferred_action", ""))
    if preferred not in candidates:
        return None
    return candidates.index(preferred)


def make_prediction(
    record: Mapping[str, Any],
    ranked_actions: Sequence[str],
    scores: Mapping[str, float],
    abstain: bool = False,
    model_type: str = "recap_lm_scpo",
) -> dict[str, Any]:
    candidates = tuple(str(action) for action in record.get("candidates", ()))
    raw_top = candidates[0] if candidates else None
    learned_top = ranked_actions[0] if ranked_actions else None
    margin_over_raw = (
        float(scores.get(learned_top, 0.0) - scores.get(raw_top, 0.0))
        if learned_top is not None and raw_top is not None
        else 0.0
    )
    return {
        "task_id": str(record["task_id"]),
        "seed": int(record.get("seed", 0)),
        "step_index": int(record.get("step_index", len(record.get("history", ())))),
        "ranked_actions": list(ranked_actions),
        "scores": dict(scores),
        "abstain": bool(abstain),
        "margin_over_raw_top1": margin_over_raw,
        "model_type": model_type,
    }
