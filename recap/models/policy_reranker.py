from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from recap.models.reranker_dataset import FEATURE_NAMES, active_feature_names, featurize_candidate


class CandidatePolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current_dim = input_dim
        for _index in range(max(num_layers, 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def candidate_matrix(
    preference: Mapping[str, Any],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[list[str], np.ndarray]:
    candidates = [str(action) for action in preference.get("candidates", ())]
    rows = []
    for action in candidates:
        features = featurize_candidate(preference, action)
        rows.append([float(features.get(name, 0.0)) for name in feature_names])
    if not rows:
        return candidates, np.zeros((0, len(feature_names)), dtype=np.float32)
    return candidates, np.asarray(rows, dtype=np.float32)


def train_policy_reranker(
    preferences: Sequence[Mapping[str, Any]],
    validation_preferences: Sequence[Mapping[str, Any]] | None = None,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    entropy_coef: float = 0.01,
    weight_decay: float = 1e-4,
    feature_set: str = "all",
    soft_target_weight: float = 0.0,
    reward_temperature: float = 0.35,
    seed: int = 0,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    feature_names = tuple(active_feature_names(feature_set))
    model = CandidatePolicy(
        len(feature_names),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    records = [
        record
        for record in preferences
        if str(record.get("preferred_action")) in tuple(str(c) for c in record.get("candidates", ()))
    ]
    train_batch = build_training_batch(
        records,
        feature_names=feature_names,
        reward_temperature=reward_temperature,
    )
    validation_records = [
        record
        for record in (validation_preferences or ())
        if str(record.get("preferred_action")) in tuple(str(c) for c in record.get("candidates", ()))
    ]
    validation_batch = build_training_batch(
        validation_records,
        feature_names=feature_names,
        reward_temperature=reward_temperature,
    ) if validation_records else None
    loss_history: list[float] = []
    validation_history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    for _epoch in range(epochs):
        model.train()
        loss = policy_loss(
            model,
            train_batch,
            entropy_coef=entropy_coef,
            soft_target_weight=soft_target_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach())
        loss_history.append(train_loss)
        if validation_batch is not None:
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    policy_loss(
                        model,
                        validation_batch,
                        entropy_coef=0.0,
                        soft_target_weight=soft_target_weight,
                    ).detach()
                )
            validation_history.append(validation_loss)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model_type": "recap_support_constrained_policy",
        "state_dict": model.state_dict(),
        "feature_names": feature_names,
        "hyperparameters": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "entropy_coef": entropy_coef,
            "weight_decay": weight_decay,
            "feature_set": feature_set,
            "soft_target_weight": soft_target_weight,
            "reward_temperature": reward_temperature,
            "seed": seed,
            "objective": "support_constrained_recap_policy",
        },
        "summary": {
            "train_preferences": len(records),
            "validation_preferences": len(validation_records),
            "loss_initial": loss_history[0] if loss_history else None,
            "loss_final": loss_history[-1] if loss_history else None,
            "validation_loss_final": validation_history[-1] if validation_history else None,
            "validation_loss_best": best_validation if best_state is not None else None,
        },
    }


def build_training_batch(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    reward_temperature: float,
) -> dict[str, torch.Tensor]:
    if not records:
        return {
            "features": torch.zeros((0, 0, len(feature_names)), dtype=torch.float32),
            "mask": torch.zeros((0, 0), dtype=torch.bool),
            "targets": torch.zeros((0,), dtype=torch.long),
            "soft_targets": torch.zeros((0, 0), dtype=torch.float32),
        }
    max_candidates = max(len(record.get("candidates", ())) for record in records)
    features = torch.zeros((len(records), max_candidates, len(feature_names)), dtype=torch.float32)
    mask = torch.zeros((len(records), max_candidates), dtype=torch.bool)
    targets = torch.zeros((len(records),), dtype=torch.long)
    soft_targets = torch.zeros((len(records), max_candidates), dtype=torch.float32)
    for row_index, record in enumerate(records):
        candidates, matrix = candidate_matrix(record, feature_names=feature_names)
        count = len(candidates)
        features[row_index, :count] = torch.from_numpy(matrix)
        mask[row_index, :count] = True
        preferred = str(record["preferred_action"])
        targets[row_index] = candidates.index(preferred)
        rewards = candidate_rewards(record, candidates)
        if rewards is None:
            soft_targets[row_index, targets[row_index]] = 1.0
        else:
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            reward_tensor[targets[row_index]] = max(float(reward_tensor[targets[row_index]]), 1.0)
            logits = reward_tensor / max(reward_temperature, 1e-4)
            soft_targets[row_index, :count] = torch.softmax(logits, dim=0)
    return {
        "features": features,
        "mask": mask,
        "targets": targets,
        "soft_targets": soft_targets,
    }


def candidate_rewards(record: Mapping[str, Any], candidates: Sequence[str]) -> list[float] | None:
    raw_rewards = record.get("candidate_replay_values", record.get("candidate_rewards"))
    if not isinstance(raw_rewards, Mapping):
        return None
    rewards = [float(raw_rewards.get(action, raw_rewards.get(str(action), 0.0))) for action in candidates]
    return rewards if any(value != 0.0 for value in rewards) else None


def policy_loss(
    model: CandidatePolicy,
    batch: Mapping[str, torch.Tensor],
    entropy_coef: float,
    soft_target_weight: float,
) -> torch.Tensor:
    features = batch["features"]
    if features.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    mask = batch["mask"]
    logits = model(features).masked_fill(~mask, -1e9)
    targets = batch["targets"]
    hard_loss = nn.functional.cross_entropy(logits, targets)
    soft_weight = max(0.0, min(float(soft_target_weight), 1.0))
    if soft_weight:
        log_probs = torch.log_softmax(logits, dim=-1)
        soft_targets = batch["soft_targets"]
        soft_loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        loss = (1.0 - soft_weight) * hard_loss + soft_weight * soft_loss
    else:
        loss = hard_loss
    if entropy_coef:
        probs = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
        loss = loss - entropy_coef * entropy
    return loss


def load_policy_model(payload: Mapping[str, Any]) -> CandidatePolicy:
    hidden_dim = int(payload.get("hyperparameters", {}).get("hidden_dim", 64))
    num_layers = int(payload.get("hyperparameters", {}).get("num_layers", 1))
    dropout = float(payload.get("hyperparameters", {}).get("dropout", 0.0))
    feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
    model = CandidatePolicy(
        len(feature_names),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def rank_candidates(
    preference: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[tuple[str, float], ...]:
    model = load_policy_model(payload)
    return rank_candidates_with_model(preference, model, payload)


def rank_candidates_with_model(
    preference: Mapping[str, Any],
    model: CandidatePolicy,
    payload: Mapping[str, Any],
) -> tuple[tuple[str, float], ...]:
    feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
    candidates, matrix = candidate_matrix(preference, feature_names=feature_names)
    if not candidates:
        return ()
    with torch.no_grad():
        logits = model(torch.from_numpy(matrix))
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    indexed = list(enumerate(zip(candidates, probs.tolist())))
    indexed.sort(key=lambda item: (-item[1][1], item[0]))
    return tuple((action, float(score)) for _index, (action, score) in indexed)
