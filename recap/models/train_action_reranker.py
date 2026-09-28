from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from recap.models.reranker_dataset import (
    FEATURE_NAMES,
    FEATURE_SETS,
    active_feature_names,
    certificate_weight,
    featurize_candidate,
    pairwise_deltas,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight ReCAP feature reranker.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, default=None)
    parser.add_argument("--keep", type=Path, default=None)
    parser.add_argument("--keep-lambda", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--loss", choices=["pairwise", "listwise"], default="pairwise")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--no-certificate-weight", action="store_true")
    parser.add_argument(
        "--negative-scope",
        choices=["all-candidates", "rejected"],
        default="all-candidates",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_records = tuple(read_jsonl(args.train))
    valid_records = tuple(read_jsonl(args.valid)) if args.valid is not None else ()
    keep_records = tuple(read_jsonl(args.keep)) if args.keep is not None else ()
    model = train_feature_reranker(
        train_records=train_records,
        valid_records=valid_records,
        keep_records=keep_records,
        keep_lambda=args.keep_lambda,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        loss=args.loss,
        feature_set=args.feature_set,
        use_certificate_weight=not args.no_certificate_weight,
        negative_scope=args.negative_scope,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(model["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def train_feature_reranker(
    train_records: tuple[Mapping[str, Any], ...],
    valid_records: tuple[Mapping[str, Any], ...] = (),
    keep_records: tuple[Mapping[str, Any], ...] = (),
    keep_lambda: float = 0.0,
    epochs: int = 80,
    learning_rate: float = 0.1,
    l2: float = 0.001,
    loss: str = "pairwise",
    feature_set: str = "all",
    use_certificate_weight: bool = True,
    negative_scope: str = "all-candidates",
) -> dict[str, Any]:
    weights = {name: 0.0 for name in FEATURE_NAMES}
    active_features = active_feature_names(feature_set)
    if not train_records:
        return model_payload(
            weights=weights,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            loss=loss,
            feature_set=feature_set,
            use_certificate_weight=use_certificate_weight,
            keep_lambda=keep_lambda,
            negative_scope=negative_scope,
            train_records=train_records,
            valid_records=valid_records,
            keep_records=keep_records,
            loss_history=[],
        )

    loss_history: list[float] = []
    for _epoch in range(epochs):
        if loss == "pairwise":
            total_loss, item_count = pairwise_epoch(
                train_records=train_records,
                weights=weights,
                active_features=active_features,
                learning_rate=learning_rate,
                l2=l2,
                negative_scope=negative_scope,
                use_certificate_weight=use_certificate_weight,
                keep_records=keep_records,
                keep_lambda=keep_lambda,
            )
        elif loss == "listwise":
            total_loss, item_count = listwise_epoch(
                train_records=train_records,
                weights=weights,
                active_features=active_features,
                learning_rate=learning_rate,
                l2=l2,
                use_certificate_weight=use_certificate_weight,
                keep_records=keep_records,
                keep_lambda=keep_lambda,
            )
        else:
            raise ValueError(f"unknown loss: {loss}")
        loss_history.append(total_loss / item_count if item_count else 0.0)

    return model_payload(
        weights=weights,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        loss=loss,
        feature_set=feature_set,
        use_certificate_weight=use_certificate_weight,
        keep_lambda=keep_lambda,
        negative_scope=negative_scope,
        train_records=train_records,
        valid_records=valid_records,
        keep_records=keep_records,
        loss_history=loss_history,
    )


def model_payload(
    weights: Mapping[str, float],
    epochs: int,
    learning_rate: float,
    l2: float,
    loss: str,
    feature_set: str,
    use_certificate_weight: bool,
    keep_lambda: float,
    negative_scope: str,
    train_records: tuple[Mapping[str, Any], ...],
    valid_records: tuple[Mapping[str, Any], ...],
    keep_records: tuple[Mapping[str, Any], ...],
    loss_history: list[float],
) -> dict[str, Any]:
    return {
        "model_type": "recap_feature_pairwise_logistic",
        "feature_names": list(FEATURE_NAMES),
        "weights": {name: float(weights.get(name, 0.0)) for name in FEATURE_NAMES},
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "loss": loss,
            "feature_set": feature_set,
            "use_certificate_weight": use_certificate_weight,
            "keep_lambda": keep_lambda,
            "negative_scope": negative_scope,
        },
        "summary": {
            "train_preferences": len(train_records),
            "valid_preferences": len(valid_records),
            "keep_preferences": len(keep_records),
            "train_pairwise_accuracy": pairwise_accuracy(
                train_records,
                weights,
                negative_scope=negative_scope,
            ),
            "valid_pairwise_accuracy": (
                pairwise_accuracy(valid_records, weights, negative_scope=negative_scope)
                if valid_records
                else None
            ),
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
        },
    }


def pairwise_epoch(
    train_records: tuple[Mapping[str, Any], ...],
    weights: dict[str, float],
    active_features: tuple[str, ...],
    learning_rate: float,
    l2: float,
    negative_scope: str,
    use_certificate_weight: bool,
    keep_records: tuple[Mapping[str, Any], ...] = (),
    keep_lambda: float = 0.0,
) -> tuple[float, int]:
    total_loss = 0.0
    pair_count = 0
    for preference, multiplier, weighted in training_records(
        train_records,
        keep_records,
        keep_lambda,
        use_certificate_weight,
    ):
        for delta in pairwise_deltas(preference, negative_scope=negative_scope):
            pair_count += 1
            margin = dot(weights, delta)
            weight = multiplier * example_weight(preference, weighted)
            probability = sigmoid(margin)
            total_loss += weight * softplus(-margin)
            scale = weight * (1.0 - probability)
            for name in active_features:
                weights[name] += learning_rate * (
                    scale * delta.get(name, 0.0) - l2 * weights[name]
                )
    return total_loss, pair_count


def listwise_epoch(
    train_records: tuple[Mapping[str, Any], ...],
    weights: dict[str, float],
    active_features: tuple[str, ...],
    learning_rate: float,
    l2: float,
    use_certificate_weight: bool,
    keep_records: tuple[Mapping[str, Any], ...] = (),
    keep_lambda: float = 0.0,
) -> tuple[float, int]:
    total_loss = 0.0
    example_count = 0
    for preference, multiplier, weighted in training_records(
        train_records,
        keep_records,
        keep_lambda,
        use_certificate_weight,
    ):
        candidates = tuple(str(action) for action in preference.get("candidates", ()))
        preferred = str(preference.get("preferred_action", ""))
        if preferred not in candidates:
            continue
        feature_rows = {
            action: featurize_candidate(preference, action)
            for action in candidates
        }
        scores = {action: dot(weights, features) for action, features in feature_rows.items()}
        probabilities = softmax(scores)
        preferred_probability = max(probabilities.get(preferred, 0.0), 1e-12)
        weight = multiplier * example_weight(preference, weighted)
        total_loss += -weight * math.log(preferred_probability)
        example_count += 1
        expected = {
            name: sum(
                probabilities[action] * feature_rows[action].get(name, 0.0)
                for action in candidates
            )
            for name in active_features
        }
        preferred_features = feature_rows[preferred]
        for name in active_features:
            gradient = weight * (preferred_features.get(name, 0.0) - expected[name])
            weights[name] += learning_rate * (gradient - l2 * weights[name])
    return total_loss, example_count


def training_records(
    train_records: tuple[Mapping[str, Any], ...],
    keep_records: tuple[Mapping[str, Any], ...],
    keep_lambda: float,
    use_certificate_weight: bool,
) -> tuple[tuple[Mapping[str, Any], float, bool], ...]:
    rows: list[tuple[Mapping[str, Any], float, bool]] = [
        (record, 1.0, use_certificate_weight)
        for record in train_records
    ]
    if keep_lambda > 0:
        rows.extend((record, keep_lambda, False) for record in keep_records)
    return tuple(rows)


def example_weight(preference: Mapping[str, Any], use_certificate_weight: bool) -> float:
    return certificate_weight(preference) if use_certificate_weight else 1.0


def pairwise_accuracy(
    records: tuple[Mapping[str, Any], ...],
    weights: Mapping[str, float],
    negative_scope: str = "all-candidates",
) -> float:
    if not records:
        return 0.0
    correct = 0
    total = 0
    for preference in records:
        for delta in pairwise_deltas(preference, negative_scope=negative_scope):
            total += 1
            if dot(weights, delta) > 0:
                correct += 1
    return correct / total if total else 0.0


def dot(weights: Mapping[str, float], features: Mapping[str, float]) -> float:
    return sum(float(weights.get(name, 0.0)) * value for name, value in features.items())


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softplus(value: float) -> float:
    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))


def softmax(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {action: math.exp(score - max_score) for action, score in scores.items()}
    total = sum(exp_scores.values())
    return {action: value / total for action, value in exp_scores.items()}


if __name__ == "__main__":
    main()
