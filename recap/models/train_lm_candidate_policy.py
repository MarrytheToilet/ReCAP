from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from recap.models.lm_candidate_policy import (
    candidate_reward_tensor,
    candidate_scores,
    encode_candidate_batch,
    kl_regularized_reward_loss,
    raw_rank_prior,
    pairwise_reward_loss,
    reward_guided_scpo_loss,
    scpo_loss,
    target_index,
)
from recap.models.reranker_dataset import read_jsonl


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = list(read_jsonl(path))
    if limit is not None:
        records = records[:limit]
    return [dict(record) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a LoRA LM policy with ReCAP support-constrained preference optimization."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--retention", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="Optional existing LoRA adapter to warm-start policy improvement from.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=12)
    parser.add_argument("--max-observation-chars", type=int, default=360)
    parser.add_argument("--candidate-chunk-size", type=int, default=0)
    parser.add_argument(
        "--max-train-candidates",
        type=int,
        default=0,
        help="Subsample candidates per training record while always keeping the target/reward-best action.",
    )
    parser.add_argument("--kl-coef", type=float, default=0.03)
    parser.add_argument("--keep-coef", type=float, default=0.25)
    parser.add_argument("--rank-prior-temperature", type=float, default=1.0)
    parser.add_argument("--use-candidate-rewards", action="store_true")
    parser.add_argument(
        "--reward-loss",
        choices=["softmax", "kl-regularized", "pairwise"],
        default="softmax",
    )
    parser.add_argument("--reward-temperature", type=float, default=1.0)
    parser.add_argument("--expected-reward-coef", type=float, default=0.25)
    parser.add_argument("--pairwise-advantage-coef", type=float, default=0.0)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--retention-ratio", type=float, default=0.5)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--retention-limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear")
    args = parser.parse_args()

    if args.use_candidate_rewards and args.reward_loss != "kl-regularized":
        if args.pairwise_advantage_coef or args.entropy_coef:
            import warnings
            warnings.warn("pairwise-advantage-coef and entropy-coef only affect --reward-loss kl-regularized; they are inactive in this run")

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": args.device}

    model = AutoModelForCausalLM.from_pretrained(str(args.base_model), **model_kwargs)
    if not args.load_in_4bit:
        model.to(args.device)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    target_modules: str | list[str]
    if args.target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]

    if args.init_adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            str(args.init_adapter),
            is_trainable=True,
            local_files_only=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.train()

    train_records = load_records(args.train, limit=args.train_limit)
    train_records = [record for record in train_records if target_index(record) is not None]
    retention_records: list[dict[str, Any]] = []
    if args.retention is not None and args.retention.exists() and args.keep_coef > 0:
        retention_records = load_records(args.retention, limit=args.retention_limit)
        retention_records = [record for record in retention_records if target_index(record) is not None]
    random.shuffle(train_records)
    random.shuffle(retention_records)

    yes_id, no_id = yes_no_token_ids(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    steps = 0
    optimizer_steps = 0
    pending_accum = 0
    loss_total = 0.0
    stats: dict[str, list[float]] = {}
    total_steps = max(len(train_records), 1) * max(args.epochs, 1)

    for epoch in range(args.epochs):
        random.shuffle(train_records)
        if retention_records:
            random.shuffle(retention_records)
        for index, record in enumerate(train_records):
            loss, metric = record_loss(
                model=model,
                tokenizer=tokenizer,
                record=record,
                yes_id=yes_id,
                no_id=no_id,
                device=args.device,
                max_length=args.max_length,
                max_history=args.max_history,
                max_observation_chars=args.max_observation_chars,
                candidate_chunk_size=args.candidate_chunk_size,
                max_train_candidates=args.max_train_candidates,
                kl_coef=args.kl_coef,
                prior_temperature=args.rank_prior_temperature,
                use_candidate_rewards=args.use_candidate_rewards,
                reward_loss=args.reward_loss,
                reward_temperature=args.reward_temperature,
                expected_reward_coef=args.expected_reward_coef,
                pairwise_advantage_coef=args.pairwise_advantage_coef,
                advantage_margin=args.advantage_margin,
                entropy_coef=args.entropy_coef,
                mode="preference",
            )
            if retention_records and random.random() < args.retention_ratio:
                keep_record = retention_records[index % len(retention_records)]
                keep_loss, keep_metric = record_loss(
                    model=model,
                    tokenizer=tokenizer,
                    record=keep_record,
                    yes_id=yes_id,
                    no_id=no_id,
                    device=args.device,
                    max_length=args.max_length,
                    max_history=args.max_history,
                    max_observation_chars=args.max_observation_chars,
                    candidate_chunk_size=args.candidate_chunk_size,
                    max_train_candidates=args.max_train_candidates,
                    kl_coef=args.kl_coef,
                    prior_temperature=args.rank_prior_temperature,
                    use_candidate_rewards=False,
                    reward_loss=args.reward_loss,
                    reward_temperature=args.reward_temperature,
                    expected_reward_coef=args.expected_reward_coef,
                    pairwise_advantage_coef=args.pairwise_advantage_coef,
                    advantage_margin=args.advantage_margin,
                    entropy_coef=args.entropy_coef,
                    mode="retention",
                )
                loss = loss + args.keep_coef * keep_loss
                metric.update(keep_metric)
            scaled_loss = loss / max(args.batch_accum, 1)
            scaled_loss.backward()
            steps += 1
            pending_accum += 1
            loss_total += float(loss.detach().cpu())
            for key, value in metric.items():
                stats.setdefault(key, []).append(float(value))
            if steps % max(args.batch_accum, 1) == 0:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_accum = 0
            if args.log_every > 0 and steps % args.log_every == 0:
                avg_loss = loss_total / max(steps, 1)
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={steps}/{total_steps} "
                    f"optimizer_steps={optimizer_steps} "
                    f"avg_loss={avg_loss:.4f}",
                    flush=True,
                )
        if pending_accum:
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending_accum = 0

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    summary = {
        "model_type": "recap_lm_scpo_lora",
        "base_model": str(args.base_model),
        "init_adapter": str(args.init_adapter) if args.init_adapter is not None else None,
        "train_preferences": len(train_records),
        "retention_records": len(retention_records),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_accum": args.batch_accum,
        "max_length": args.max_length,
        "max_history": args.max_history,
        "max_observation_chars": args.max_observation_chars,
        "candidate_chunk_size": args.candidate_chunk_size,
        "max_train_candidates": args.max_train_candidates,
        "kl_coef": args.kl_coef,
        "keep_coef": args.keep_coef,
        "retention_ratio": args.retention_ratio,
        "rank_prior_temperature": args.rank_prior_temperature,
        "use_candidate_rewards": args.use_candidate_rewards,
        "reward_loss": args.reward_loss,
        "reward_temperature": args.reward_temperature,
        "expected_reward_coef": args.expected_reward_coef,
        "pairwise_advantage_coef": args.pairwise_advantage_coef,
        "advantage_margin": args.advantage_margin,
        "entropy_coef": args.entropy_coef,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "avg_loss": loss_total / max(steps, 1),
        "metrics": {key: sum(values) / len(values) for key, values in stats.items()},
    }
    summary_path = args.summary_out or (args.out / "recap_scpo_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    from recap.models.lm_candidate_policy import yes_no_token_ids as _ids

    return _ids(tokenizer)


def record_loss(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    yes_id: int,
    no_id: int,
    device: str,
    max_length: int,
    max_history: int,
    max_observation_chars: int,
    candidate_chunk_size: int,
    max_train_candidates: int,
    kl_coef: float,
    prior_temperature: float,
    use_candidate_rewards: bool,
    reward_loss: str,
    reward_temperature: float,
    expected_reward_coef: float,
    pairwise_advantage_coef: float,
    advantage_margin: float,
    entropy_coef: float,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    record = limited_record_candidates(
        record,
        max_candidates=max_train_candidates,
        use_candidate_rewards=use_candidate_rewards,
    )
    batch = encode_candidate_batch(
        tokenizer,
        record,
        max_length=max_length,
        max_history=max_history,
        max_observation_chars=max_observation_chars,
        device=device,
    )
    target = target_index(record)
    if target is None:
        raise ValueError("record has no target candidate")
    scores = candidate_scores(
        model,
        batch,
        yes_id=yes_id,
        no_id=no_id,
        chunk_size=candidate_chunk_size,
    )
    prior = raw_rank_prior(
        len(batch.candidates),
        temperature=prior_temperature,
        device=scores.device,
    )
    if use_candidate_rewards:
        rewards = candidate_reward_tensor(record, batch.candidates, device=scores.device)
        if rewards is not None:
            if reward_loss == "pairwise":
                return pairwise_reward_loss(scores, rewards, prior, kl_coef=kl_coef, mode=mode)
            if reward_loss == "kl-regularized":
                return kl_regularized_reward_loss(
                    scores,
                    rewards,
                    prior,
                    kl_coef=kl_coef,
                    reward_temperature=reward_temperature,
                    expected_reward_coef=expected_reward_coef,
                    pairwise_advantage_coef=pairwise_advantage_coef,
                    advantage_margin=advantage_margin,
                    entropy_coef=entropy_coef,
                    mode=mode,
                )
            return reward_guided_scpo_loss(
                scores,
                rewards,
                prior,
                kl_coef=kl_coef,
                reward_temperature=reward_temperature,
                expected_reward_coef=expected_reward_coef,
                mode=mode,
            )
    return scpo_loss(scores, target, prior, kl_coef=kl_coef, mode=mode)


def limited_record_candidates(
    record: dict[str, Any],
    max_candidates: int,
    use_candidate_rewards: bool,
) -> dict[str, Any]:
    if max_candidates <= 0:
        return record
    candidates = [str(action) for action in record.get("candidates", ())]
    if len(candidates) <= max_candidates:
        return record

    target = str(record.get("preferred_action", ""))
    rewards = record.get("candidate_rewards")
    reward_values = rewards if isinstance(rewards, dict) else {}
    if use_candidate_rewards and reward_values:
        target = max(candidates, key=lambda action: float(reward_values.get(action, 0.0)))
    if target not in candidates:
        return record

    # Keep the most competitive negatives: original high-ranked candidates and
    # high-reward alternatives. This preserves the local decision difficulty
    # while making online-pool policy optimization tractable on one GPU.
    kept: list[str] = [target]
    for action in candidates:
        if action != target and action not in kept:
            kept.append(action)
        if len(kept) >= max_candidates:
            break
    if use_candidate_rewards and reward_values:
        by_reward = sorted(
            (action for action in candidates if action != target),
            key=lambda action: float(reward_values.get(action, 0.0)),
            reverse=True,
        )
        for action in by_reward:
            if action not in kept:
                if len(kept) >= max_candidates:
                    kept[-1] = action
                else:
                    kept.append(action)
            if len(kept) >= max_candidates and all(action in kept for action in by_reward[:2]):
                break

    ordered = [action for action in candidates if action in kept]
    if target not in ordered:
        ordered.append(target)
    limited = dict(record)
    limited["candidates"] = tuple(ordered[:max_candidates])
    if target not in limited["candidates"]:
        limited["candidates"] = tuple(list(limited["candidates"])[:-1] + [target])
    limited["candidate_count"] = len(limited["candidates"])
    limited["preferred_action"] = target
    limited["preferred_rank_before"] = list(limited["candidates"]).index(target) + 1
    if reward_values:
        limited["candidate_rewards"] = {
            action: float(reward_values.get(action, 0.0)) for action in limited["candidates"]
        }
    rejected = next((action for action in limited["candidates"] if action != target), None)
    if rejected is not None:
        limited["rejected_action"] = rejected
        limited["rejected_rank_before"] = list(limited["candidates"]).index(rejected) + 1
    return limited


if __name__ == "__main__":
    main()
