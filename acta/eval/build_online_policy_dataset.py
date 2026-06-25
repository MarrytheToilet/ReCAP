from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from acta.agents.llm_agent import rank_admissible_actions
from acta.agents.lm_policy_agent import (
    is_inverse_navigation,
    is_manipulation,
    is_semantic_undo,
    objective_action_overlap,
)
from acta.envs.textworld_adapter import TextWorldAdapter
from acta.models.reranker_dataset import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build online-style candidate-policy records from replayed TextWorld states."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--candidate-pool-limit", type=int, default=40)
    parser.add_argument("--max-policy-depth", type=int, default=5)
    parser.add_argument("--force-include-gold", action="store_true")
    parser.add_argument("--gold-reward", type=float, default=2.0)
    parser.add_argument("--suffix-reward", type=float, default=0.8)
    parser.add_argument("--recent-repeat-penalty", type=float, default=1.0)
    parser.add_argument("--inverse-penalty", type=float, default=1.0)
    parser.add_argument("--static-penalty", type=float, default=0.25)
    parser.add_argument("--semantic-undo-penalty", type=float, default=0.0)
    parser.add_argument("--objective-overlap-bonus", type=float, default=0.0)
    parser.add_argument("--navigation-bonus", type=float, default=0.0)
    parser.add_argument("--nonobjective-manipulation-penalty", type=float, default=0.0)
    args = parser.parse_args()

    trajectories = read_jsonl(args.trajectories)
    if args.max_trajectories is not None:
        trajectories = trajectories[: args.max_trajectories]
    records, summary = build_online_policy_records(
        trajectories=trajectories,
        candidate_pool_limit=args.candidate_pool_limit,
        max_policy_depth=args.max_policy_depth,
        force_include_gold=args.force_include_gold,
        gold_reward=args.gold_reward,
        suffix_reward=args.suffix_reward,
        recent_repeat_penalty=args.recent_repeat_penalty,
        inverse_penalty=args.inverse_penalty,
        static_penalty=args.static_penalty,
        semantic_undo_penalty=args.semantic_undo_penalty,
        objective_overlap_bonus=args.objective_overlap_bonus,
        navigation_bonus=args.navigation_bonus,
        nonobjective_manipulation_penalty=args.nonobjective_manipulation_penalty,
        max_records=args.max_records,
    )
    write_jsonl(args.out, records)
    summary["out"] = str(args.out)
    summary_path = args.summary_out or args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


def build_online_policy_records(
    trajectories: list[Mapping[str, Any]],
    candidate_pool_limit: int,
    max_policy_depth: int,
    force_include_gold: bool,
    gold_reward: float,
    suffix_reward: float,
    recent_repeat_penalty: float,
    inverse_penalty: float,
    static_penalty: float,
    semantic_undo_penalty: float,
    objective_overlap_bonus: float,
    navigation_bonus: float,
    nonobjective_manipulation_penalty: float,
    max_records: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter = TextWorldAdapter()
    records: list[dict[str, Any]] = []
    skipped = {
        "done_prefix": 0,
        "invalid_replay": 0,
        "no_policy": 0,
        "gold_absent": 0,
        "single_candidate": 0,
    }
    try:
        for trajectory in trajectories:
            task_id = str(trajectory["task_id"])
            seed = int(trajectory.get("seed", trajectory.get("rollout_seed", 0)))
            initial = adapter.reset(task_id=task_id, seed=seed).observation
            history: list[str] = []
            for step in trajectory.get("steps", ()):
                replay = adapter.replay(task_id=task_id, prefix_actions=tuple(history), seed=seed)
                if replay.done:
                    skipped["done_prefix"] += 1
                    break
                if not replay.valid:
                    skipped["invalid_replay"] += 1
                    history.append(str(step.get("action", "")))
                    continue
                policy = tuple(str(action) for action in adapter.policy_commands(replay.state))
                if not policy:
                    skipped["no_policy"] += 1
                    history.append(str(step.get("action", "")))
                    continue
                gold = policy[0]
                admissible = tuple(str(action) for action in adapter.admissible_actions(replay.state))
                candidates = list(rank_admissible_actions(admissible, tuple(history))[:candidate_pool_limit])
                if gold not in candidates:
                    if not force_include_gold:
                        skipped["gold_absent"] += 1
                        history.append(str(step.get("action", "")))
                        continue
                    if len(candidates) >= candidate_pool_limit and candidates:
                        candidates[-1] = gold
                    else:
                        candidates.append(gold)
                deduped = []
                for action in candidates:
                    if action not in deduped:
                        deduped.append(action)
                candidates = deduped
                if len(candidates) < 2:
                    skipped["single_candidate"] += 1
                    history.append(str(step.get("action", "")))
                    continue
                rewards = shaped_rewards(
                    candidates=tuple(candidates),
                    policy=policy[:max_policy_depth],
                    history=tuple(history),
                    gold_reward=gold_reward,
                    suffix_reward=suffix_reward,
                    recent_repeat_penalty=recent_repeat_penalty,
                    inverse_penalty=inverse_penalty,
                    static_penalty=static_penalty,
                    semantic_undo_penalty=semantic_undo_penalty,
                    objective_overlap_bonus=objective_overlap_bonus,
                    navigation_bonus=navigation_bonus,
                    nonobjective_manipulation_penalty=nonobjective_manipulation_penalty,
                    objective=initial,
                )
                rejected = first_non_gold(tuple(candidates), gold)
                if rejected is None:
                    skipped["single_candidate"] += 1
                    history.append(str(step.get("action", "")))
                    continue
                records.append(
                    {
                        "task_id": task_id,
                        "trajectory_id": str(trajectory.get("trajectory_id", "")),
                        "env": str(trajectory.get("env", "textworld")),
                        "difficulty": str(trajectory.get("difficulty", "")),
                        "game_id": str(trajectory.get("game_id", "")),
                        "game_seed": trajectory.get("game_seed"),
                        "seed": seed,
                        "rollout_seed": int(trajectory.get("rollout_seed", seed)),
                        "step_index": int(step.get("step_index", len(history))),
                        "history": tuple(history),
                        "observation": replay.observation,
                        "initial_observation": initial,
                        "candidates": tuple(candidates),
                        "candidate_count": len(candidates),
                        "preferred_action": gold,
                        "rejected_action": rejected,
                        "preferred_rank_before": candidates.index(gold) + 1,
                        "rejected_rank_before": candidates.index(rejected) + 1,
                        "candidate_rewards": rewards,
                        "policy_suffix": policy[:max_policy_depth],
                        "source": "online_policy_suffix_reward",
                        "certificate_level": "K2_policy_suffix_online_pool",
                    }
                )
                history.append(str(step.get("action", "")))
                if max_records is not None and len(records) >= max_records:
                    return records, summary(records, skipped, trajectories)
    finally:
        adapter.close()
    return records, summary(records, skipped, trajectories)


def shaped_rewards(
    candidates: tuple[str, ...],
    policy: tuple[str, ...],
    history: tuple[str, ...],
    gold_reward: float,
    suffix_reward: float,
    recent_repeat_penalty: float,
    inverse_penalty: float,
    static_penalty: float,
    semantic_undo_penalty: float,
    objective_overlap_bonus: float,
    navigation_bonus: float,
    nonobjective_manipulation_penalty: float,
    objective: str = "",
) -> dict[str, float]:
    rewards: dict[str, float] = {}
    suffix = {action: suffix_reward / (index + 1) for index, action in enumerate(policy)}
    for action in candidates:
        reward = suffix.get(action, 0.0)
        if action == policy[0]:
            reward += gold_reward
        if action in history[-4:]:
            reward -= recent_repeat_penalty
        if history and is_inverse_navigation(history[-1], action):
            reward -= inverse_penalty
        if any(is_semantic_undo(previous, action) for previous in history[-3:]):
            reward -= semantic_undo_penalty
        if action.split()[:1] in (["look"], ["inventory"], ["examine"]):
            reward -= static_penalty
        overlap = objective_action_overlap(action, objective)
        reward += objective_overlap_bonus * overlap
        if action.startswith("go "):
            reward += navigation_bonus
        if is_manipulation(action) and overlap <= 0.0:
            reward -= nonobjective_manipulation_penalty
        rewards[action] = float(reward)
    return rewards


def first_non_gold(candidates: tuple[str, ...], gold: str) -> str | None:
    for action in candidates:
        if action != gold:
            return action
    return None


def summary(
    records: list[Mapping[str, Any]],
    skipped: Mapping[str, int],
    trajectories: list[Mapping[str, Any]],
) -> dict[str, Any]:
    ranks = [int(record["preferred_rank_before"]) for record in records]
    return {
        "trajectories": len(trajectories),
        "records": len(records),
        "avg_gold_rank": sum(ranks) / len(ranks) if ranks else 0.0,
        "gold_top1_rate": sum(1 for rank in ranks if rank == 1) / len(ranks) if ranks else 0.0,
        "skipped": dict(skipped),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
