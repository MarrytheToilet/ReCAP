from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Mapping

from recap.envs.textworld_adapter import TextWorldAdapter
from recap.models.reranker_dataset import normalize_action
from recap.probe.trace_edit_probe import replay_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay every logged candidate to measure repair multiplicity and tie-break sensitivity."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--failed-only", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-rows", type=Path, default=None)
    args = parser.parse_args()

    episodes = tuple(read_jsonl(args.trajectories))
    if args.failed_only:
        episodes = tuple(episode for episode in episodes if not bool(episode.get("success", False)))
    rows = evaluate_episodes(episodes, workers=args.workers)
    output = {
        "summary": summarize_rows(rows),
        "rows": rows if args.out_rows is None else str(args.out_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_rows is not None:
        args.out_rows.parent.mkdir(parents=True, exist_ok=True)
        args.out_rows.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote={args.out}")
    if args.out_rows is not None:
        print(f"rows={args.out_rows}")


def evaluate_episodes(
    episodes: tuple[Mapping[str, Any], ...],
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers <= 1:
        adapter = TextWorldAdapter()
        try:
            rows: list[dict[str, Any]] = []
            for episode in episodes:
                rows.extend(evaluate_episode(adapter, episode))
            return rows
        finally:
            adapter.close()

    rows_by_index: list[list[dict[str, Any]] | None] = [None] * len(episodes)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_episode_worker, dict(episode)): index
            for index, episode in enumerate(episodes)
        }
        for future in as_completed(futures):
            rows_by_index[futures[future]] = future.result()
    rows: list[dict[str, Any]] = []
    for item in rows_by_index:
        rows.extend(item or [])
    return rows


def evaluate_episode_worker(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    adapter = TextWorldAdapter()
    try:
        return evaluate_episode(adapter, episode)
    finally:
        adapter.close()


def evaluate_episode(adapter: TextWorldAdapter, episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    task_id = str(episode["task_id"])
    seed = int(episode.get("seed", 0))
    trajectory_id = str(episode.get("trajectory_id") or f"{Path(task_id).stem}:seed{seed}")
    steps = tuple(episode.get("steps", ()))
    executed_actions = tuple(str(step.get("action", "")) for step in steps)
    rows: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        candidates = dedupe_actions(step.get("candidates_before") or step.get("candidates") or ())
        if not candidates:
            rows.append(
                base_row(
                    episode=episode,
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    seed=seed,
                    step_index=step_index,
                    executed_action=executed_actions[step_index],
                    candidates=candidates,
                    successful=(),
                )
            )
            continue
        prefix = executed_actions[:step_index]
        successes: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            result = replay_candidate_with_policy_suffix(
                adapter=adapter,
                task_id=task_id,
                seed=seed,
                prefix=prefix,
                candidate=candidate,
            )
            if result["success"]:
                successes.append(
                    {
                        "action": candidate,
                        "rank": rank,
                        "suffix_len": result["suffix_len"],
                        "total_len": len(prefix) + 1 + result["suffix_len"],
                        "valid": result["valid"],
                    }
                )
        rows.append(
            base_row(
                episode=episode,
                trajectory_id=trajectory_id,
                task_id=task_id,
                seed=seed,
                step_index=step_index,
                executed_action=executed_actions[step_index],
                candidates=candidates,
                successful=tuple(successes),
            )
        )
    return rows


def replay_candidate_with_policy_suffix(
    adapter: TextWorldAdapter,
    task_id: str,
    seed: int,
    prefix: tuple[str, ...],
    candidate: str,
) -> dict[str, Any]:
    prefix_with_candidate = prefix + (candidate,)
    candidate_replay = adapter.replay(task_id=task_id, prefix_actions=prefix_with_candidate, seed=seed)
    suffix = tuple(str(action) for action in adapter.policy_commands(candidate_replay.state))
    full_replay = adapter.replay(
        task_id=task_id,
        prefix_actions=prefix_with_candidate + suffix,
        seed=seed,
    )
    return {
        "success": replay_success(full_replay.state, full_replay.done),
        "valid": bool(candidate_replay.valid and full_replay.valid),
        "suffix_len": len(suffix),
    }


def base_row(
    episode: Mapping[str, Any],
    trajectory_id: str,
    task_id: str,
    seed: int,
    step_index: int,
    executed_action: str,
    candidates: tuple[str, ...],
    successful: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    successful_actions = tuple(str(item["action"]) for item in successful)
    successful_alternatives = tuple(
        item for item in successful if normalize_action(str(item["action"])) != normalize_action(executed_action)
    )
    shortest = min(successful, key=lambda item: (int(item["suffix_len"]), int(item["rank"])), default=None)
    first_rank = min(successful, key=lambda item: int(item["rank"]), default=None)
    return {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "game_id": str(episode.get("game_id") or Path(task_id).stem),
        "game_seed": episode.get("game_seed"),
        "seed": seed,
        "step_index": step_index,
        "executed_action": executed_action,
        "candidate_count": len(candidates),
        "successful_logged_candidates": len(successful),
        "successful_alternatives": len(successful_alternatives),
        "has_successful_logged_candidate": bool(successful),
        "has_successful_alternative": bool(successful_alternatives),
        "has_multiple_successful_logged_candidates": len(successful) > 1,
        "has_multiple_successful_alternatives": len(successful_alternatives) > 1,
        "successful_actions": list(successful_actions),
        "successful_ranks": [int(item["rank"]) for item in successful],
        "successful_suffix_lens": [int(item["suffix_len"]) for item in successful],
        "first_rank_success": first_rank,
        "shortest_suffix_success": shortest,
        "tie_break_disagreement": (
            first_rank is not None
            and shortest is not None
            and normalize_action(str(first_rank["action"])) != normalize_action(str(shortest["action"]))
        ),
    }


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_steps = [row for row in rows if int(row.get("candidate_count", 0)) > 0]
    repairable = [row for row in candidate_steps if row.get("has_successful_logged_candidate")]
    alternative = [row for row in candidate_steps if row.get("has_successful_alternative")]
    multiple_logged = [
        row for row in candidate_steps if row.get("has_multiple_successful_logged_candidates")
    ]
    multiple_alternative = [
        row for row in candidate_steps if row.get("has_multiple_successful_alternatives")
    ]
    disagreements = [row for row in multiple_logged if row.get("tie_break_disagreement")]
    min_suffix_ties = [
        row for row in repairable if min_suffix_tie_count(row, alternatives_only=False) > 1
    ]
    min_suffix_alt_ties = [
        row for row in repairable if min_suffix_tie_count(row, alternatives_only=True) > 1
    ]
    return {
        "trajectories": len({str(row.get("trajectory_id", "")) for row in rows}),
        "candidate_steps": len(candidate_steps),
        "steps_with_successful_logged_candidate": len(repairable),
        "steps_with_successful_alternative": len(alternative),
        "steps_with_multiple_successful_logged_candidates": len(multiple_logged),
        "steps_with_multiple_successful_alternatives": len(multiple_alternative),
        "tie_break_disagreement_steps": len(disagreements),
        "steps_with_multiple_min_suffix_logged_candidates": len(min_suffix_ties),
        "steps_with_multiple_min_suffix_alternatives": len(min_suffix_alt_ties),
        "successful_logged_candidate_step_rate": safe_div(len(repairable), len(candidate_steps)),
        "successful_alternative_step_rate": safe_div(len(alternative), len(candidate_steps)),
        "multiple_successful_logged_rate_over_candidate_steps": safe_div(
            len(multiple_logged),
            len(candidate_steps),
        ),
        "multiple_successful_logged_rate_over_repairable_steps": safe_div(
            len(multiple_logged),
            len(repairable),
        ),
        "multiple_successful_alternative_rate_over_repairable_steps": safe_div(
            len(multiple_alternative),
            len(repairable),
        ),
        "tie_break_disagreement_rate_over_multi_success_steps": safe_div(
            len(disagreements),
            len(multiple_logged),
        ),
        "multiple_min_suffix_logged_rate_over_repairable_steps": safe_div(
            len(min_suffix_ties),
            len(repairable),
        ),
        "multiple_min_suffix_alternative_rate_over_repairable_steps": safe_div(
            len(min_suffix_alt_ties),
            len(repairable),
        ),
        "avg_successful_logged_candidates_on_repairable_steps": average(
            float(row.get("successful_logged_candidates", 0)) for row in repairable
        ),
        "avg_successful_alternatives_on_alternative_steps": average(
            float(row.get("successful_alternatives", 0)) for row in alternative
        ),
    }


def min_suffix_tie_count(row: Mapping[str, Any], alternatives_only: bool = False) -> int:
    actions = [str(action) for action in row.get("successful_actions", ())]
    suffix_lens = [int(value) for value in row.get("successful_suffix_lens", ())]
    executed = normalize_action(str(row.get("executed_action", "")))
    pairs = [
        (action, suffix_len)
        for action, suffix_len in zip(actions, suffix_lens)
        if not alternatives_only or normalize_action(action) != executed
    ]
    if not pairs:
        return 0
    min_len = min(suffix_len for _action, suffix_len in pairs)
    return sum(1 for _action, suffix_len in pairs if suffix_len == min_len)


def dedupe_actions(actions: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for item in actions:
        action = str(item)
        norm = normalize_action(action)
        if norm in seen:
            continue
        seen.add(norm)
        output.append(action)
    return tuple(output)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def average(values: Any) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


if __name__ == "__main__":
    main()
