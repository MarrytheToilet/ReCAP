from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from acta.agents import (
    LLMConfig,
    LearnedRerankerAgent,
    LocalLMPolicyAgent,
    MockLLMAgent,
    NoisyCandidateAgent,
    OpenAIChatAgent,
    PreferenceRerankAgent,
    RandomAgent,
    TraceFeedbackAgent,
)
from acta.agents.llm_agent import load_env_file
from acta.agents.learned_reranker_agent import load_reranker_model
from acta.agents.intervention_verifier import LLMInterventionVerifier, VerifierConfig
from acta.models.intervention_gate import load_gate_model
from acta.agents.preference_agent import load_action_preferences
from acta.controllers import ActAController, ControllerConfig
from acta.controllers import (
    ReplayRepairConfig,
    ReplayRepairController,
    ReplayVerifiedProposalController,
)
from acta.envs.textworld_adapter import TextWorldAdapter
from acta.eval.agent_loop import AgentEpisodeResult, run_episode
from acta.probe.trace_feedback import load_trace_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple agents with optional ActA control.")
    parser.add_argument("games", type=Path, nargs="+")
    parser.add_argument("--agent", choices=["random", "mock-llm", "openai", "lm-policy"], default="random")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--candidate-noise",
        choices=["none", "frontload-existing-structural", "frontload-structural"],
        default="none",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--lm-adapter", type=Path, default=None)
    parser.add_argument("--lm-candidate-pool-limit", type=int, default=20)
    parser.add_argument("--lm-max-length", type=int, default=384)
    parser.add_argument("--lm-max-history", type=int, default=12)
    parser.add_argument("--lm-max-observation-chars", type=int, default=220)
    parser.add_argument("--lm-candidate-chunk-size", type=int, default=2)
    parser.add_argument("--lm-load-in-4bit", action="store_true")
    parser.add_argument("--lm-recent-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--lm-inverse-penalty", type=float, default=0.0)
    parser.add_argument("--lm-static-penalty", type=float, default=0.0)
    parser.add_argument("--lm-semantic-undo-penalty", type=float, default=0.0)
    parser.add_argument("--lm-objective-overlap-bonus", type=float, default=0.0)
    parser.add_argument("--lm-navigation-bonus", type=float, default=0.0)
    parser.add_argument("--lm-nonobjective-manipulation-penalty", type=float, default=0.0)
    parser.add_argument("--lm-pool-ranker", choices=["default", "progress"], default="default")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm-max-retries", type=int, default=4)
    parser.add_argument("--llm-retry-base-delay", type=float, default=2.0)
    parser.add_argument("--llm-call-delay", type=float, default=0.0)
    parser.add_argument("--llm-cache", type=Path, default=None)
    parser.add_argument("--llm-cache-only", action="store_true",
                        help="Never call the API; on cache miss fall back to admissible-action heuristic.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--controller",
        choices=["none", "acta", "recap-replay", "recap-verified-proposal"],
        default="none",
    )
    parser.add_argument("--fast-controller", action="store_true")
    parser.add_argument("--noop-penalty", type=float, default=6.0)
    parser.add_argument("--seen-state-penalty", type=float, default=4.0)
    parser.add_argument("--recent-repeat-penalty", type=float, default=2.0)
    parser.add_argument("--absorbed-penalty", type=float, default=3.0)
    parser.add_argument("--inverse-penalty", type=float, default=4.0)
    parser.add_argument("--replay-repair-max-suffix", type=int, default=12)
    parser.add_argument("--replay-repair-max-proposals", type=int, default=None)
    parser.add_argument("--replay-repair-require-raw-failure", action="store_true")
    parser.add_argument("--replay-repair-min-suffix-improvement", type=int, default=1)
    parser.add_argument("--feedback-report", type=Path, default=None)
    parser.add_argument("--feedback-mode", choices=["safe", "oracle"], default="safe")
    parser.add_argument("--feedback-max-items", type=int, default=6)
    parser.add_argument("--feedback-source", choices=["all", "failures", "successes"], default="all")
    parser.add_argument("--preference-data", type=Path, default=None)
    parser.add_argument("--preference-bonus", type=float, default=100.0)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--reranker-margin", type=float, default=0.0)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-gate-model", type=Path, default=None)
    parser.add_argument("--reranker-gate-threshold", type=float, default=None)
    parser.add_argument("--reranker-verifier-model", default=None)
    parser.add_argument("--reranker-verifier-temperature", type=float, default=0.0)
    parser.add_argument("--reranker-verifier-cache", type=Path, default=None)
    parser.add_argument(
        "--reranker-safety-mode",
        choices=["none", "no-repeat", "conservative", "loop-safe", "loop-break"],
        default="none",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--episodes-out", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--episode-delay", type=float, default=0.0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    args.feedback_by_task = (
        load_trace_feedback(
            args.feedback_report,
            mode=args.feedback_mode,
            max_items=args.feedback_max_items,
            source=args.feedback_source,
        )
        if args.feedback_report is not None
        else {}
    )
    args.action_preferences = (
        load_action_preferences(args.preference_data)
        if args.preference_data is not None
        else ()
    )
    args.reranker_model_payload = (
        load_reranker_model(args.reranker_model, device=args.reranker_device)
        if args.reranker_model is not None
        else None
    )
    args.reranker_gate_payload = load_gate_model(args.reranker_gate_model)
    args.reranker_verifier_payload = (
        LLMInterventionVerifier(
            VerifierConfig(
                model=args.reranker_verifier_model,
                temperature=args.reranker_verifier_temperature,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                cache_path=args.reranker_verifier_cache,
                max_retries=args.llm_max_retries,
                retry_base_delay=args.llm_retry_base_delay,
                call_delay=args.llm_call_delay,
            )
        )
        if args.reranker_verifier_model is not None
        else None
    )
    adapter = TextWorldAdapter()
    controller = build_controller(args, adapter)
    results: list[AgentEpisodeResult] = []
    episode_writer = None
    if args.episodes_out is not None:
        args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
        episode_writer = args.episodes_out.open("w", encoding="utf-8")

    try:
        shared_agent = build_agent(args, 0) if args.agent == "lm-policy" else None
        for index, game in enumerate(args.games):
            agent = shared_agent if shared_agent is not None else build_agent(args, index)
            try:
                result = run_episode(
                    adapter=adapter,
                    agent=agent,
                    task_id=str(game),
                    seed=args.seed,
                    max_steps=args.max_steps,
                    controller=controller,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                result = make_error_result(
                    adapter=adapter,
                    task_id=str(game),
                    seed=args.seed,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            if episode_writer is not None:
                episode_writer.write(json.dumps(to_jsonable(result), ensure_ascii=False) + "\n")
                episode_writer.flush()
            if args.progress:
                error_suffix = f" error={result.error}" if result.error else ""
                print(
                    f"episode={index + 1}/{len(args.games)} game={game} "
                    f"success={result.success} steps={len(result.steps)} "
                    f"reward={result.total_reward}{error_suffix}",
                    flush=True,
                )
            if args.episode_delay > 0 and index + 1 < len(args.games):
                time.sleep(args.episode_delay)
    finally:
        if episode_writer is not None:
            episode_writer.close()

    summary = summarize(results, controller=args.controller)
    summary["agent"] = args.agent
    if args.agent == "openai":
        summary["model"] = args.model or os.environ.get("ACTA_LLM_MODEL")
    if controller is not None:
        summary["controller_cache_hits"] = dict(controller.cache_hits)
        summary["controller_cache_misses"] = dict(controller.cache_misses)
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(to_jsonable({"summary": summary, "episodes": results}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote={args.out}")


def make_error_result(
    adapter: TextWorldAdapter,
    task_id: str,
    seed: int,
    error: str,
) -> AgentEpisodeResult:
    try:
        reset = adapter.reset(task_id=task_id, seed=seed)
        final_signature = adapter.signature(reset.state, mode="full")
    except Exception:
        final_signature = ("episode_error", error)
    return AgentEpisodeResult(
        task_id=task_id,
        seed=seed,
        success=False,
        total_reward=0.0,
        steps=(),
        final_signature=final_signature,
        metrics={"steps": 0.0, "episode_errors": 1.0},
        error=error,
    )


def build_controller_config(args: argparse.Namespace) -> ControllerConfig:
    return ControllerConfig(
        noop_penalty=args.noop_penalty,
        seen_state_penalty=args.seen_state_penalty,
        recent_repeat_penalty=args.recent_repeat_penalty,
        absorbed_penalty=args.absorbed_penalty,
        inverse_penalty=args.inverse_penalty,
        enable_pair_penalties=not args.fast_controller,
    )


def build_controller(args: argparse.Namespace, adapter: TextWorldAdapter) -> Any:
    if args.controller == "acta":
        return ActAController(
            adapter,
            env_name="textworld",
            config=build_controller_config(args),
        )
    if args.controller == "recap-replay":
        return ReplayRepairController(
            adapter,
            config=ReplayRepairConfig(
                max_suffix_steps=args.replay_repair_max_suffix,
                max_proposals_to_verify=args.replay_repair_max_proposals,
                require_raw_failure=args.replay_repair_require_raw_failure,
                min_suffix_improvement=args.replay_repair_min_suffix_improvement,
            ),
        )
    if args.controller == "recap-verified-proposal":
        return ReplayVerifiedProposalController(
            adapter,
            config=ReplayRepairConfig(
                max_suffix_steps=args.replay_repair_max_suffix,
                max_proposals_to_verify=args.replay_repair_max_proposals,
                require_raw_failure=args.replay_repair_require_raw_failure,
                min_suffix_improvement=args.replay_repair_min_suffix_improvement,
            ),
        )
    return None


def summarize(results: list[AgentEpisodeResult], controller: str) -> dict[str, Any]:
    episodes = len(results)
    metric_keys = sorted({key for result in results for key in result.metrics})
    metrics = {
        key: sum(result.metrics.get(key, 0.0) for result in results) / episodes
        if episodes
        else 0.0
        for key in metric_keys
    }
    return {
        "controller": controller,
        "episodes": episodes,
        "success_rate": sum(result.success for result in results) / episodes if episodes else 0.0,
        "avg_reward": sum(result.total_reward for result in results) / episodes if episodes else 0.0,
        **{f"avg_{key}": value for key, value in metrics.items()},
    }


def build_agent(args: argparse.Namespace, index: int) -> Any:
    max_candidates = args.max_candidates or 5
    if args.agent == "random":
        agent = RandomAgent(seed=args.agent_seed + index, max_candidates=args.max_candidates)
    elif args.agent == "mock-llm":
        agent = MockLLMAgent(max_candidates=max_candidates)
    elif args.agent == "openai":
        model = args.model or os.environ.get("ACTA_LLM_MODEL")
        if model is None:
            raise SystemExit("--model or ACTA_LLM_MODEL is required when --agent openai")
        agent = OpenAIChatAgent(
            LLMConfig(
                model=model,
                max_candidates=max_candidates,
                temperature=args.temperature,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                max_retries=args.llm_max_retries,
                retry_base_delay=args.llm_retry_base_delay,
                call_delay=args.llm_call_delay,
                cache_path=args.llm_cache,
                cache_only=args.llm_cache_only,
            )
        )
    elif args.agent == "lm-policy":
        if args.model is None:
            raise SystemExit("--model must point to a local HF model when --agent lm-policy")
        agent = LocalLMPolicyAgent(
            base_model=Path(args.model),
            adapter=args.lm_adapter,
            max_candidates=max_candidates,
            candidate_pool_limit=args.lm_candidate_pool_limit,
            max_length=args.lm_max_length,
            max_history=args.lm_max_history,
            max_observation_chars=args.lm_max_observation_chars,
            candidate_chunk_size=args.lm_candidate_chunk_size,
            recent_repeat_penalty=args.lm_recent_repeat_penalty,
            inverse_penalty=args.lm_inverse_penalty,
            static_penalty=args.lm_static_penalty,
            semantic_undo_penalty=args.lm_semantic_undo_penalty,
            objective_overlap_bonus=args.lm_objective_overlap_bonus,
            navigation_bonus=args.lm_navigation_bonus,
            nonobjective_manipulation_penalty=args.lm_nonobjective_manipulation_penalty,
            pool_ranker=args.lm_pool_ranker,
            device=args.reranker_device or "cuda",
            load_in_4bit=args.lm_load_in_4bit,
            model_type="local_lm_policy_rl" if args.lm_adapter else "local_lm_policy_base",
        )
    else:
        raise ValueError(f"unknown agent: {args.agent}")

    if args.candidate_noise != "none":
        if getattr(args, "feedback_by_task", None):
            agent = TraceFeedbackAgent(agent, args.feedback_by_task)
        agent = NoisyCandidateAgent(
            agent,
            mode=args.candidate_noise,
            max_candidates=args.max_candidates,
        )
        if getattr(args, "action_preferences", ()):
            agent = PreferenceRerankAgent(
                agent,
                args.action_preferences,
                bonus=args.preference_bonus,
            )
        return agent
    if getattr(args, "feedback_by_task", None):
        agent = TraceFeedbackAgent(agent, args.feedback_by_task)
    if getattr(args, "action_preferences", ()):
        agent = PreferenceRerankAgent(
            agent,
            args.action_preferences,
            bonus=args.preference_bonus,
        )
    if getattr(args, "reranker_model_payload", None) is not None:
        agent = LearnedRerankerAgent(
            agent,
            args.reranker_model_payload,
            abstain_margin=args.reranker_margin,
            safety_mode=args.reranker_safety_mode,
            gate_model=args.reranker_gate_payload,
            gate_threshold=args.reranker_gate_threshold,
            intervention_verifier=args.reranker_verifier_payload,
        )
    return agent


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_jsonable(getattr(value, key))
            for key in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
