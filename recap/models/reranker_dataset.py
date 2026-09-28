from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


COMMON_VERBS = (
    "go",
    "take",
    "drop",
    "put",
    "open",
    "close",
    "unlock",
    "lock",
    "look",
    "inventory",
    "examine",
    "read",
    "eat",
    "insert",
)

FEATURE_NAMES = (
    "raw_rank_reciprocal",
    "raw_rank_from_bottom",
    "is_raw_top1",
    "is_rejected_action",
    "is_recent_repeat",
    "is_look_inventory_examine",
    "is_navigation",
    "is_manipulation",
    "action_token_count",
    "overlap_with_recent_history",
    *(f"verb_{verb}" for verb in COMMON_VERBS),
)

RANK_FEATURES = frozenset(
    {
        "raw_rank_reciprocal",
        "raw_rank_from_bottom",
        "is_raw_top1",
        "is_rejected_action",
    }
)
HISTORY_FEATURES = frozenset(
    {
        "is_recent_repeat",
        "overlap_with_recent_history",
    }
)
ACTION_TYPE_FEATURES = frozenset(
    {
        "is_look_inventory_examine",
        "is_navigation",
        "is_manipulation",
        "action_token_count",
        *(f"verb_{verb}" for verb in COMMON_VERBS),
    }
)

FEATURE_SETS = ("all", "rank-only", "action-only", "no-rank", "no-action-type", "no-history")


def active_feature_names(feature_set: str = "all") -> tuple[str, ...]:
    if feature_set == "all":
        return FEATURE_NAMES
    if feature_set == "rank-only":
        return tuple(name for name in FEATURE_NAMES if name in RANK_FEATURES)
    if feature_set == "action-only":
        return tuple(name for name in FEATURE_NAMES if name in ACTION_TYPE_FEATURES)
    if feature_set == "no-rank":
        return tuple(name for name in FEATURE_NAMES if name not in RANK_FEATURES)
    if feature_set == "no-action-type":
        return tuple(name for name in FEATURE_NAMES if name not in ACTION_TYPE_FEATURES)
    if feature_set == "no-history":
        return tuple(name for name in FEATURE_NAMES if name not in HISTORY_FEATURES)
    raise ValueError(f"unknown feature set: {feature_set}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def score_candidate(
    preference: Mapping[str, Any],
    action: str,
    weights: Mapping[str, float],
) -> float:
    features = featurize_candidate(preference, action)
    return sum(float(weights.get(name, 0.0)) * value for name, value in features.items())


def featurize_candidate(preference: Mapping[str, Any], action: str) -> dict[str, float]:
    candidates = tuple(str(candidate) for candidate in preference.get("candidates", ()))
    history = tuple(str(item) for item in preference.get("history", ()))
    rejected = str(preference.get("rejected_action", ""))
    rank = candidate_rank(candidates, action)
    candidate_count = len(candidates)
    action_norm = normalize_action(action)
    verb = action_verb(action_norm)
    tokens = tokenize(action_norm)
    recent_tokens = set(tokenize(" ".join(history[-4:])))
    overlap = len(set(tokens) & recent_tokens) / len(set(tokens)) if tokens else 0.0
    features = {
        "raw_rank_reciprocal": 1.0 / rank if rank else 0.0,
        "raw_rank_from_bottom": (
            (candidate_count - rank) / max(candidate_count - 1, 1) if rank else 0.0
        ),
        "is_raw_top1": 1.0 if rank == 1 else 0.0,
        "is_rejected_action": 1.0 if action == rejected else 0.0,
        "is_recent_repeat": 1.0 if action in history[-4:] else 0.0,
        "is_look_inventory_examine": (
            1.0 if verb in {"look", "inventory", "examine"} else 0.0
        ),
        "is_navigation": 1.0 if verb == "go" else 0.0,
        "is_manipulation": (
            1.0
            if verb in {"take", "drop", "put", "open", "close", "unlock", "lock", "insert"}
            else 0.0
        ),
        "action_token_count": math.log1p(len(tokens)),
        "overlap_with_recent_history": overlap,
    }
    for common_verb in COMMON_VERBS:
        features[f"verb_{common_verb}"] = 1.0 if verb == common_verb else 0.0
    return features


def pairwise_delta(preference: Mapping[str, Any]) -> dict[str, float]:
    preferred = str(preference["preferred_action"])
    rejected = str(preference["rejected_action"])
    return candidate_pairwise_delta(preference, preferred, rejected)


def pairwise_deltas(
    preference: Mapping[str, Any],
    negative_scope: str = "all-candidates",
) -> tuple[dict[str, float], ...]:
    preferred = str(preference["preferred_action"])
    if negative_scope == "rejected":
        return (pairwise_delta(preference),)
    if negative_scope != "all-candidates":
        raise ValueError(f"unknown negative scope: {negative_scope}")
    negatives = [
        str(candidate)
        for candidate in preference.get("candidates", ())
        if str(candidate) != preferred
    ]
    if not negatives:
        negatives = [str(preference["rejected_action"])]
    return tuple(
        candidate_pairwise_delta(preference, preferred, negative)
        for negative in negatives
    )


def candidate_pairwise_delta(
    preference: Mapping[str, Any],
    preferred: str,
    negative: str,
) -> dict[str, float]:
    positive = featurize_candidate(preference, preferred)
    negative_features = featurize_candidate(preference, negative)
    return {
        name: positive.get(name, 0.0) - negative_features.get(name, 0.0)
        for name in FEATURE_NAMES
    }


def certificate_weight(preference: Mapping[str, Any]) -> float:
    level = str(preference.get("certificate_level", ""))
    if level.startswith("C4"):
        base = 2.0
    elif level.startswith("C3"):
        base = 1.5
    elif level.startswith("C2"):
        base = 1.25
    elif level.startswith("C1"):
        base = 1.0
    else:
        base = 0.5
    suffix_len = preference.get("repair_suffix_len")
    if suffix_len is None:
        return base
    return base / (1.0 + 0.02 * max(float(suffix_len), 0.0))


def rank_candidates(
    preference: Mapping[str, Any],
    weights: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    candidates = tuple(str(candidate) for candidate in preference.get("candidates", ()))
    indexed = tuple(enumerate(candidates))
    scored = [
        (index, action, score_candidate(preference, action, weights))
        for index, action in indexed
    ]
    scored.sort(key=lambda item: (-item[2], item[0]))
    return tuple((action, score) for _index, action, score in scored)


def candidate_rank(candidates: tuple[str, ...], action: str) -> int | None:
    try:
        return candidates.index(action) + 1
    except ValueError:
        return None


def action_verb(action: str) -> str:
    tokens = action.split()
    return tokens[0] if tokens else ""


def normalize_action(action: str) -> str:
    return re.sub(r"\s+", " ", action.strip().lower())


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))
