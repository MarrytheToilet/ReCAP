from __future__ import annotations

from recap.eval.eval_candidate_ranking import evaluate_candidate_ranking, index_predictions, summarize_ranking
from recap.eval.bootstrap_candidate_ranking import bootstrap_candidate_ranking
from recap.eval.eval_candidate_ranking_by_action_type import evaluate_by_action_type
from recap.models.eval_exact_memory_reranker import predict_preferences as predict_exact_preferences
from recap.models.eval_heuristic_reranker import (
    build_heuristic_model,
    predict_preferences as predict_heuristic_preferences,
)
from recap.models.eval_nn_reranker import predict_preferences as predict_nn_preferences
from recap.models.eval_random_reranker import predict_random
from recap.models.eval_sklearn_reranker import predict_preferences as predict_sklearn_preferences
from recap.models.exact_memory_reranker import build_exact_memory_model
from recap.models.eval_action_reranker import predict_preferences
from recap.models.cross_encoder_reranker import context_text, pointwise_training_rows, rank_candidates as rank_cross_encoder_candidates
from recap.models.embedding_reranker import candidate_text_rows, joint_text
from recap.models.nn_memory_reranker import build_memory_model
from recap.models.policy_reranker import train_policy_reranker, rank_candidates as rank_policy_candidates
from recap.models.reranker_dataset import featurize_candidate, pairwise_delta
from recap.models.sklearn_reranker import fit_sklearn_reranker
from recap.models.train_action_reranker import train_feature_reranker


def preference() -> dict[str, object]:
    return {
        "task_id": "game.z8",
        "seed": 0,
        "step_index": 0,
        "history": (),
        "candidates": ["look", "open door"],
        "preferred_action": "open door",
        "rejected_action": "look",
        "certificate_level": "C3_failure_repair",
        "repair_suffix_len": 1,
    }


def test_feature_dataset_marks_rejected_and_candidate_rank() -> None:
    record = preference()

    look = featurize_candidate(record, "look")
    open_door = featurize_candidate(record, "open door")
    delta = pairwise_delta(record)

    assert look["is_rejected_action"] == 1.0
    assert look["raw_rank_reciprocal"] == 1.0
    assert open_door["raw_rank_reciprocal"] == 0.5
    assert delta["is_rejected_action"] == -1.0
    assert delta["verb_open"] == 1.0
    assert delta["verb_look"] == -1.0


def test_feature_reranker_trains_and_emits_learned_metrics() -> None:
    record = preference()
    model = train_feature_reranker(
        train_records=(record,),
        epochs=50,
        learning_rate=0.2,
        l2=0.0,
    )
    predictions = predict_preferences((record,), model)
    summary = summarize_ranking(
        evaluate_candidate_ranking((record,), index_predictions(tuple(predictions)))
    )

    assert model["summary"]["train_pairwise_accuracy"] == 1.0
    assert predictions[0]["ranked_actions"][0] == "open door"
    assert summary["raw_mrr"] == 0.5
    assert summary["learned_mrr"] == 1.0
    assert summary["learned_top1_correction_rate"] == 1.0


def test_feature_reranker_supports_listwise_loss() -> None:
    record = preference()
    model = train_feature_reranker(
        train_records=(record,),
        epochs=50,
        learning_rate=0.2,
        l2=0.0,
        loss="listwise",
    )
    predictions = predict_preferences((record,), model)

    assert model["hyperparameters"]["loss"] == "listwise"
    assert predictions[0]["ranked_actions"][0] == "open door"


def test_feature_reranker_feature_set_masks_weights() -> None:
    record = preference()
    model = train_feature_reranker(
        train_records=(record,),
        epochs=10,
        learning_rate=0.2,
        l2=0.0,
        feature_set="rank-only",
    )

    assert model["hyperparameters"]["feature_set"] == "rank-only"
    assert model["weights"]["verb_open"] == 0.0


def test_feature_reranker_abstention_suppresses_learned_metric() -> None:
    record = preference()
    model = train_feature_reranker(
        train_records=(record,),
        epochs=5,
        learning_rate=0.05,
        l2=0.0,
    )
    predictions = predict_preferences((record,), model, abstain_margin=999.0)
    summary = summarize_ranking(
        evaluate_candidate_ranking((record,), index_predictions(tuple(predictions)))
    )

    assert predictions[0]["abstain"] is True
    assert summary["learned_abstain_rate"] == 1.0
    assert summary["learned_mrr"] is None
    assert summary["learned_top1_correction_rate"] is None


def test_nn_memory_reranker_emits_learned_metrics() -> None:
    train = {
        "task_id": "train.z8",
        "seed": 0,
        "step_index": 0,
        "history": ("go east", "take key"),
        "candidates": ["open chest", "go west"],
        "preferred_action": "go west",
        "rejected_action": "open chest",
    }
    test = {
        "task_id": "test.z8",
        "seed": 0,
        "step_index": 0,
        "history": ("go east", "take key"),
        "candidates": ["open chest", "go west"],
        "preferred_action": "go west",
        "rejected_action": "open chest",
    }

    model = build_memory_model((train,), neighbors=1)
    predictions = predict_nn_preferences((test,), model)
    summary = summarize_ranking(
        evaluate_candidate_ranking((test,), index_predictions(tuple(predictions)))
    )

    assert model["summary"]["positive_exemplars"] == 1
    assert model["summary"]["negative_exemplars"] == 1
    assert predictions[0]["ranked_actions"][0] == "go west"
    assert summary["raw_mrr"] == 0.5
    assert summary["learned_mrr"] == 1.0
    assert summary["learned_top1_correction_rate"] == 1.0


def test_random_reranker_is_deterministic_and_emits_predictions() -> None:
    records = (preference(), preference())

    first = predict_random(records, seed=7)
    second = predict_random(records, seed=7)

    assert first == second
    assert len(first) == 2
    assert sorted(first[0]["ranked_actions"]) == ["look", "open door"]


def test_sklearn_mlp_reranker_emits_learned_metrics() -> None:
    train = (
        preference(),
        {
            "task_id": "train2.z8",
            "seed": 0,
            "step_index": 0,
            "history": (),
            "candidates": ["open door", "go west"],
            "preferred_action": "go west",
            "rejected_action": "open door",
            "certificate_level": "C3_failure_repair",
            "repair_suffix_len": 1,
        },
    )
    test = (
        preference(),
    )

    model = fit_sklearn_reranker(train, model_type="mlp", seed=0)
    predictions = predict_sklearn_preferences(test, model)
    summary = summarize_ranking(
        evaluate_candidate_ranking(test, index_predictions(tuple(predictions)))
    )

    assert model["summary"]["train_candidate_rows"] == 4
    assert len(predictions[0]["ranked_actions"]) == 2
    assert summary["learned_predictions"] == 1


def test_cross_encoder_text_includes_history_and_candidate_order() -> None:
    record = {
        "task_id": "game.z8",
        "seed": 0,
        "step_index": 2,
        "history": ("go east", "open door"),
        "candidates": ("look", "go west"),
        "preferred_action": "go west",
        "rejected_action": "look",
    }

    text = context_text(record)
    rows = pointwise_training_rows((record,))

    assert "Recent executed actions" in text
    assert "1. look" in text
    assert rows[0][2] == 0.0
    assert rows[1][2] == 1.0


def test_cross_encoder_text_includes_observations_when_available() -> None:
    record = {
        "task_id": "game.z8",
        "seed": 0,
        "step_index": 2,
        "initial_observation": "You must put the key on the table.",
        "observation": "You are in the kitchen. The table is here.",
        "history": ("take key",),
        "candidates": ("look", "put key on table"),
        "preferred_action": "put key on table",
        "rejected_action": "look",
    }

    text = context_text(record, max_observation_chars=120)

    assert "Objective/state excerpt" in text
    assert "put the key on the table" in text
    assert "Current observation excerpt" in text
    assert "The table is here" in text


def test_cross_encoder_text_omits_observations_by_default() -> None:
    record = {
        "task_id": "game.z8",
        "seed": 0,
        "step_index": 2,
        "initial_observation": "You must put the key on the table.",
        "observation": "You are in the kitchen. The table is here.",
        "history": ("take key",),
        "candidates": ("look", "put key on table"),
        "preferred_action": "put key on table",
        "rejected_action": "look",
    }

    text = context_text(record)

    assert "put the key on the table" not in text
    assert "The table is here" not in text


def test_cross_encoder_ranking_accepts_mock_predictor() -> None:
    class MockCrossEncoder:
        def predict(self, pairs, batch_size=16, show_progress_bar=False):
            return [1.0 if "open door" in pair[1] else 0.0 for pair in pairs]

    record = preference()
    ranked = rank_cross_encoder_candidates(record, MockCrossEncoder())

    assert ranked[0][0] == "open door"
    assert ranked[0][1] == 1.0


def test_embedding_reranker_rows_use_joint_decision_text() -> None:
    record = preference()

    text = joint_text(record, "open door")
    rows, labels = candidate_text_rows((record,))

    assert "Text-game decision state" in text
    assert "Proposed action: open door" in text
    assert len(rows) == 2
    assert labels.tolist() == [0, 1]


def test_policy_gradient_reranker_trains_policy_scores() -> None:
    record = preference()
    model = train_policy_reranker(
        (record,),
        hidden_dim=8,
        epochs=20,
        learning_rate=0.01,
        entropy_coef=0.0,
        seed=0,
    )
    ranked = rank_policy_candidates(record, model)

    assert model["model_type"] == "recap_support_constrained_policy"
    assert ranked[0][0] == "open door"


def test_exact_memory_only_fires_on_exact_history_action_match() -> None:
    train = preference()
    same = preference()
    different = dict(preference())
    different["history"] = ("go north",)

    model = build_exact_memory_model((train,))
    same_predictions = predict_exact_preferences((same,), model)
    different_predictions = predict_exact_preferences((different,), model)

    assert same_predictions[0]["ranked_actions"][0] == "open door"
    assert different_predictions[0]["ranked_actions"][0] == "look"


def test_heuristic_navigation_prior_is_simple_baseline() -> None:
    record = {
        "task_id": "game.z8",
        "seed": 0,
        "step_index": 0,
        "history": (),
        "candidates": ["open door", "go west", "look"],
        "preferred_action": "go west",
        "rejected_action": "open door",
    }
    model = build_heuristic_model("navigation-prior")
    predictions = predict_heuristic_preferences((record,), model)

    assert predictions[0]["ranked_actions"][0] == "go west"


def test_learned_verb_prior_uses_train_distribution() -> None:
    train = {
        "task_id": "train.z8",
        "seed": 0,
        "step_index": 0,
        "history": (),
        "candidates": ["open chest", "go west"],
        "preferred_action": "go west",
        "rejected_action": "open chest",
    }
    test = {
        "task_id": "test.z8",
        "seed": 0,
        "step_index": 0,
        "history": (),
        "candidates": ["open chest", "go south"],
        "preferred_action": "go south",
        "rejected_action": "open chest",
    }
    model = build_heuristic_model("learned-verb-prior", (train,))
    predictions = predict_heuristic_preferences((test,), model)

    assert predictions[0]["ranked_actions"][0] == "go south"


def test_candidate_ranking_by_action_type_splits_navigation() -> None:
    nav = {
        "task_id": "nav.z8",
        "seed": 0,
        "step_index": 0,
        "candidates": ["open door", "go west"],
        "preferred_action": "go west",
        "rejected_action": "open door",
    }
    non_nav = {
        "task_id": "item.z8",
        "seed": 0,
        "step_index": 0,
        "candidates": ["go west", "take key"],
        "preferred_action": "take key",
        "rejected_action": "go west",
    }

    output = evaluate_by_action_type((nav, non_nav))

    assert output["summary"]["navigation"]["preferences"] == 1
    assert output["summary"]["non_navigation"]["preferences"] == 1
    assert output["verb_counts"] == {"go": 1, "take": 1}


def test_bootstrap_candidate_ranking_includes_learned_metrics() -> None:
    record = preference()
    model = train_feature_reranker(
        train_records=(record,),
        epochs=50,
        learning_rate=0.2,
        l2=0.0,
    )
    predictions = tuple(predict_preferences((record,), model))

    output = bootstrap_candidate_ranking(
        preferences=(record,),
        predictions=predictions,
        iterations=5,
    )

    assert output["metrics"]["learned_mrr"]["point"] == 1.0
    assert output["metrics"]["learned_top1_correction_rate"]["point"] == 1.0
