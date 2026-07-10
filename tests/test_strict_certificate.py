import json

from recap.eval.eval_strict_certificate import (
    audit_preference,
    load_reports,
    summarize,
)


def make_report(task_id: str, actions: list[str], suffix_lens: dict[int, int]) -> dict:
    edits = [
        {
            "edit_type": "policy_suffix",
            "index": index,
            "repair_suffix": ["a"] * length,
        }
        for index, length in suffix_lens.items()
    ]
    return {
        "task_id": task_id,
        "seed": 0,
        "original_success": False,
        "original_actions": actions,
        "edits": edits,
    }


def make_pref(task_id: str, step_index: int, repair_suffix_len: int) -> dict:
    return {
        "task_id": task_id,
        "seed": 0,
        "step_index": step_index,
        "history": ["go east"] * step_index,
        "preferred_action": "take key",
        "executed_action": "go east",
        "repair_suffix_len": repair_suffix_len,
        "source": "policy_repair_suffix",
    }


def reports_for(report: dict) -> dict:
    key = (report["task_id"], 0)
    return {key: [report]}


def test_strict_misrank_when_executed_action_makes_no_progress():
    # l*(s_1) = 5 via the repair; after executing, remaining cost is still 5.
    report = make_report("g1", ["go east", "go west"], {0: 5, 1: 5})
    row = audit_preference(make_pref("g1", 1, 5), reports_for(report))
    assert row["status"] == "strict_misrank"
    assert row["delta_steps"] == 1


def test_tie_when_executed_action_lies_on_equally_short_path():
    report = make_report("g2", ["go east", "go west"], {0: 5, 1: 4})
    row = audit_preference(make_pref("g2", 1, 5), reports_for(report))
    assert row["status"] == "tied_repair"
    assert row["delta_steps"] == 0


def test_verifier_suboptimality_when_cost_drops_by_more_than_one():
    report = make_report("g3", ["go east", "go west"], {0: 8, 1: 5})
    row = audit_preference(make_pref("g3", 1, 8), reports_for(report))
    assert row["status"] == "verifier_shorter_after_exec"
    assert row["delta_steps"] == -2


def test_missing_suffix_is_reported_not_guessed():
    report = make_report("g4", ["go east", "go west"], {0: 5})
    row = audit_preference(make_pref("g4", 1, 5), reports_for(report))
    assert row["status"] == "missing_suffix"


def test_unmatched_when_history_prefix_differs():
    report = make_report("g5", ["go north", "go west"], {0: 5, 1: 5})
    row = audit_preference(make_pref("g5", 1, 5), reports_for(report))
    assert row["status"] == "unmatched"


def test_summarize_counts_strict_pass_by_delta():
    reports = {}
    prefs = []
    for i, (l_state, l_exec) in enumerate([(5, 5), (5, 6), (5, 4), (8, 5)]):
        task = f"t{i}"
        reports[(task, 0)] = [
            make_report(task, ["go east", "go west"], {0: l_state, 1: l_exec})
        ]
        prefs.append(make_pref(task, 1, l_state))
    rows = [audit_preference(pref, reports) for pref in prefs]
    summary = summarize(rows, deltas=[0, 1, 2])
    assert summary["statuses"] == {
        "strict_misrank": 2,
        "tied_repair": 1,
        "verifier_shorter_after_exec": 1,
    }
    assert summary["strict_pass_by_delta"] == {"0": 3, "1": 2, "2": 1}


def test_load_reports_dedupes_identical_rollouts(tmp_path):
    report = make_report("g6", ["go east"], {0: 3, 1: 2})
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({"reports": [report]}))
    loaded = load_reports([str(tmp_path / "*.json")])
    assert len(loaded[("g6", 0)]) == 1
