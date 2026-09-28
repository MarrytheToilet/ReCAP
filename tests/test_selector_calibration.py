"""Verify complete-validation gate selection, abstention, and artifact freezing."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from scripts import calibrate_browser_selector as browser
from scripts import calibrate_tau2_selector as tau


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def browser_fixture(root, raw_reward=0, new_reward=1, raw_steps=3, new_steps=2):
    write(root / "protocol.json", dict(tasks=["task"], validation_seeds=[100], test_seeds=[2000]))
    model = dict(constant=1, threshold=.1)
    write(root / "trained_selector.json", model)
    raw = dict(status="completed", task="task", seed=100,
               reward=raw_reward, records=[{}] * raw_steps)
    write(root / "validation/task_100_raw_rank1.json", raw)
    for threshold in browser.THRESHOLDS:
        digest = hashlib.sha256(json.dumps({**model, "threshold": threshold}, sort_keys=True).encode()).hexdigest()
        write(root / f"learned_validation_{threshold}/task_100.json",
              {**raw, "reward": new_reward, "records": [{}] * new_steps,
               "intervened": True, "selector_sha256": digest})


def tau_fixture(root, raw_reward=0, new_reward=1, threshold=0.0):
    # Find two validation IDs using the original salt, plus a nonvalidation ID.
    ids = tau.validation_ids({"retail": {"train": [str(i) for i in range(100)]}})["retail"][:2]
    other = next(str(i) for i in range(100) if not tau.validation_ids(
        {"retail": {"train": [str(i)]}})["retail"])
    write(root / "full_baseline_manifest.json", {"splits": {
        "retail": {"train": ids + [other], "test": ["never-read-test"]}}})
    model = dict(constant=1, threshold=threshold, intervention_budget=1)
    folder = root / "tau2_learned_selector"
    write(folder / "selector.json", model)
    for task in ids:
        write(root / f"tau2_aliyun/retail_{task}_rank1.json",
              dict(status="completed", domain="retail", task_id=task, reward={"reward": raw_reward}))
    for index, value in enumerate(sorted({threshold, max(threshold, .1), max(threshold, .25)})):
        gate_path = folder / f"validation_gate{index}.json"
        write(gate_path, {**model, "threshold": value})
        digest = hashlib.sha256(gate_path.read_bytes()).hexdigest()
        for task in ids:
            write(root / f"tau2_learned_validation_gate{index}/retail_{task}_rank1.json",
                  dict(status="completed", domain="retail", task_id=task,
                       reward={"reward": new_reward}, decisions=[{"selected": 1}], selector_sha256=digest))
    return ids


def forbid_test_reads(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if "test" in path.name or any(part.endswith("_test") or part == "test" for part in path.parts):
            raise AssertionError(f"Accessed test trajectory: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def test_browser_freeze_uses_validation_only_and_highest_tied_gate(tmp_path, monkeypatch):
    root, output = tmp_path / "data", tmp_path / "frozen"
    browser_fixture(root)
    forbid_test_reads(monkeypatch)
    artifact, freeze = browser.calibrate(root, output)
    assert artifact["threshold"] == .25
    assert freeze["test_instances"] == 1
    assert freeze["sha256"] == hashlib.sha256((output / "selected_selector.json").read_bytes()).hexdigest()
    assert not (root / "selected_selector.json").exists()


@pytest.mark.parametrize("raw,new,old_steps,new_steps,expected", [
    (1, 1, 3, 2, .25),  # Equal success, lower capped cost: deploy.
    (1, 1, 2, 2, 2.0),  # No improvement: abstain.
    (0, 0, 3, 1, 2.0),  # Failed tasks both cost 12, regardless of length.
    (1, 0, 3, 1, 2.0),
])
def test_browser_eligibility(tmp_path, raw, new, old_steps, new_steps, expected):
    browser_fixture(tmp_path, raw, new, old_steps, new_steps)
    assert browser.calibrate(tmp_path)[0]["threshold"] == expected


@pytest.mark.parametrize("kind", ["missing", "error", "malformed"])
@pytest.mark.parametrize("which", ["raw", "last_gate"])
def test_browser_incomplete_never_overwrites_freeze(tmp_path, kind, which):
    browser_fixture(tmp_path)
    path = tmp_path / ("validation/task_100_raw_rank1.json" if which == "raw"
                       else "learned_validation_0.25/task_100.json")
    if kind == "missing":
        path.unlink()
    elif kind == "error":
        write(path, {"status": "error"})
    else:
        path.write_text("{")
    output = tmp_path / "frozen"
    write(output / "selector_freeze.json", {"sentinel": True})
    before = (output / "selector_freeze.json").read_bytes()
    with pytest.raises(ValueError, match="Incomplete validation"):
        browser.calibrate(tmp_path, output)
    assert (output / "selector_freeze.json").read_bytes() == before
    assert not (output / "selected_selector.json").exists()


def test_tau_freeze_validation_only_tie_and_repeatability(tmp_path, monkeypatch):
    ids = tau_fixture(tmp_path)
    forbid_test_reads(monkeypatch)
    artifact, freeze = tau.calibrate(tmp_path)
    assert artifact["threshold"] == .25
    assert artifact["intervention_budget"] == 1
    assert artifact["closed_loop_validation_selection"]["planned"] == 2
    assert freeze["closed_loop_validation_task_ids"] == {"retail": ids}
    path = tmp_path / "tau2_learned_selector/selector.json"
    assert freeze["selector_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tau.calibrate(tmp_path)[0] == artifact


def test_tau_deduplicates_gates_and_abstains_on_zero_gain(tmp_path):
    tau_fixture(tmp_path, 1, 1, threshold=.3)
    artifact, freeze = tau.calibrate(tmp_path, tmp_path / "frozen")
    assert len(freeze["gates"]) == 1
    assert artifact["threshold"] == 1.01
    assert not freeze["deploy_learned_gate"]


def test_tau_rejects_gate_from_different_training_artifact(tmp_path):
    tau_fixture(tmp_path)
    path = tmp_path / "tau2_learned_selector/validation_gate2.json"
    row = json.loads(path.read_text())
    write(path, {**row, "constant": -1})
    with pytest.raises(ValueError, match="Gate differs"):
        tau.calibrate(tmp_path, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_zero_net_with_harm_is_ineligible_even_when_browser_cost_improves():
    assert browser.choose_gate([dict(net=0, harms=1, cost_gain=100,
                                    interventions=2, threshold=.25)]) is None
    assert tau.choose_gate([dict(net=0, harms=0, intervened_tasks=0, threshold=.25)]) is None


@pytest.mark.parametrize("which", ["raw", "last_gate"])
@pytest.mark.parametrize("kind", ["missing", "error", "malformed"])
def test_tau_rejects_incomplete_not_best_of_partial(tmp_path, which, kind):
    ids = tau_fixture(tmp_path)
    folder = "tau2_aliyun" if which == "raw" else "tau2_learned_validation_gate2"
    path = tmp_path / folder / f"retail_{ids[-1]}_rank1.json"
    if kind == "missing":
        path.unlink()
    elif kind == "error":
        write(path, {"status": "error"})
    else:
        path.write_text("{")
    output = tmp_path / "frozen"
    with pytest.raises(ValueError, match="Incomplete validation"):
        tau.calibrate(tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize("module,fixture,record", [
    (browser, browser_fixture, "learned_validation_0.0/task_100.json"),
    (tau, tau_fixture, None),
])
def test_rejects_mismatched_selector_hash(tmp_path, module, fixture, record):
    ids = fixture(tmp_path)
    if record is None:
        record = f"tau2_learned_validation_gate0/retail_{ids[0]}_rank1.json"
    path = tmp_path / record
    row = json.loads(path.read_text())
    write(path, {**row, "selector_sha256": "wrong"})
    with pytest.raises(ValueError, match="hash mismatch"):
        module.calibrate(tmp_path, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize("module,base,changes", [
    (browser, dict(net=2, harms=1, cost_gain=0, interventions=2, threshold=.1),
     [dict(net=3, harms=9), dict(harms=0, cost_gain=-99),
      dict(cost_gain=1, interventions=99), dict(interventions=1, threshold=0), dict(threshold=.25)]),
    (tau, dict(net=2, harms=1, intervened_tasks=2, threshold=.1),
     [dict(net=3, harms=9), dict(harms=0, intervened_tasks=99),
      dict(intervened_tasks=1, threshold=0), dict(threshold=.25)]),
])
def test_original_tie_break_priority(module, base, changes):
    for change in changes:
        preferred = {**base, **change}
        assert module.choose_gate([base, preferred]) == preferred
        assert module.choose_gate([preferred, base]) == preferred


@pytest.mark.parametrize("module", [browser, tau])
def test_import_has_no_top_level_filesystem_reads(module, monkeypatch):
    spec = importlib.util.spec_from_file_location("calibration_import_check", module.__file__)
    fresh = importlib.util.module_from_spec(spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("Filesystem access at import time")

    for method in ("read_text", "read_bytes", "open", "exists", "glob", "iterdir"):
        monkeypatch.setattr(Path, method, forbidden)
    spec.loader.exec_module(fresh)
