"""Freeze a learned tau2 gate from complete closed-loop validation outputs.

Evaluate every gate on the complete hash-selected training holdout. Select
positive net task benefit, breaking ties by fewer harms, fewer interventions,
then higher threshold; otherwise freeze an abstaining selector.
Inputs under --root: full_baseline_manifest.json, tau2_aliyun/,
tau2_learned_validation_gateN/, and tau2_learned_selector/ containing the
train-produced proposed_selector.json (or uncalibrated selector.json) and
validation_gateN.json. --output is a directory, defaulting to the selector folder.
No training, subprocesses, deadlines, or test-trajectory reads are performed.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def completed(path):
    try:
        row = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Incomplete validation: {path}") from exc
    if row.get("status") != "completed":
        raise ValueError(f"Incomplete validation: {path}")
    return row


def validation_ids(splits):
    return {domain: [task for task in split["train"] if int(hashlib.sha256(
        ("tau2-selector-v1:" + domain + ":" + task).encode()).hexdigest()[:8], 16) % 5 == 0]
        for domain, split in splits.items()}


def choose_gate(rows):
    return max((r for r in rows if r["net"] > 0), key=lambda r: (
        r["net"], -r["harms"], -r["intervened_tasks"], r["threshold"]), default=None)


def calibrate(root, output=None):
    root = Path(root)
    folder = root / "tau2_learned_selector"
    output = Path(output) if output is not None else folder
    splits = read_json(root / "full_baseline_manifest.json")["splits"]
    ids = validation_ids(splits)
    planned = sum(map(len, ids.values()))
    if not planned or any(len(ts) != len(set(ts)) for ts in ids.values()):
        raise ValueError("Validation tasks must be nonempty and unique")
    source = folder / "proposed_selector.json"
    if not source.exists():
        source = folder / "selector.json"
    proposed_bytes = source.read_bytes()
    model = json.loads(proposed_bytes)
    if "closed_loop_validation_selection" in model:
        raise ValueError("Need train-produced model, not an already calibrated selector")
    threshold = model["threshold"]
    thresholds = sorted({threshold, max(threshold, .1), max(threshold, .25)})
    gates, table, gate_bytes = [], [], {}
    for index, threshold in enumerate(thresholds):
        stage = f"validation_gate{index}"
        path = folder / f"{stage}.json"
        payload = path.read_bytes()
        if json.loads(payload) != {**model, "threshold": threshold}:
            raise ValueError(f"Gate differs from train-produced selector: {path}")
        digest = hashlib.sha256(payload).hexdigest()
        gate_bytes[path.name] = payload
        gate = dict(stage=stage, artifact=str(output / path.name), threshold=threshold, sha256=digest)
        gates.append(gate)
        rows = []
        for domain, tasks in ids.items():
            for task in tasks:
                name = f"{domain}_{task}_rank1.json"
                raw = completed(root / "tau2_aliyun" / name)
                new = completed(root / f"tau2_learned_{stage}" / name)
                for row in (raw, new):
                    if row.get("domain") != domain or row.get("task_id") != task:
                        raise ValueError("Validation task/domain mismatch")
                if "selector_sha256" in new and new["selector_sha256"] != digest:
                    raise ValueError("Validation selector hash mismatch")
                rows.append(dict(domain=domain, task_id=task,
                                 raw=int(raw["reward"]["reward"] == 1),
                                 learned=int(new["reward"]["reward"] == 1),
                                 intervened=any(r.get("selected", 0) != 0 for r in new["decisions"])))
        rescues = sum(r["learned"] > r["raw"] for r in rows)
        harms = sum(r["learned"] < r["raw"] for r in rows)
        table.append(dict(**gate, planned=planned, scored=len(rows), missing=[],
                          rescues=rescues, harms=harms, net=rescues - harms,
                          intervened_tasks=sum(r["intervened"] for r in rows), rows=rows))
    winner = choose_gate(table)
    chosen = {**model, "threshold": winner["threshold"] if winner else 1.01}
    chosen["closed_loop_validation_selection"] = dict(
        deploy_learned_gate=winner is not None,
        selected_stage=winner["stage"] if winner else "abstain",
        rescues=winner["rescues"] if winner else 0,
        harms=winner["harms"] if winner else 0,
        net=winner["net"] if winner else 0, planned=planned,
        gate_search=[{k: v[k] for k in ("stage", "threshold", "rescues", "harms", "net", "intervened_tasks")} for v in table],
        rule="Fixed thresholds={prefix gate,max(gate,.1),max(gate,.25)}, deduplicated. "
             f"All {planned} internal-validation tasks required. Maximize positive net task rescues; "
             "break ties by fewer harms, fewer interventions, higher threshold. "
             "Otherwise abstain. Empirical selection, not a safety guarantee.")
    payload = json.dumps(chosen, indent=2)
    freeze = dict(stage="final_selector_frozen",
                  frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  artifact=str(output / "selector.json"),
                  selector_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  deploy_learned_gate=winner is not None,
                  closed_loop_validation_task_ids=ids, gates=gates,
                  heldout_task_count=sum(len(s.get("test", [])) for s in splits.values()))
    # Preserve the training proposal so in-place calibration is repeatable.
    output.mkdir(parents=True, exist_ok=True)
    (output / "proposed_selector.json").write_bytes(proposed_bytes)
    for name, content in gate_bytes.items():
        (output / name).write_bytes(content)
    (output / "closed_loop_validation.json").write_text(json.dumps(table, indent=2))
    (output / "selector.json").write_text(payload)
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2))
    return chosen, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/experiments/new_benchmarks"))
    parser.add_argument("--output", type=Path, help="Output directory; defaults to --root/tau2_learned_selector")
    args = parser.parse_args()
    _, freeze = calibrate(args.root, args.output)
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
