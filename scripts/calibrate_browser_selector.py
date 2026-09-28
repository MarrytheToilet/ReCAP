"""Freeze a learned browser gate from complete validation trajectories only.

Select gates by net task benefit, harms, failure-capped cost, intervention count,
and threshold. Deploy a gate with positive net benefit, or unchanged success
with zero harms and lower cost; otherwise freeze an abstaining selector.
--root contains protocol.json, trained_selector.json, validation/ and
learned_validation_{0.0,0.1,0.25}/. --output is a directory (default: --root).
No training, evaluation, or test-trajectory reads are performed.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


THRESHOLDS = (0.0, 0.1, 0.25)


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


def choose_gate(rows):
    eligible = [r for r in rows if r["net"] > 0 or (
        r["net"] == 0 and r["harms"] == 0 and r["cost_gain"] > 0)]
    return max(eligible, key=lambda r: (
        r["net"], -r["harms"], r["cost_gain"],
        -r["interventions"], r["threshold"]), default=None)


def calibrate(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root
    protocol = read_json(root / "protocol.json")
    tasks, seeds = protocol["tasks"], protocol["validation_seeds"]
    if not tasks or not seeds or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError("Validation tasks/seeds must be nonempty and unique")
    artifact = read_json(root / "trained_selector.json")
    tables = []
    for threshold in THRESHOLDS:
        gate = {**artifact, "threshold": threshold}
        digest = hashlib.sha256(json.dumps(gate, sort_keys=True).encode()).hexdigest()
        comparisons = []
        for task in tasks:
            for seed in seeds:
                raw = completed(root / "validation" / f"{task}_{seed}_raw_rank1.json")
                new = completed(root / f"learned_validation_{threshold}" / f"{task}_{seed}.json")
                for row in (raw, new):
                    if row.get("task") != task or row.get("seed") != seed:
                        raise ValueError("Validation task/seed mismatch")
                    if row["reward"] not in (0, 1):
                        raise ValueError("Expected binary browser reward")
                if "selector_sha256" in new and new["selector_sha256"] != digest:
                    raise ValueError("Validation selector hash mismatch")
                comparisons.append(dict(
                    task=task, seed=seed, raw=raw["reward"], new=new["reward"],
                    raw_cost=len(raw["records"]) if raw["reward"] else 12,
                    new_cost=len(new["records"]) if new["reward"] else 12,
                    intervened=new.get("intervened", False)))
        tables.append(dict(
            threshold=threshold, net=sum(r["new"] - r["raw"] for r in comparisons),
            harms=sum(r["new"] < r["raw"] for r in comparisons),
            cost_gain=sum(r["raw_cost"] - r["new_cost"] for r in comparisons),
            interventions=sum(r["intervened"] for r in comparisons), rows=comparisons))
    winner = choose_gate(tables)
    artifact["threshold"] = winner["threshold"] if winner else 2.0
    artifact["validation_selection"] = dict(
        winner=winner, all_gates=tables,
        note=f"Empirical gate selection over {len(tasks) * len(seeds)} validation instances, not a safety guarantee.")
    payload = json.dumps(artifact, indent=2)
    freeze = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  selected_threshold=artifact["threshold"],
                  test_instances=len(tasks) * len(protocol.get("test_seeds", [])))
    # Do not create or change outputs until every gate is complete.
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_selector.json").write_text(payload)
    (output / "selector_freeze.json").write_text(json.dumps(freeze, indent=2))
    return artifact, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/benchmarks/miniwob"))
    parser.add_argument("--output", type=Path, help="Output directory; defaults to --root")
    args = parser.parse_args()
    _, freeze = calibrate(args.root, args.output)
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
