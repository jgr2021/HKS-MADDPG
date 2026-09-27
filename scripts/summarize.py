"""Aggregate five selected-checkpoint evaluations per method (seed is the unit)."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def aggregate(records):
    methods = sorted({row["method"] for row in records})
    if not methods:
        raise ValueError("no evaluation summaries supplied")
    banks = {(r["seed_start"], r["eval_episodes"], r["episode_length"], r["device"]) for r in records}
    if len(banks) != 1:
        raise ValueError("all evaluations must share the final bank, horizon, and device")
    if next(iter(banks))[1:3] != (500, 25):
        raise ValueError("paper summary requires 500 episodes of 25 steps")
    curve_banks = {(r["curve_seed_start"], r["curve_episodes"]) for r in records}
    if len(curve_banks) != 1 or next(iter(curve_banks))[1] != 100:
        raise ValueError("all selections must share a 100-episode curve bank")
    result = []
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        if sorted(r["seed"] for r in rows) != [41, 42, 43, 44, 45]:
            raise ValueError(f"{method}: require exactly one summary per seed 41-45")
        if any(r["selection"] != "highest_mean_curve_return" for r in rows):
            raise ValueError("use selected-checkpoint evaluations, not final-step evaluations")
        for metric in ("return", "coverage_fraction", "full_coverage_rate", "collision_step_rate"):
            values = [r[metric] for r in rows]
            result.append(dict(method=method, metric=metric, n_seeds=5,
                               mean=statistics.mean(values), sample_std=statistics.stdev(values)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    if options.output.exists():
        parser.error(f"output already exists: {options.output}")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in options.summaries]
    rows = aggregate(records)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(options.output)


if __name__ == "__main__":
    main()
