"""Evaluate arbitrary MADDPG checkpoints on exactly matched initializations."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_vector_signal_gsp_experiment import METRICS, evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH; repeat for every checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-base", type=int, default=3_700_000)
    parser.add_argument("--horizon", type=int, default=25)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in args.checkpoint:
        label, path_text = item.split("=", 1)
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        evaluated = evaluate_model(
            "simple_spread", path, args.episodes, args.horizon, args.seed_base
        )
        for row in evaluated:
            row["checkpoint"] = label
        rows.extend(evaluated)
    with (output / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    labels = [item.split("=", 1)[0] for item in args.checkpoint]
    summary = []
    for label in labels:
        selected = [row for row in rows if row["checkpoint"] == label]
        result = {"checkpoint": label, "episodes": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_std"] = float(values.std(ddof=1))
        summary.append(result)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    baseline = labels[0]
    baseline_rows = sorted(
        (row for row in rows if row["checkpoint"] == baseline),
        key=lambda row: row["test_seed"],
    )
    paired = []
    for label in labels[1:]:
        candidate_rows = sorted(
            (row for row in rows if row["checkpoint"] == label),
            key=lambda row: row["test_seed"],
        )
        for metric in METRICS:
            delta = np.asarray([
                candidate[metric] - reference[metric]
                for reference, candidate in zip(baseline_rows, candidate_rows)
            ], dtype=np.float64)
            half = 1.9647293909876649 * delta.std(ddof=1) / np.sqrt(len(delta))
            paired.append({
                "comparison": f"{label}_vs_{baseline}", "metric": metric,
                "mean_delta": float(delta.mean()),
                "ci95_low": float(delta.mean() - half),
                "ci95_high": float(delta.mean() + half),
                "fraction_positive": float((delta > 0).mean()),
            })
    with (output / "paired_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    (output / "metadata.json").write_text(json.dumps({
        "checkpoints": args.checkpoint, "episodes": args.episodes,
        "seed_base": args.seed_base, "deterministic": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
