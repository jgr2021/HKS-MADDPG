"""Evaluate a first-update-then-freeze control from registered checkpoints.

The Stage-B model_step100 checkpoint is saved immediately after the first
actor-update event.  If actor updates are disabled thereafter, the deployed
actor is exactly this checkpoint, so no retraining is required to evaluate the
early-stop policy on the registered final evaluation seeds.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_vector_signal_gsp_experiment import evaluate_model


SOURCE_METHOD = "hybrid_unanchored"
FROZEN_METHOD = "hybrid_frozen"
OUTPUT_METHOD = "hybrid_first_update_then_freeze"
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "minimum_agent_separation",
)
TCRIT_N3 = 4.302652729696142


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    half = TCRIT_N3 * values.std(ddof=1) / math.sqrt(len(values))
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(values.mean() - half),
        "ci95_high": float(values.mean() + half),
        "seeds_positive": int((values > 0).sum()),
        "seeds_negative": int((values < 0).sum()),
        "per_seed_deltas": ";".join(f"{value:.9g}" for value in values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=100)
    parser.add_argument(
        "--output-name",
        default="early_stop_first_update_analysis",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = run_dir / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    runner = json.loads(
        (run_dir / "runner_config.json").read_text(encoding="utf-8")
    )
    seeds = [int(seed) for seed in metadata["seeds"]]
    if len(seeds) != 3:
        raise ValueError(f"Expected three registered Stage-B seeds, got {seeds}")

    episode_rows = []
    seed_rows = []
    deltas = {metric: [] for metric in METRICS}
    for seed in seeds:
        frozen_path = (
            run_dir
            / FROZEN_METHOD
            / f"seed_{seed}"
            / "per_evaluation_episode_metrics.csv"
        )
        frozen = sorted(
            read_csv(frozen_path), key=lambda row: int(row["test_seed"])
        )
        test_seeds = [int(row["test_seed"]) for row in frozen]
        if test_seeds != list(range(test_seeds[0], test_seeds[0] + len(frozen))):
            raise ValueError(f"Evaluation seeds are not contiguous for seed {seed}")

        checkpoint = (
            run_dir
            / SOURCE_METHOD
            / f"seed_{seed}"
            / "checkpoints"
            / f"model_step{args.checkpoint_step}.pt"
        )
        candidate = evaluate_model(
            metadata["method_specs"][SOURCE_METHOD]["env_id"],
            checkpoint,
            episodes=len(frozen),
            episode_length=int(runner["episode_length"]),
            seed=test_seeds[0],
        )
        if [int(row["test_seed"]) for row in candidate] != test_seeds:
            raise ValueError(f"Unmatched early-stop evaluation rows for seed {seed}")

        summary = {
            "method": OUTPUT_METHOD,
            "training_seed": seed,
            "checkpoint_step": args.checkpoint_step,
            "episodes": len(candidate),
        }
        for metric in METRICS:
            reference = np.asarray(
                [float(row[metric]) for row in frozen], dtype=np.float64
            )
            values = np.asarray(
                [float(row[metric]) for row in candidate], dtype=np.float64
            )
            delta = float((values - reference).mean())
            summary[f"final_{metric}"] = float(values.mean())
            summary[f"delta_vs_frozen_{metric}"] = delta
            deltas[metric].append(delta)
        seed_rows.append(summary)
        episode_rows.extend(
            {
                **row,
                "method": OUTPUT_METHOD,
                "training_seed": seed,
                "source_checkpoint": str(checkpoint),
            }
            for row in candidate
        )

    interval_rows = [
        {
            "method": OUTPUT_METHOD,
            "comparison": "method_minus_frozen",
            "metric": metric,
            **interval(values),
        }
        for metric, values in deltas.items()
    ]
    lookup = {row["metric"]: row for row in interval_rows}

    write_csv(output / "per_episode.csv", episode_rows)
    write_csv(output / "seed_summaries.csv", seed_rows)
    write_csv(output / "training_seed_intervals.csv", interval_rows)
    lines = [
        "# First-Update-Then-Freeze Control",
        "",
        "The deployment actor is the registered unanchored model_step100 "
        "checkpoint, saved immediately after the first update event. It is "
        "evaluated on the exact 500 final seeds used by each frozen baseline.",
        "",
        "| H delta [95% CI] | AUC delta [95% CI] | collision delta [95% CI] |",
        "| ---: | ---: | ---: |",
    ]
    h = lookup["hungarian_assignment_distance"]
    auc = lookup["coverage_radius_auc"]
    collision = lookup["collision_step_rate"]
    lines.append(
        f"| {h['mean_delta']:+.4f} [{h['ci95_low']:+.4f}, "
        f"{h['ci95_high']:+.4f}] | {auc['mean_delta']:+.4f} "
        f"[{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}] | "
        f"{collision['mean_delta']:+.5f} "
        f"[{collision['ci95_low']:+.5f}, "
        f"{collision['ci95_high']:+.5f}] |"
    )
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "source_method": SOURCE_METHOD,
                "output_method": OUTPUT_METHOD,
                "checkpoint_step": args.checkpoint_step,
                "training_seeds": seeds,
                "evaluation_episodes_per_seed": len(frozen),
                "statistical_unit": "training_seed",
                "t_critical": TCRIT_N3,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
