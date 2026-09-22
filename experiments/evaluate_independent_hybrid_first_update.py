"""Evaluate the first unanchored update across independent distilled actors."""

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


REPLICATES = (
    "fold21_t101_s101",
    "fold22_t102_s102",
    "fold23_t103_s103",
)
FROZEN = "hybrid_frozen"
SOURCE = "hybrid_unanchored"
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
        "replicates_positive": int((values > 0).sum()),
        "replicates_negative": int((values < 0).sum()),
        "per_replicate_deltas": ";".join(
            f"{value:.9g}" for value in values
        ),
    }


def only_run_dir(base):
    runs = sorted(base.glob("run_*"))
    if len(runs) != 1:
        raise ValueError(f"Expected one run under {base}, got {runs}")
    return runs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiments/independent_hybrid_first_update_analysis_20260717",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    replicate_rows = []
    episode_rows = []
    deltas = {metric: [] for metric in METRICS}
    run_dirs = {}
    for replicate in REPLICATES:
        run_dir = only_run_dir(
            Path(
                f"experiments/independent_hybrid_online_"
                f"{replicate}_20260716"
            )
        )
        run_dirs[replicate] = run_dir
        metadata = json.loads(
            (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
        )
        seed = int(metadata["seeds"][0])
        frozen = sorted(
            read_csv(
                run_dir
                / FROZEN
                / f"seed_{seed}"
                / "per_evaluation_episode_metrics.csv"
            ),
            key=lambda row: int(row["test_seed"]),
        )
        test_seeds = [int(row["test_seed"]) for row in frozen]
        checkpoint = (
            run_dir
            / SOURCE
            / f"seed_{seed}"
            / "checkpoints"
            / "model_step100.pt"
        )
        candidate = evaluate_model(
            metadata["method_specs"][SOURCE]["env_id"],
            checkpoint,
            episodes=len(frozen),
            episode_length=25,
            seed=test_seeds[0],
        )
        if [int(row["test_seed"]) for row in candidate] != test_seeds:
            raise ValueError(f"Unmatched evaluation rows for {replicate}")

        summary = {
            "replicate": replicate,
            "method": OUTPUT_METHOD,
            "training_seed": seed,
            "checkpoint_step": 100,
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
        replicate_rows.append(summary)
        episode_rows.extend(
            {
                **row,
                "replicate": replicate,
                "method": OUTPUT_METHOD,
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
    write_csv(output / "replicate_summaries.csv", replicate_rows)
    write_csv(output / "replicate_intervals.csv", interval_rows)

    h = lookup["hungarian_assignment_distance"]
    auc = lookup["coverage_radius_auc"]
    collision = lookup["collision_step_rate"]
    lines = [
        "# Independent-Actor First-Update Result",
        "",
        "The independently distilled actor is the statistical unit. Intervals "
        "are paired 95% t intervals over three teacher/student replicates.",
        "",
        "| H delta [95% CI] | AUC delta [95% CI] | collision delta [95% CI] |",
        "| ---: | ---: | ---: |",
        (
            f"| {h['mean_delta']:+.4f} [{h['ci95_low']:+.4f}, "
            f"{h['ci95_high']:+.4f}] | {auc['mean_delta']:+.4f} "
            f"[{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}] | "
            f"{collision['mean_delta']:+.5f} "
            f"[{collision['ci95_low']:+.5f}, "
            f"{collision['ci95_high']:+.5f}] |"
        ),
    ]
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "run_dirs": {
                    key: str(value) for key, value in run_dirs.items()
                },
                "replicates": list(REPLICATES),
                "checkpoint_step": 100,
                "evaluation_episodes_per_replicate": len(frozen),
                "statistical_unit": "independent teacher/student actor",
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
