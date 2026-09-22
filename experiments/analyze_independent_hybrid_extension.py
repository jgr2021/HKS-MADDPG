"""Five-actor analysis for independently distilled hybrid policies."""

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


REPLICATES = {
    "fold21_t101_s101": ("20260716", 201),
    "fold22_t102_s102": ("20260716", 202),
    "fold23_t103_s103": ("20260716", 203),
    "fold24_t104_s104": ("20260811", 204),
    "fold25_t105_s105": ("20260811", 205),
}
METHODS = (
    "hybrid_frozen",
    "hybrid_unanchored",
    "hybrid_fixed_teacher_kl002",
)
FROZEN = "hybrid_frozen"
UNANCHORED = "hybrid_unanchored"
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "minimum_agent_separation",
    "return",
)
EXPECTED_CHECKPOINTS = [64, 100, 1000, 5000, 10000, 20000]
TCRIT_N5 = 2.7764451051977987


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def only_run_dir(base):
    runs = sorted(base.glob("run_*"))
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one run under {base}, got {runs}")
    return runs[0]


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    half = TCRIT_N5 * values.std(ddof=1) / math.sqrt(len(values))
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(values.mean() - half),
        "ci95_high": float(values.mean() + half),
        "replicates_positive": int((values > 0).sum()),
        "replicates_negative": int((values < 0).sum()),
        "per_replicate_deltas": ";".join(f"{v:.9g}" for v in values),
    }


def summarize(candidate, reference):
    summary = {"episodes": len(candidate)}
    for metric in METRICS:
        values = np.asarray(
            [float(row[metric]) for row in candidate], dtype=np.float64
        )
        frozen = np.asarray(
            [float(row[metric]) for row in reference], dtype=np.float64
        )
        summary[f"final_{metric}"] = float(values.mean())
        summary[f"delta_vs_frozen_{metric}"] = float(
            (values - frozen).mean()
        )
    return summary


def make_intervals(rows, phase):
    output = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        for metric in METRICS:
            output.append({
                "phase": phase,
                "method": method,
                "comparison": "method_minus_frozen",
                "metric": metric,
                **interval([
                    row[f"delta_vs_frozen_{metric}"] for row in selected
                ]),
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiments/independent_hybrid_extension_analysis_20260811",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    final_rows = []
    first_rows = []
    run_dirs = {}
    protocol_ok = True
    for replicate, (artifact_date, expected_seed) in REPLICATES.items():
        run_dir = only_run_dir(
            Path("experiments")
            / f"independent_hybrid_online_{replicate}_{artifact_date}"
        )
        run_dirs[replicate] = str(run_dir)
        metadata = json.loads(
            (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
        )
        seed = int(metadata["seeds"][0])
        protocol_ok &= seed == expected_seed
        protocol_ok &= int(metadata["total_env_steps"]) == 20_000
        protocol_ok &= metadata["methods"] == list(METHODS)

        per_method = {}
        for method in METHODS:
            seed_dir = run_dir / method / f"seed_{seed}"
            counters = json.loads(
                (seed_dir / "training_counters.json").read_text(
                    encoding="utf-8"
                )
            )
            protocol_ok &= counters["saved_checkpoints"] == EXPECTED_CHECKPOINTS
            per_method[method] = sorted(
                read_csv(seed_dir / "per_evaluation_episode_metrics.csv"),
                key=lambda row: int(row["test_seed"]),
            )

        frozen = per_method[FROZEN]
        frozen_seeds = [int(row["test_seed"]) for row in frozen]
        for method in METHODS:
            candidate = per_method[method]
            if [int(row["test_seed"]) for row in candidate] != frozen_seeds:
                raise ValueError(f"Unmatched final rows: {replicate}, {method}")
            final_rows.append({
                "replicate": replicate,
                "method": method,
                "training_seed": seed,
                **summarize(candidate, frozen),
            })

        checkpoint = (
            run_dir / UNANCHORED / f"seed_{seed}" / "checkpoints"
            / "model_step100.pt"
        )
        first = evaluate_model(
            metadata["method_specs"][UNANCHORED]["env_id"],
            checkpoint,
            episodes=len(frozen),
            episode_length=25,
            seed=frozen_seeds[0],
        )
        if [int(row["test_seed"]) for row in first] != frozen_seeds:
            raise ValueError(f"Unmatched first-update rows: {replicate}")
        first_rows.append({
            "replicate": replicate,
            "method": "hybrid_first_update_then_freeze",
            "training_seed": seed,
            "checkpoint_step": 100,
            **summarize(first, frozen),
        })

    interval_rows = (
        make_intervals(final_rows, "step20000")
        + make_intervals(first_rows, "first_update_step100")
    )
    write_csv(output / "final_replicate_summaries.csv", final_rows)
    write_csv(output / "first_update_replicate_summaries.csv", first_rows)
    write_csv(output / "replicate_intervals.csv", interval_rows)

    lookup = {
        (row["phase"], row["method"], row["metric"]): row
        for row in interval_rows
    }
    lines = [
        "# Five-Actor Independent-Distillation Extension",
        "",
        "The independent distilled actor is the statistical unit. All "
        "intervals are paired two-sided 95% t intervals over five actors.",
        "",
        f"Protocol integrity: **{'PASS' if protocol_ok else 'FAIL'}**.",
        "",
        "| phase | method | H delta [95% CI] | AUC delta [95% CI] | "
        "collision delta [95% CI] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for phase, method in (
        ("first_update_step100", "hybrid_first_update_then_freeze"),
        ("step20000", "hybrid_unanchored"),
        ("step20000", "hybrid_fixed_teacher_kl002"),
    ):
        h = lookup[(phase, method, "hungarian_assignment_distance")]
        auc = lookup[(phase, method, "coverage_radius_auc")]
        col = lookup[(phase, method, "collision_step_rate")]
        fmt = lambda row, digits: (
            f"{row['mean_delta']:+.{digits}f} "
            f"[{row['ci95_low']:+.{digits}f}, "
            f"{row['ci95_high']:+.{digits}f}]"
        )
        lines.append(
            f"| {phase} | {method} | {fmt(h, 4)} | {fmt(auc, 4)} | "
            f"{fmt(col, 5)} |"
        )
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps({
            "run_dirs": run_dirs,
            "replicates": list(REPLICATES),
            "statistical_unit": "independent distilled actor",
            "t_critical_df4": TCRIT_N5,
            "protocol_integrity": protocol_ok,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
