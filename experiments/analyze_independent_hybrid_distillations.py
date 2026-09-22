"""Apply the locked offline gates to independent hybrid distillations."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


REPLICATES = (
    "fold21_t101_s101",
    "fold22_t102_s102",
    "fold23_t103_s103",
)
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "minimum_agent_separation",
    "final_coverage",
    "max_coverage",
)
STUDENT = "hybrid_distilled_actor"
TEACHER = "hybrid_teacher"
HUNGARIAN_LIMIT = 0.30
AUC_LIMIT = 0.70
COLLISION_LIMIT = 0.015
MAX_PAIRED_COVERAGE_DEGRADATION = 0.05


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(reference, candidate):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    critical = 1.9647293909876649
    half = critical * delta.std(ddof=1) / math.sqrt(len(delta))
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(delta.mean() - half),
        "ci95_high": float(delta.mean() + half),
        "fraction_positive": float((delta > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="experiments/independent_hybrid_distillation_analysis_20260716",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    summary_rows = []
    interval_rows = []
    decisions = []
    for replicate in REPLICATES:
        source = (
            root
            / "experiments"
            / f"independent_hybrid_distillation_{replicate}_20260716"
            / "per_episode.csv"
        )
        rows = read_csv(source)
        by_controller = {
            controller: sorted(
                [row for row in rows if row["controller"] == controller],
                key=lambda row: int(row["test_seed"]),
            )
            for controller in (TEACHER, STUDENT)
        }
        teacher = by_controller[TEACHER]
        student = by_controller[STUDENT]
        if [row["test_seed"] for row in teacher] != [
            row["test_seed"] for row in student
        ]:
            raise ValueError(f"Unmatched evaluation seeds for {replicate}")

        controller_means = {}
        for controller, selected in by_controller.items():
            summary = {
                "replicate": replicate,
                "controller": controller,
                "episodes": len(selected),
            }
            controller_means[controller] = {}
            for metric in METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in selected], dtype=np.float64
                )
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1))
                controller_means[controller][metric] = float(values.mean())
            summary_rows.append(summary)

        interval_lookup = {}
        for metric in METRICS:
            result = paired_interval(
                [float(row[metric]) for row in teacher],
                [float(row[metric]) for row in student],
            )
            interval_rows.append({
                "replicate": replicate,
                "comparison": "student_minus_teacher",
                "metric": metric,
                **result,
            })
            interval_lookup[metric] = result

        student_means = controller_means[STUDENT]
        h_absolute = (
            student_means["hungarian_assignment_distance"] <= HUNGARIAN_LIMIT
        )
        auc_absolute = student_means["coverage_radius_auc"] >= AUC_LIMIT
        collision_absolute = (
            student_means["collision_step_rate"] <= COLLISION_LIMIT
        )
        no_resolved_collision_penalty = (
            interval_lookup["collision_step_rate"]["ci95_low"] <= 0.0
        )
        no_large_hungarian_degradation = (
            interval_lookup["hungarian_assignment_distance"]["ci95_low"]
            <= MAX_PAIRED_COVERAGE_DEGRADATION
        )
        no_large_auc_degradation = (
            interval_lookup["coverage_radius_auc"]["ci95_high"]
            >= -MAX_PAIRED_COVERAGE_DEGRADATION
        )
        eligible = all((
            h_absolute,
            auc_absolute,
            collision_absolute,
            no_resolved_collision_penalty,
            no_large_hungarian_degradation,
            no_large_auc_degradation,
        ))
        decisions.append({
            "replicate": replicate,
            "eligible": eligible,
            "student_hungarian_at_most_030": h_absolute,
            "student_auc_at_least_070": auc_absolute,
            "student_collision_at_most_0015": collision_absolute,
            "no_resolved_collision_penalty": no_resolved_collision_penalty,
            "no_resolved_hungarian_degradation_over_005": (
                no_large_hungarian_degradation
            ),
            "no_resolved_auc_degradation_over_005": no_large_auc_degradation,
        })

    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "paired_intervals.csv", interval_rows)
    write_csv(output / "decisions.csv", decisions)

    lines = [
        "# Independent Hybrid Distillation Analysis",
        "",
        "Intervals are paired 95% normal intervals over 500 matched episodes.",
        "",
        "| replicate | eligible | H <= .30 | AUC >= .70 | collision <= .015 | collision no-penalty | H loss <= .05 | AUC loss <= .05 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in decisions:
        yes = lambda key: "yes" if row[key] else "no"
        lines.append(
            f"| {row['replicate']} | {'PASS' if row['eligible'] else 'FAIL'} | "
            f"{yes('student_hungarian_at_most_030')} | "
            f"{yes('student_auc_at_least_070')} | "
            f"{yes('student_collision_at_most_0015')} | "
            f"{yes('no_resolved_collision_penalty')} | "
            f"{yes('no_resolved_hungarian_degradation_over_005')} | "
            f"{yes('no_resolved_auc_degradation_over_005')} |"
        )
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
