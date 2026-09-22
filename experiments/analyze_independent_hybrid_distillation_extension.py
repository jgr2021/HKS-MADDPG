"""Offline fidelity and safety audit across five independent distillations."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


REPLICATES = {
    "fold21_t101_s101": "20260716",
    "fold22_t102_s102": "20260716",
    "fold23_t103_s103": "20260716",
    "fold24_t104_s104": "20260811",
    "fold25_t105_s105": "20260811",
}
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "minimum_agent_separation",
    "return",
)
STUDENT = "hybrid_distilled_actor"
TEACHER = "hybrid_teacher"
TCRIT_EPISODE = 1.9647293909876649
TCRIT_ACTOR_N5 = 2.7764451051977987


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, critical):
    values = np.asarray(values, dtype=np.float64)
    half = critical * values.std(ddof=1) / math.sqrt(len(values))
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(values.mean() - half),
        "ci95_high": float(values.mean() + half),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/independent_hybrid_distillation_"
            "extension_analysis_20260811"
        ),
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    summaries = []
    episode_intervals = []
    actor_deltas = {metric: [] for metric in METRICS}
    decisions = []
    for replicate, artifact_date in REPLICATES.items():
        source = (
            Path("experiments")
            / f"independent_hybrid_distillation_{replicate}_{artifact_date}"
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
            raise ValueError(f"Unmatched offline rows for {replicate}")

        student_means = {}
        intervals = {}
        for controller, selected in by_controller.items():
            summary = {
                "replicate": replicate,
                "controller": controller,
                "episodes": len(selected),
            }
            for metric in METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in selected], dtype=np.float64
                )
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1))
                if controller == STUDENT:
                    student_means[metric] = float(values.mean())
            summaries.append(summary)

        for metric in METRICS:
            delta = np.asarray(
                [float(row[metric]) for row in student], dtype=np.float64
            ) - np.asarray(
                [float(row[metric]) for row in teacher], dtype=np.float64
            )
            result = interval(delta, TCRIT_EPISODE)
            intervals[metric] = result
            actor_deltas[metric].append(result["mean_delta"])
            episode_intervals.append({
                "replicate": replicate,
                "comparison": "student_minus_teacher",
                "metric": metric,
                **result,
            })

        decisions.append({
            "replicate": replicate,
            "eligible": all((
                student_means["hungarian_assignment_distance"] <= 0.30,
                student_means["coverage_radius_auc"] >= 0.70,
                student_means["collision_step_rate"] <= 0.015,
                intervals["collision_step_rate"]["ci95_low"] <= 0.0,
                intervals["hungarian_assignment_distance"]["ci95_low"]
                <= 0.05,
                intervals["coverage_radius_auc"]["ci95_high"] >= -0.05,
            )),
            "student_hungarian": student_means[
                "hungarian_assignment_distance"
            ],
            "student_auc": student_means["coverage_radius_auc"],
            "student_collision": student_means["collision_step_rate"],
        })

    actor_intervals = [
        {
            "comparison": "student_minus_teacher",
            "metric": metric,
            "actors": len(REPLICATES),
            **interval(values, TCRIT_ACTOR_N5),
            "per_actor_deltas": ";".join(f"{v:.9g}" for v in values),
        }
        for metric, values in actor_deltas.items()
    ]
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "episode_paired_intervals.csv", episode_intervals)
    write_csv(output / "actor_paired_intervals.csv", actor_intervals)
    write_csv(output / "decisions.csv", decisions)

    actor_lookup = {row["metric"]: row for row in actor_intervals}
    lines = [
        "# Five-Actor Offline Distillation Audit",
        "",
        "Actor-level intervals treat each independently distilled actor as "
        "the statistical unit; episode-level gates remain diagnostics.",
        "",
        "| replicate | gate | student H | student AUC | student collision |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in decisions:
        lines.append(
            f"| {row['replicate']} | "
            f"{'PASS' if row['eligible'] else 'FAIL'} | "
            f"{row['student_hungarian']:.4f} | {row['student_auc']:.4f} | "
            f"{row['student_collision']:.5f} |"
        )
    lines += [
        "",
        "| metric | student - teacher [actor-level 95% CI] |",
        "| --- | ---: |",
    ]
    for metric in METRICS:
        row = actor_lookup[metric]
        lines.append(
            f"| {metric} | {row['mean_delta']:+.5f} "
            f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] |"
        )
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
