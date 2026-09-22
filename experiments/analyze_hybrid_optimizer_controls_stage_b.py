"""Training-seed analysis for the fresh Stage-B optimizer confirmation."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "final_coverage",
    "max_coverage",
    "minimum_agent_separation",
)
FROZEN = "hybrid_frozen"
FIXED_TEACHER = "hybrid_fixed_teacher_kl002"
TCRIT_N3 = 4.302652729696142


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def final_audit_kl(seed_dir):
    payload = json.loads(
        (seed_dir / "tensorboard_summary.json").read_text(encoding="utf-8")
    )
    values = [
        float(series[-1][2])
        for key, series in payload.items()
        if key.endswith("initial_reference_kl") and series
    ]
    return float(np.mean(values)) if values else float("nan")


def training_seed_interval(values):
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
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = run_dir / "stage_b_analysis"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    methods = metadata["methods"]
    seeds = [int(seed) for seed in metadata["seeds"]]
    if len(seeds) != 3:
        raise ValueError(f"Expected three Stage-B seeds, received {seeds}")

    per_episode = []
    seed_summaries = []
    episode_intervals = []
    audit_rows = []
    rows_by_method_seed = {}
    counters_ok = True
    for method in methods:
        for seed in seeds:
            seed_dir = run_dir / method / f"seed_{seed}"
            rows = sorted(
                read_csv(seed_dir / "per_evaluation_episode_metrics.csv"),
                key=lambda row: int(row["test_seed"]),
            )
            rows_by_method_seed[(method, seed)] = rows
            per_episode.extend(
                {**row, "method": method, "training_seed": seed}
                for row in rows
            )
            audit_rows.append({
                "method": method,
                "training_seed": seed,
                "final_initial_reference_kl": final_audit_kl(seed_dir),
            })
            counters = json.loads(
                (seed_dir / "training_counters.json").read_text(encoding="utf-8")
            )
            counters_ok &= (
                int(counters["global_env_steps"]) == 20_000
                and counters["saved_checkpoints"]
                == [64, 100, 1000, 4996, 5000, 10000, 20000]
            )

    for method in methods:
        for seed in seeds:
            frozen = rows_by_method_seed[(FROZEN, seed)]
            candidate = rows_by_method_seed[(method, seed)]
            if [row["test_seed"] for row in frozen] != [
                row["test_seed"] for row in candidate
            ]:
                raise ValueError(f"Unmatched episodes for {method}, seed {seed}")
            summary = {
                "method": method,
                "training_seed": seed,
                "episodes": len(candidate),
            }
            for metric in METRICS:
                reference = np.asarray(
                    [float(row[metric]) for row in frozen], dtype=np.float64
                )
                values = np.asarray(
                    [float(row[metric]) for row in candidate], dtype=np.float64
                )
                delta = values - reference
                summary[f"final_{metric}"] = float(values.mean())
                summary[f"delta_vs_frozen_{metric}"] = float(delta.mean())
                half = (
                    1.9647293909876649
                    * delta.std(ddof=1)
                    / math.sqrt(len(delta))
                )
                episode_intervals.append({
                    "method": method,
                    "training_seed": seed,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "ci95_low": float(delta.mean() - half),
                    "ci95_high": float(delta.mean() + half),
                })
            seed_summaries.append(summary)

    training_intervals = []
    lookup = {}
    for method in methods:
        selected = [row for row in seed_summaries if row["method"] == method]
        for metric in METRICS:
            result = training_seed_interval(
                [row[f"delta_vs_frozen_{metric}"] for row in selected]
            )
            row = {
                "method": method,
                "comparison": "method_minus_frozen",
                "metric": metric,
                "n_training_seeds": len(selected),
                **result,
            }
            training_intervals.append(row)
            lookup[(method, metric)] = row

    decisions = []
    for method in methods:
        h = lookup[(method, "hungarian_assignment_distance")]
        auc = lookup[(method, "coverage_radius_auc")]
        collision = lookup[(method, "collision_step_rate")]
        no_h_degradation = h["ci95_low"] <= 0.0
        no_auc_degradation = auc["ci95_high"] >= 0.0
        no_collision_penalty = collision["ci95_low"] <= 0.0
        durable = (
            no_h_degradation
            and no_auc_degradation
            and no_collision_penalty
        )
        positive_improvement = (
            h["ci95_high"] < 0.0
            and auc["ci95_low"] > 0.0
            and no_collision_penalty
        )
        decisions.append({
            "method": method,
            "durable": durable,
            "positive_improvement": positive_improvement,
            "no_resolved_hungarian_degradation": no_h_degradation,
            "no_resolved_auc_degradation": no_auc_degradation,
            "no_resolved_collision_penalty": no_collision_penalty,
        })

    write_csv(output / "per_episode.csv", per_episode)
    write_csv(output / "seed_summaries.csv", seed_summaries)
    write_csv(output / "episode_paired_intervals.csv", episode_intervals)
    write_csv(output / "training_seed_intervals.csv", training_intervals)
    write_csv(output / "policy_audit_kl.csv", audit_rows)
    write_csv(output / "decisions.csv", decisions)

    final_means = {}
    final_sds = {}
    for method in methods:
        selected = [row for row in seed_summaries if row["method"] == method]
        for metric in METRICS:
            values = np.asarray(
                [row[f"final_{metric}"] for row in selected],
                dtype=np.float64,
            )
            final_means[(method, metric)] = float(values.mean())
            final_sds[(method, metric)] = float(values.std(ddof=1))

    lines = [
        "# Hybrid Optimizer Controls: Stage-B Result",
        "",
        "The training seed is the statistical unit; intervals are paired 95% "
        "t intervals over three fresh seeds.",
        "",
        f"Protocol integrity: **{'PASS' if counters_ok else 'FAIL'}**.",
        "",
        "| method | durable | positive improvement | Hungarian final | H delta [95% CI] | AUC final | AUC delta [95% CI] | collision delta [95% CI] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    decision_lookup = {row["method"]: row for row in decisions}
    for method in methods:
        h = lookup[(method, "hungarian_assignment_distance")]
        auc = lookup[(method, "coverage_radius_auc")]
        collision = lookup[(method, "collision_step_rate")]
        decision = decision_lookup[method]
        lines.append(
            f"| {method} | {'yes' if decision['durable'] else 'no'} | "
            f"{'yes' if decision['positive_improvement'] else 'no'} | "
            f"{final_means[(method, 'hungarian_assignment_distance')]:.4f} "
            f"+/- {final_sds[(method, 'hungarian_assignment_distance')]:.4f} | "
            f"{h['mean_delta']:+.4f} [{h['ci95_low']:+.4f}, {h['ci95_high']:+.4f}] | "
            f"{final_means[(method, 'coverage_radius_auc')]:.4f} "
            f"+/- {final_sds[(method, 'coverage_radius_auc')]:.4f} | "
            f"{auc['mean_delta']:+.4f} [{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}] | "
            f"{collision['mean_delta']:+.5f} "
            f"[{collision['ci95_low']:+.5f}, {collision['ci95_high']:+.5f}] |"
        )
    lines += [
        "",
        "Primary-mechanism selection must compare every durable Stage-A "
        "survivor. The fixed-teacher method is not promoted merely because "
        "it is durable.",
    ]
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps({
            "run_dir": str(run_dir),
            "methods": methods,
            "training_seeds": seeds,
            "training_seed_t_critical": TCRIT_N3,
            "counters_and_checkpoints_verified": counters_ok,
            "fixed_teacher_present": FIXED_TEACHER in methods,
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
