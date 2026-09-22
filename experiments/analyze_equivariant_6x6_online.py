"""Payload-level analysis of the locked 6x6 online experiment."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PAYLOAD_SEEDS = (1, 2, 3)
METHODS = (
    "equivariant_6x6_frozen",
    "equivariant_6x6_unanchored",
    "equivariant_6x6_actor_lr1e4",
    "equivariant_6x6_fixed_teacher_kl002",
)
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "final_coverage",
    "max_coverage",
    "minimum_agent_separation",
)
FROZEN = "equivariant_6x6_frozen"
TCRIT_N3 = 4.302652729696142
EXPECTED_CHECKPOINTS = [64, 100, 1000, 5000]


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
        "payloads_positive": int((values > 0).sum()),
        "payloads_negative": int((values < 0).sum()),
        "per_payload_deltas": ";".join(
            f"{value:.9g}" for value in values
        ),
    }


def only_run_dir(base):
    runs = sorted(base.glob("run_*"))
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one run under {base}, got {runs}")
    return runs[0]


def final_audit_kl(seed_dir):
    payload = json.loads(
        (seed_dir / "tensorboard_summary.json").read_text(encoding="utf-8")
    )
    values = [
        float(series[-1][2])
        for key, series in payload.items()
        if key.endswith("initial_reference_kl") and series
    ]
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="experiments/equivariant_6x6_online_analysis_20260716",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    rows_by_payload_method = {}
    run_dirs = {}
    audit_rows = []
    protocol_ok = True
    selected_actor_models = set()
    for payload_seed in PAYLOAD_SEEDS:
        base = (
            root
            / "experiments"
            / f"equivariant_6x6_online_payload{payload_seed}_20260716"
        )
        run_dir = only_run_dir(base)
        run_dirs[payload_seed] = run_dir
        metadata = json.loads(
            (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
        )
        protocol_ok &= metadata["methods"] == list(METHODS)
        protocol_ok &= int(metadata["total_env_steps"]) == 5_000
        actor_models = {
            spec["actor_model"]
            for spec in metadata["method_specs"].values()
        }
        protocol_ok &= len(actor_models) == 1
        selected_actor_models.update(actor_models)
        seed = int(metadata["seeds"][0])
        for method in METHODS:
            seed_dir = run_dir / method / f"seed_{seed}"
            rows_by_payload_method[(payload_seed, method)] = sorted(
                read_csv(seed_dir / "per_evaluation_episode_metrics.csv"),
                key=lambda row: int(row["test_seed"]),
            )
            counters = json.loads(
                (seed_dir / "training_counters.json").read_text(
                    encoding="utf-8"
                )
            )
            protocol_ok &= (
                int(counters["global_env_steps"]) == 5_000
                and counters["saved_checkpoints"] == EXPECTED_CHECKPOINTS
            )
            audit_rows.append({
                "payload_seed": payload_seed,
                "method": method,
                "training_seed": seed,
                "final_initial_reference_kl": final_audit_kl(seed_dir),
            })
    protocol_ok &= len(selected_actor_models) == 1

    payload_rows = []
    episode_rows = []
    for payload_seed in PAYLOAD_SEEDS:
        reference = rows_by_payload_method[(payload_seed, FROZEN)]
        reference_seeds = [row["test_seed"] for row in reference]
        for method in METHODS:
            candidate = rows_by_payload_method[(payload_seed, method)]
            if [row["test_seed"] for row in candidate] != reference_seeds:
                raise ValueError(
                    f"Unmatched rows for payload {payload_seed}, {method}"
                )
            summary = {
                "payload_seed": payload_seed,
                "method": method,
                "episodes": len(candidate),
            }
            for metric in METRICS:
                frozen_values = np.asarray(
                    [float(row[metric]) for row in reference],
                    dtype=np.float64,
                )
                values = np.asarray(
                    [float(row[metric]) for row in candidate],
                    dtype=np.float64,
                )
                summary[f"final_{metric}"] = float(values.mean())
                summary[f"delta_vs_frozen_{metric}"] = float(
                    (values - frozen_values).mean()
                )
            payload_rows.append(summary)
            episode_rows.extend(
                {**row, "payload_seed": payload_seed, "method": method}
                for row in candidate
            )

    comparisons = []
    lookup = {}
    for method in METHODS:
        selected = [row for row in payload_rows if row["method"] == method]
        for metric in METRICS:
            result = interval([
                row[f"delta_vs_frozen_{metric}"] for row in selected
            ])
            item = {
                "method": method,
                "comparison": "method_minus_frozen",
                "metric": metric,
                **result,
            }
            comparisons.append(item)
            lookup[(method, metric)] = item

    decisions = []
    for method in METHODS:
        h = lookup[(method, "hungarian_assignment_distance")]
        auc = lookup[(method, "coverage_radius_auc")]
        collision = lookup[(method, "collision_step_rate")]
        durable = (
            h["ci95_low"] <= 0.0
            and auc["ci95_high"] >= 0.0
            and collision["ci95_low"] <= 0.0
        )
        positive = (
            h["ci95_high"] < 0.0
            and auc["ci95_low"] > 0.0
            and collision["ci95_low"] <= 0.0
        )
        decisions.append({
            "method": method,
            "durable": durable,
            "positive_improvement": positive,
        })

    write_csv(output / "per_episode.csv", episode_rows)
    write_csv(output / "payload_summaries.csv", payload_rows)
    write_csv(output / "payload_intervals.csv", comparisons)
    write_csv(output / "policy_audit_kl.csv", audit_rows)
    write_csv(output / "decisions.csv", decisions)

    lines = [
        "# Equivariant 6x6 Online Result",
        "",
        "The independently trained distilled payload is the statistical unit. "
        "Intervals are paired 95% t intervals over three payloads.",
        "",
        f"Protocol integrity: **{'PASS' if protocol_ok else 'FAIL'}**.",
        "",
        "| method | durable | positive improvement | H delta [95% CI] | "
        "AUC delta [95% CI] | collision delta [95% CI] |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for decision in decisions:
        method = decision["method"]
        h = lookup[(method, "hungarian_assignment_distance")]
        auc = lookup[(method, "coverage_radius_auc")]
        collision = lookup[(method, "collision_step_rate")]
        lines.append(
            f"| {method} | {'yes' if decision['durable'] else 'no'} | "
            f"{'yes' if decision['positive_improvement'] else 'no'} | "
            f"{h['mean_delta']:+.4f} "
            f"[{h['ci95_low']:+.4f}, {h['ci95_high']:+.4f}] | "
            f"{auc['mean_delta']:+.4f} "
            f"[{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}] | "
            f"{collision['mean_delta']:+.5f} "
            f"[{collision['ci95_low']:+.5f}, "
            f"{collision['ci95_high']:+.5f}] |"
        )
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps({
            "run_dirs": {
                str(key): str(value) for key, value in run_dirs.items()
            },
            "payload_seeds": list(PAYLOAD_SEEDS),
            "methods": list(METHODS),
            "actor_models": sorted(selected_actor_models),
            "t_critical": TCRIT_N3,
            "protocol_integrity": protocol_ok,
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
