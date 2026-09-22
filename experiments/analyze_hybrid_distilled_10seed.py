"""Combine the locked 3+7 seed runs and compare to the frozen actor."""

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


FIRST = Path("experiments/hybrid_distilled_anchored_100k/run_20260713_203744")
EXTENSION = Path("experiments/hybrid_distilled_anchored_100k_extension/run_20260713_211744")
INITIAL_CHECKPOINT = Path(
    "experiments/hybrid_distilled_anchored_20k/run_20260713_195847/"
    "hybrid_distilled_kl002/seed_41/checkpoints/model_step64.pt"
)
OUTPUT = Path("experiments/hybrid_distilled_anchored_10seed_analysis_20260713")
METRICS = (
    "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
    "return", "final_coverage", "max_coverage", "minimum_agent_separation",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def root_and_eval_base(seed):
    return (FIRST, 4_100_000) if seed <= 43 else (EXTENSION, 4_400_000)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    all_initial, all_final, seed_summary, seed_deltas = [], [], [], []
    counters_ok = True
    for seed in range(41, 51):
        root, base = root_and_eval_base(seed)
        seed_dir = root / "hybrid_distilled_kl002" / f"seed_{seed}"
        final_rows = read_csv(seed_dir / "per_evaluation_episode_metrics.csv")
        final_rows = [{**row, "seed": seed, "phase": "anchored_100k"} for row in final_rows]
        initial_rows = evaluate_model(
            "simple_spread", INITIAL_CHECKPOINT, 500, 25, base + seed * 10000
        )
        initial_rows = [{**row, "seed": seed, "phase": "frozen_initial"} for row in initial_rows]
        all_initial.extend(initial_rows); all_final.extend(final_rows)
        counters = json.loads((seed_dir / "training_counters.json").read_text())
        counters_ok &= (
            counters["global_env_steps"] == 100000
            and counters["saved_checkpoints"] == [20000, 40000, 60000, 80000, 100000]
        )
        result = {"seed": seed, "eval_episodes": 500}
        final_sorted = sorted(final_rows, key=lambda row: int(row["test_seed"]))
        initial_sorted = sorted(initial_rows, key=lambda row: int(row["test_seed"]))
        for metric in METRICS:
            before = np.asarray([float(row[metric]) for row in initial_sorted])
            after = np.asarray([float(row[metric]) for row in final_sorted])
            delta = after - before
            result[f"initial_{metric}"] = float(before.mean())
            result[f"final_{metric}"] = float(after.mean())
            result[f"delta_{metric}"] = float(delta.mean())
            half = 1.9647293909876649 * delta.std(ddof=1) / math.sqrt(len(delta))
            seed_deltas.append({
                "seed": seed, "metric": metric, "mean_delta": float(delta.mean()),
                "episode_ci95_low": float(delta.mean() - half),
                "episode_ci95_high": float(delta.mean() + half),
            })
        seed_summary.append(result)
    write_csv(OUTPUT / "frozen_initial_per_episode.csv", all_initial)
    write_csv(OUTPUT / "anchored_100k_per_episode.csv", all_final)
    write_csv(OUTPUT / "seed_summary.csv", seed_summary)
    write_csv(OUTPUT / "paired_seed_deltas.csv", seed_deltas)

    intervals = []
    tcrit = 2.2621571627409915
    for metric in METRICS:
        values = np.asarray([row[f"delta_{metric}"] for row in seed_summary])
        mean = float(values.mean())
        half = tcrit * values.std(ddof=1) / math.sqrt(len(values))
        intervals.append({
            "metric": metric, "n_training_seeds": 10, "mean_delta": mean,
            "ci95_low": float(mean - half), "ci95_high": float(mean + half),
            "seeds_positive": int((values > 0).sum()),
            "seeds_negative": int((values < 0).sum()),
            "per_seed_deltas": ";".join(f"{value:.9g}" for value in values),
        })
    write_csv(OUTPUT / "training_seed_intervals.csv", intervals)
    lookup = {row["metric"]: row for row in intervals}
    final_means = {
        metric: np.mean([row[f"final_{metric}"] for row in seed_summary])
        for metric in METRICS
    }
    final_sds = {
        metric: np.std([row[f"final_{metric}"] for row in seed_summary], ddof=1)
        for metric in METRICS
    }
    passes = (
        lookup["hungarian_assignment_distance"]["ci95_high"] < 0
        and lookup["coverage_radius_auc"]["ci95_low"] > 0
        and lookup["collision_step_rate"]["ci95_low"] <= 0
    )
    lines = [
        "# Hybrid-Distilled Anchored Actor: Ten-Seed 100k Result", "",
        "The training seed is the statistical unit. Every seed used 500 matched deterministic evaluation initializations before/after training.", "",
        f"Protocol integrity: **{'PASS' if counters_ok else 'FAIL'}**. Promotion: **{'PASS' if passes else 'FAIL'}**.", "",
        "| metric | final mean +/- seed SD | paired delta | training-seed 95% CI | direction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        row = lookup[metric]
        lines.append(
            f"| {metric} | {final_means[metric]:.5f} +/- {final_sds[metric]:.5f} | "
            f"{row['mean_delta']:+.5f} | [{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] | "
            f"{row['seeds_negative']} neg / {row['seeds_positive']} pos |"
        )
    lines += ["", "The method and KL radius were unchanged across all ten seeds. Minimum separation remains a secondary risk even when collision rate has no resolved penalty."]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "roots": [str(FIRST), str(EXTENSION)], "initial_checkpoint": str(INITIAL_CHECKPOINT),
        "training_seeds": list(range(41, 51)), "eval_episodes_per_seed": 500,
        "counters_and_checkpoints_verified": counters_ok, "promotion_pass": passes,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
