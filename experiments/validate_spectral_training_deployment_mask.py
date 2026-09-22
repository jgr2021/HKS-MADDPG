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

from experiments.diagnose_learned_active_gsp_masks import evaluate as evaluate_masked
from experiments.run_passive_gsp_topology_pilot import evaluate_model


T_CRITICAL_95_DF2 = 4.3026527297
METHODS = [
    "raw_mlp",
    "action_control",
    "active_full",
    "active_potentials_deploy",
    "active_crowding_deploy",
]
LABELS = {
    "raw_mlp": "Raw MADDPG",
    "action_control": "Action-aware control",
    "active_full": "Full Active GSP",
    "active_potentials_deploy": "Spectral-train / potentials-deploy",
    "active_crowding_deploy": "Spectral-train / crowding-energy deploy",
}
MASKS = {
    "active_full": (1.0, 1.0, 1.0, 1.0),
    "active_potentials_deploy": (1.0, 1.0, 0.0, 0.0),
    "active_crowding_deploy": (1.0, 1.0, 0.0, 1.0),
}
METRICS = [
    "return",
    "final_coverage",
    "max_coverage",
    "final3",
    "max3",
    "nearest_landmark_distance",
    "collisions",
    "collision_step_rate",
]
COMPARISONS = [
    ("deploy_vs_control", "active_potentials_deploy", "action_control"),
    ("deploy_vs_full", "active_potentials_deploy", "active_full"),
    ("crowding_deploy_vs_control", "active_crowding_deploy", "action_control"),
    ("full_vs_control", "active_full", "action_control"),
    ("deploy_vs_raw", "active_potentials_deploy", "raw_mlp"),
]


def final_model(run_dir):
    matches = sorted((run_dir / "checkpoints").glob("model_final_*.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one final model in {run_dir}, got {matches}")
    return matches[0]


def source_map(args):
    seed1_raw = Path(args.seed1_raw_root)
    seed1_active = Path(args.seed1_active_root)
    seeds23 = Path(args.seeds23_root)
    sources = {
        1: {
            "raw_mlp": final_model(seed1_raw / "raw_mlp" / "seed_1"),
            "action_control": final_model(
                seed1_active / "learned_raw_potential_residual" / "seed_1"
            ),
            "active": final_model(
                seed1_active / "learned_active_gsp_residual" / "seed_1"
            ),
        }
    }
    for seed in [2, 3]:
        sources[seed] = {
            "raw_mlp": final_model(seeds23 / "raw_mlp" / f"seed_{seed}"),
            "action_control": final_model(
                seeds23 / "learned_raw_potential_residual" / f"seed_{seed}"
            ),
            "active": final_model(
                seeds23 / "learned_active_gsp_residual" / f"seed_{seed}"
            ),
        }
    return sources


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(method, model_path, episodes, episode_length, seed_base):
    if method in MASKS:
        return evaluate_masked(
            model_path, MASKS[method], episodes, episode_length, seed_base
        )
    return evaluate_model(
        "simple_spread", model_path, episodes, episode_length, seed_base
    )


def seed_summaries(rows):
    output = []
    for method in METHODS:
        for seed in [1, 2, 3]:
            selected = [
                row for row in rows
                if row["method"] == method and int(row["train_seed"]) == seed
            ]
            summary = {
                "method": method,
                "label": LABELS[method],
                "train_seed": seed,
                "episodes": len(selected),
            }
            for metric in METRICS:
                summary[metric] = float(np.mean([row[metric] for row in selected]))
            output.append(summary)
    return output


def aggregate(seed_rows):
    output = []
    for method in METHODS:
        selected = [row for row in seed_rows if row["method"] == method]
        summary = {"method": method, "label": LABELS[method], "n_seeds": 3}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(summary)
    return output


def paired(seed_rows):
    lookup = {
        (row["method"], int(row["train_seed"])): row for row in seed_rows
    }
    output = []
    for comparison, left, right in COMPARISONS:
        for metric in METRICS:
            deltas = np.asarray(
                [
                    lookup[(left, seed)][metric] - lookup[(right, seed)][metric]
                    for seed in [1, 2, 3]
                ],
                dtype=np.float64,
            )
            mean = float(deltas.mean())
            std = float(deltas.std(ddof=1))
            half = T_CRITICAL_95_DF2 * std / math.sqrt(3.0)
            output.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "seed1_delta": deltas[0],
                    "seed2_delta": deltas[1],
                    "seed3_delta": deltas[2],
                }
            )
    return output


def write_readme(output_dir, aggregate_rows, paired_rows, seed_base):
    agg = {row["method"]: row for row in aggregate_rows}
    pair = {(row["comparison"], row["metric"]): row for row in paired_rows}
    lines = [
        "# Held-Out Spectral-Train / Geometry-Deploy Validation",
        "",
        f"Evaluation seed base {seed_base:,} was not used to train any model.",
        "Every method uses 500 matched episodes for each of three training seeds.",
        "No model is retrained or modified on disk.",
        "",
        "## Aggregate",
        "",
        "| method | return | final cov. | max cov. | distance | collision-step |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = agg[method]
        lines.append(
            f"| {LABELS[method]} | {row['return_mean']:.3f} +/- "
            f"{row['return_std']:.3f} | {row['final_coverage_mean']:.3f} +/- "
            f"{row['final_coverage_std']:.3f} | {row['max_coverage_mean']:.3f} +/- "
            f"{row['max_coverage_std']:.3f} | "
            f"{row['nearest_landmark_distance_mean']:.3f} +/- "
            f"{row['nearest_landmark_distance_std']:.3f} | "
            f"{row['collision_step_rate_mean']:.3f} +/- "
            f"{row['collision_step_rate_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Deployment Mask Versus Separately Trained Control",
            "",
            "| metric | mean delta | 95% interval |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in [
        "return",
        "final_coverage",
        "max_coverage",
        "nearest_landmark_distance",
        "collision_step_rate",
    ]:
        row = pair[("deploy_vs_control", metric)]
        lines.append(
            f"| {metric} | {row['mean_delta']:+.4f} | "
            f"[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment tests a pre-specified deployment mask on held-out",
            "evaluation seeds. It can support a training-time spectral",
            "augmentation claim only if the masked Active checkpoints beat the",
            "separately trained action-aware controls consistently across",
            "training seeds.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed1-raw-root", required=True)
    parser.add_argument("--seed1-active-root", required=True)
    parser.add_argument("--seeds23-root", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=2000000)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item]
    methods = [item for item in args.methods.split(",") if item]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_map(args)

    all_rows = []
    for seed in seeds:
        for method in methods:
            model_key = "active" if method in MASKS else method
            print(f"Evaluating seed={seed} method={method}", flush=True)
            rows = evaluate_one(
                method,
                sources[seed][model_key],
                args.episodes,
                args.episode_length,
                args.seed_base,
            )
            for row in rows:
                row["method"] = method
                row["train_seed"] = seed
            all_rows.extend(rows)

    write_csv(output_dir / "per_episode.csv", all_rows)
    if seeds == [1, 2, 3] and methods == METHODS:
        seed_rows = seed_summaries(all_rows)
        aggregate_rows = aggregate(seed_rows)
        paired_rows = paired(seed_rows)
        write_csv(output_dir / "seed_summaries.csv", seed_rows)
        write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
        write_csv(output_dir / "paired_training_seed_deltas.csv", paired_rows)
        write_readme(output_dir, aggregate_rows, paired_rows, args.seed_base)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "episodes": args.episodes,
                "episode_length": args.episode_length,
                "seed_base": args.seed_base,
                "seeds": seeds,
                "methods": methods,
                "masks": MASKS,
                "sources": {
                    str(seed): {key: str(path) for key, path in values.items()}
                    for seed, values in sources.items()
                },
                "interpretation": (
                    "held-out evaluation of post-training deployment masks; "
                    "not retraining"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
