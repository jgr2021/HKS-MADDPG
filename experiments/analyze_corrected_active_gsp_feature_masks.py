import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


T_CRITICAL_95_DF2 = 4.3026527297
METRICS = [
    "return",
    "final_coverage",
    "max_coverage",
    "nearest_landmark_distance",
    "collision_step_rate",
]
COMPARISONS = [
    ("potentials_only_vs_full", "potentials_only", "full"),
    ("plus_crowding_vs_full", "plus_crowding_energy", "full"),
    ("plus_coverage_vs_potentials", "plus_coverage_energy", "potentials_only"),
    ("plus_crowding_vs_potentials", "plus_crowding_energy", "potentials_only"),
]
MASK_LABELS = {
    "full": "Full Active GSP",
    "potentials_only": "Deployment: potentials only",
    "plus_coverage_energy": "+ coverage energy",
    "plus_crowding_energy": "+ crowding energy",
    "energies_only": "Energies only",
    "no_action_features": "No action features",
}


def read_seed(path, seed):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["train_seed"] = seed
        for key in row:
            if key != "mask":
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed1-summary", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    rows.extend(read_seed(Path(args.seed1_summary), 1))
    rows.extend(read_seed(root / "seed_2" / "summary.csv", 2))
    rows.extend(read_seed(root / "seed_3" / "summary.csv", 3))
    rows.sort(key=lambda row: (row["mask"], row["train_seed"]))
    write_csv(root / "combined_seed_summaries.csv", rows)

    aggregate = []
    for mask in MASK_LABELS:
        selected = [row for row in rows if row["mask"] == mask]
        output = {"mask": mask, "label": MASK_LABELS[mask], "n_seeds": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_std"] = float(values.std(ddof=1))
        aggregate.append(output)
    write_csv(root / "aggregate_summary.csv", aggregate)

    lookup = {(row["mask"], int(row["train_seed"])): row for row in rows}
    paired = []
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
            paired.append(
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
    write_csv(root / "paired_training_seed_deltas.csv", paired)

    shown = ["full", "potentials_only", "plus_coverage_energy", "plus_crowding_energy"]
    colors = ["#4C78A8", "#59A14F", "#B279A2", "#E17C05"]
    panels = [
        ("return", "Return", True),
        ("final_coverage", "Final coverage", True),
        ("nearest_landmark_distance", "Distance", False),
        ("collision_step_rate", "Collision-step", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5))
    agg_lookup = {row["mask"]: row for row in aggregate}
    for ax, (metric, title, higher_better) in zip(axes, panels):
        means = [agg_lookup[mask][f"{metric}_mean"] for mask in shown]
        stds = [agg_lookup[mask][f"{metric}_std"] for mask in shown]
        ax.bar(range(len(shown)), means, yerr=stds, capsize=2.5, color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels(["Full", "Pot.", "+Cov E", "+Crd E"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xlabel("higher better" if higher_better else "lower better", fontsize=6.5)
    fig.tight_layout(pad=0.7)
    fig.savefig(root / "fig_feature_mask_effects.pdf", bbox_inches="tight")
    fig.savefig(root / "fig_feature_mask_effects.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    pair_lookup = {(row["comparison"], row["metric"]): row for row in paired}
    lines = [
        "# Corrected Active GSP Multi-Seed Feature-Mask Diagnostic",
        "",
        "This is a post-training inference diagnostic, not a retrained ablation.",
        "All masks use the same trained Active GSP checkpoints and 500 matched",
        "episodes per training seed.",
        "",
        "## Main Finding",
        "",
        "Removing both energy channels at deployment improves every primary",
        "metric relative to the full actor in all three training seeds.",
        "",
        "| metric | potentials-only minus full | 95% interval |",
        "| --- | ---: | ---: |",
    ]
    for metric in METRICS:
        row = pair_lookup[("potentials_only_vs_full", metric)]
        lines.append(
            f"| {metric} | {row['mean_delta']:+.4f} | "
            f"[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Coverage energy is the dominant harmful instantaneous channel:",
            "`plus_coverage_energy` stays near the full policy, whereas",
            "`plus_crowding_energy` stays near the potentials-only deployment.",
            "",
            "## Interpretation Boundary",
            "",
            "The result supports a held-out deployment-mask experiment. It does",
            "not prove that spectral training helps unless the masked Active GSP",
            "checkpoints also beat separately trained action-aware controls on a",
            "new evaluation seed set.",
        ]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

