"""Matched-initialization analysis for the exact 6x6 first-update stress test."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_equivariant_6x6_online_stress import evaluate_model


METRICS = [
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "final_coverage",
    "max_coverage",
    "minimum_agent_separation",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, rng, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    sampled = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    return [float(item) for item in np.percentile(sampled, [2.5, 97.5])]


def paired_summary(before, after, rng):
    before_by_seed = {int(row["test_seed"]): row for row in before}
    after_by_seed = {int(row["test_seed"]): row for row in after}
    seeds = sorted(set(before_by_seed) & set(after_by_seed))
    if len(seeds) != len(before) or len(seeds) != len(after):
        raise RuntimeError("Matched evaluation rows do not have identical test seeds")
    output = {"n_episodes": len(seeds), "metrics": {}}
    for metric in METRICS:
        lhs = np.asarray([float(before_by_seed[seed][metric]) for seed in seeds])
        rhs = np.asarray([float(after_by_seed[seed][metric]) for seed in seeds])
        delta = rhs - lhs
        output["metrics"][metric] = {
            "before_mean": float(lhs.mean()),
            "after_mean": float(rhs.mean()),
            "delta_mean": float(delta.mean()),
            "delta_ci95": bootstrap_ci(delta, rng),
            "improved_episode_fraction_raw_delta_negative": float((delta < 0).mean()),
        }
    return output


def anchor_audit(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = {"old_kl": [], "proposed_kl": [], "accepted_kl": [], "accepted_scale": []}
    for key, records in payload.items():
        for suffix in groups:
            if key.endswith("anchor_" + suffix):
                groups[suffix].extend(float(record[2]) for record in records)
    output = {}
    for key, values in groups.items():
        array = np.asarray(values, dtype=np.float64)
        output[key] = {
            "n": int(array.size), "mean": float(array.mean()),
            "min": float(array.min()), "max": float(array.max()),
        }
    accepted = np.asarray(groups["accepted_kl"], dtype=np.float64)
    output["accepted_kl"]["fraction_above_0_002"] = float((accepted > 0.002 + 1e-9).mean())
    return output


def table_rows(label, summary):
    rows = []
    for metric, item in summary["metrics"].items():
        rows.append(
            f"| {label} | {metric} | {item['before_mean']:.6f} | "
            f"{item['after_mean']:.6f} | {item['delta_mean']:+.6f} | "
            f"[{item['delta_ci95'][0]:+.6f}, {item['delta_ci95'][1]:+.6f}] |"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=730071)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    runner = json.loads((run_dir / "runner_config.json").read_text(encoding="utf-8"))
    seed = int(str(runner["seeds"]).split(",")[0])
    episodes = int(runner["eval_episodes"])
    horizon = int(runner["episode_length"])
    eval_seed = int(runner["eval_seed_base"]) + seed * 10000
    unanchored = run_dir / "equivariant_6x6_unanchored" / f"seed_{seed}"
    anchored = run_dir / "equivariant_6x6_kl002" / f"seed_{seed}"
    matched_pre_path = run_dir / "matched_pre_update_episode_metrics.csv"
    if matched_pre_path.exists():
        pre_rows = read_csv(matched_pre_path)
    else:
        checkpoint = unanchored / "checkpoints" / "model_step64.pt"
        pre_rows = evaluate_model(
            "simple_spread_6x6", checkpoint, episodes=episodes,
            episode_length=horizon, seed=eval_seed,
        )
        write_csv(matched_pre_path, pre_rows)
    unanchored_rows = read_csv(unanchored / "per_evaluation_episode_metrics.csv")
    anchored_rows = read_csv(anchored / "per_evaluation_episode_metrics.csv")
    rng = np.random.default_rng(args.bootstrap_seed)
    results = {
        "protocol": {
            "training_seed": seed, "episodes": episodes, "horizon": horizon,
            "matched_eval_seed_base": eval_seed,
            "pre_update_checkpoint": "model_step64",
            "post_update_checkpoint": "model_step100",
        },
        "unanchored_post_minus_pre": paired_summary(pre_rows, unanchored_rows, rng),
        "anchored_post_minus_pre": paired_summary(pre_rows, anchored_rows, rng),
        "anchored_minus_unanchored_post": paired_summary(unanchored_rows, anchored_rows, rng),
        "anchor_audit": anchor_audit(anchored / "tensorboard_summary.json"),
    }
    (run_dir / "matched_first_update_analysis.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Exact 6x6 first-update stress: matched analysis", "",
        "All rows use the same 100 deterministic evaluation initializations. Deltas are",
        "`after - before`; for Hungarian distance and collision rate, negative is better,",
        "while for AUC, return, coverage, and minimum separation, positive is better.", "",
        "| comparison | metric | before | after | delta | paired bootstrap 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines += table_rows("unanchored post - pre", results["unanchored_post_minus_pre"])
    lines += table_rows("KL=.002 post - pre", results["anchored_post_minus_pre"])
    lines += table_rows("KL=.002 - unanchored post", results["anchored_minus_unanchored_post"])
    audit = results["anchor_audit"]
    lines += [
        "", "## Trust-region audit", "",
        f"The 24 raw actor proposals had mean KL {audit['proposed_kl']['mean']:.6f} "
        f"(max {audit['proposed_kl']['max']:.6f}). Accepted scale averaged "
        f"{audit['accepted_scale']['mean']:.6f}. Accepted mini-batch KL averaged "
        f"{audit['accepted_kl']['mean']:.6f}; "
        f"{audit['accepted_kl']['fraction_above_0_002']:.1%} of logged mini-batches were "
        "above 0.002 because the fixed-teacher projection is enforced on the current "
        "mini-batch, not globally over the replay-state distribution.", "",
        "## Decision", "",
        "One unanchored MADDPG update event catastrophically destroys the distilled actor. "
        "KL=0.002 prevents that collapse, but this one-seed stress is only a safety gate: "
        "it is not evidence that online learning improves the teacher. Do not start a "
        "100k screening run until the exact actor is accelerated and the anchored first-update "
        "result is repeated across fresh training/replay seeds.", "",
    ]
    (run_dir / "MATCHED_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(run_dir / "MATCHED_ANALYSIS.md")


if __name__ == "__main__":
    main()
