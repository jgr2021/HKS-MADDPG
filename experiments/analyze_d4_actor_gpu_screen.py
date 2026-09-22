"""Verify the completed D4 seed-1 pair and report all predeclared outcomes."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.run_d4_actor_gpu_screen import METHODS, sha256, source_hashes, training_args, validate_run
from run_vector_signal_gsp_experiment import METRICS
from utils.d4_graph_residual import transform_local_geometry


LABELS = {
    "d4_geometric_control": "D4 geometry",
    "d4_dirichlet_gsp": "D4 GSP",
    "archive_raw_mlp": "Archived raw",
    "archive_learned_raw_potential_residual": "Archived geometry",
    "archive_learned_active_gsp_residual": "Archived full GSP",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_episodes(rows, expected_episodes=500, seed_base=1000000):
    if len(rows) != expected_episodes:
        raise ValueError("Incorrect final evaluation episode count")
    if [int(row["test_seed"]) for row in rows] != list(range(seed_base, seed_base + expected_episodes)):
        raise ValueError("Evaluation states are missing, reordered, duplicated or mismatched")
    if [int(row["eval_episode"]) for row in rows] != list(range(expected_episodes)):
        raise ValueError("Evaluation episode indices are not contiguous")
    for metric in METRICS:
        if not np.isfinite([float(row[metric]) for row in rows]).all():
            raise ValueError(f"Non-finite evaluation metric: {metric}")


def trained_policy_checks(checkpoint):
    model = MADDPG.init_from_save(checkpoint)
    model.prep_rollouts(device="cpu")
    generator = torch.Generator().manual_seed(81373)
    observations = torch.randn(64, 18, generator=generator)
    rows = []
    with torch.no_grad():
        for index, agent in enumerate(model.agents):
            policy = agent.policy
            residual = policy.graph_residual(observations)
            orbit = transform_local_geometry(observations, policy.d4_matrices)
            errors = []
            for group in range(8):
                transformed = policy.graph_residual(orbit[:, group])[:, policy.d4_actions[group]]
                # Summation order changes in group averaging at float32 precision.
                torch.testing.assert_close(transformed, residual, rtol=2e-4, atol=2e-4)
                errors.append(float((transformed - residual).abs().max()))
            translated = observations.clone()
            translated[:, 2:4] += 100
            torch.testing.assert_close(policy.graph_residual(translated), residual, rtol=0, atol=0)
            altered = observations.clone()
            altered[1:] += 100
            torch.testing.assert_close(policy(altered)[:1], policy(observations)[:1], rtol=0, atol=0)
            raw_logits = policy.mlp(observations)
            rows.append({"agent": index, "max_residual_equivariance_error": max(errors),
                         "raw_branch_logit_rms": float(raw_logits.square().mean().sqrt()),
                         "residual_logit_rms": float(residual.square().mean().sqrt()),
                         "parameters": sum(parameter.numel() for parameter in policy.parameters()),
                         "critic_dim": agent.critic.fc1.in_features,
                         "target_critic_dim": agent.target_critic.fc1.in_features})
    return rows


def analyze(root, output):
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "completed" or status["phase"] != "formal":
        raise RuntimeError("Wait for the original formal driver to complete; do not restart it")
    if status["completed_methods"] != list(METHODS):
        raise RuntimeError("Incomplete matched method pair")
    if status["source_hashes"] != source_hashes():
        raise RuntimeError("Current source differs from the completed run")
    args = training_args("formal")
    if status["training"] != vars(args):
        raise RuntimeError("The run used a different training protocol")
    torch.set_num_threads(1)
    summaries = read_csv(root / "seed_summaries.csv")
    lookup = {row["method"]: row for row in summaries}
    if set(lookup) != set(LABELS) or len(summaries) != len(LABELS):
        raise RuntimeError("Missing or duplicate final result rows")
    evidence, episodes, curves, table = {}, {}, {}, []
    for method in LABELS:
        summary = lookup[method]
        if int(summary["seed"]) != 1 or int(summary["eval_episodes"]) != 500:
            raise RuntimeError("Only seed 1 with 500 final episodes is allowed")
        if method in METHODS:
            directory = root / method / "seed_1"
            evidence[method] = validate_run(directory, args)
            config = json.loads((directory / "resolved_config.json").read_text())
            assert config["actor_anchor"] is None and config["pretrained_policy_path"] is None
            assert config["actor_lr"] == config["critic_lr"] == config["lr"] == 0.01
            assert config["actor_update_start_step"] == 0 and config["actor_update_end_step"] > 100000
            assert config["env_id"] == "simple_spread" and config["discrete_action"]
            counters = json.loads((directory / "training_counters.json").read_text())
            assert counters["episode_batches"] == 1000 and counters["env_episodes_started"] == 4000
            evidence[method]["checkpoint_hashes"] = {}
            for step in args.checkpoint_steps:
                checkpoint = directory / "checkpoints" / f"model_step{step}.pt"
                model = MADDPG.init_from_save(checkpoint)
                for agent in model.agents:
                    for name in ("policy", "critic", "target_policy", "target_critic"):
                        assert all(torch.isfinite(value).all() for value in getattr(agent, name).state_dict().values())
                evidence[method]["checkpoint_hashes"][str(step)] = sha256(checkpoint)
            final = directory / "checkpoints/model_final_100000.pt"
            evidence[method]["trained_actor_checks"] = trained_policy_checks(final)
            curves[method] = read_csv(directory / "learning_curve_eval.csv")
            assert [row["checkpoint"] for row in curves[method]] == [f"model_step{step}" for step in args.checkpoint_steps]
            assert all(int(row["eval_episodes"]) == 100 for row in curves[method])
            episode_path = directory / "per_evaluation_episode_metrics.csv"
        else:
            episode_path = root / f"{method}_evaluation.csv"
            assert sha256(summary["checkpoint"]) == summary["checkpoint_sha256"]
        episodes[method] = read_csv(episode_path)
        validate_episodes(episodes[method])
        row = {"method": method, "n_training_seeds": 1, "evaluation_episodes": 500}
        for metric in METRICS:
            value = float(np.mean([float(item[metric]) for item in episodes[method]]))
            np.testing.assert_allclose(value, float(summary[metric]), rtol=1e-10, atol=1e-10)
            row[metric] = value
        table.append(row)
    paired = []
    target = "d4_dirichlet_gsp"
    for baseline in LABELS:
        if baseline == target:
            continue
        for metric in METRICS:
            differences = np.asarray([float(left[metric]) - float(right[metric])
                                      for left, right in zip(episodes[target], episodes[baseline])])
            paired.append({"baseline": baseline, "metric": metric, "mean_delta": float(differences.mean()),
                           "episode_delta_std": float(differences.std(ddof=1)),
                           "interpretation": "paired evaluation only; one trained seed, not seed robustness"})
    delta = {row["metric"]: row["mean_delta"] for row in paired if row["baseline"] == "d4_geometric_control"}
    passed = (delta["return"] > 0 and delta["hungarian_assignment_distance"] < 0
              and delta["coverage_radius_auc"] > 0 and delta["collision_step_rate"] <= 0)
    recorded_gate = json.loads((root / "exploratory_gate.json").read_text())
    assert recorded_gate["passed"] == passed
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "verification.json", {"passed": True, "run_root": str(root), "evidence": evidence,
                                             "n_training_seeds": 1, "paper_level_claim": False,
                                             "exploratory_promotion_gate": passed})
    write_csv(output / "final_metrics.csv", table)
    write_csv(output / "paired_evaluation_deltas.csv", paired)
    colors = {"d4_geometric_control": "#367B58", "d4_dirichlet_gsp": "#B53B49"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric, title in zip(axes.flat,
            ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate"),
            ("Return (higher)", "Assignment distance (lower)", "Coverage-radius AUC (higher)", "Collision rate (lower)")):
        for method in METHODS:
            axis.plot(np.asarray(args.checkpoint_steps) / 1000,
                      [float(row[metric]) for row in curves[method]], marker="o",
                      color=colors[method], label=LABELS[method])
        axis.set(title=title, xlabel="Training transitions (thousands)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("D4 actor-only screen: seed 1, 100 evaluation episodes per checkpoint")
    figure.savefig(output / "learning_curves.png", dpi=160)
    plt.close(figure)
    lines = ["# D4 Actor-Only Seed-1 Result", "", "Protocol verification: PASS.", "",
             "Both methods completed exactly 100000 training transitions and 12000 CUDA agent updates.",
             "Five checkpoints per method; 100 episodes per checkpoint; 500 paired final episodes.",
             "The statistical training unit is one seed. No cross-seed significance is claimed.", "",
             "| Method | Return | Assignment distance | Radius AUC | Collision rate | Coverage@0.10 count |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        lines.append(f"| {LABELS[row['method']]} | {row['return']:.4f} | {row['hungarian_assignment_distance']:.5f} | "
                     f"{row['coverage_radius_auc']:.5f} | {row['collision_step_rate']:.5f} | {row['final_coverage']:.4f} |")
    lines += ["", "## Decision", "", f"Predeclared matched-pair exploratory gate: {'PASS' if passed else 'FAIL'}.", "",
              "Archived baselines are evaluated on the same states but their historical training device and",
              "audit sampling are not matched to the current pair. They are contextual comparisons only.",
              "No manuscript-level positive claim is established by this single-seed screen.", "",
              "![Learning curves](learning_curves.png)", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return {"output": str(output), "verification": "passed", "exploratory_gate": passed, "metrics": table}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(analyze(options.run_dir.resolve(), options.output_dir.resolve()), indent=2))
