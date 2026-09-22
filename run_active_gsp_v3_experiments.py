import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import run as train_run
from utils.make_env import make_env


EXPERIMENTS = [
    ("raw_mlp", "simple_spread", "mlp"),
    ("action_score_control", "simple_spread", "action_score_control"),
    ("action_raw_potential", "simple_spread", "action_raw_potential"),
    ("v3_active_gsp", "simple_spread", "active_gsp_v3"),
    ("action_raw_potential_scaled", "simple_spread", "action_raw_potential_scaled"),
    ("v3_active_gsp_scaled", "simple_spread", "active_gsp_v3_scaled"),
    ("action_raw_potential_prior", "simple_spread", "action_raw_potential_prior"),
    ("v3_active_gsp_prior", "simple_spread", "active_gsp_v3_prior"),
    ("action_raw_potential_residual005", "simple_spread", "action_raw_potential_residual005"),
    ("v3_active_gsp_residual005", "simple_spread", "active_gsp_v3_residual005"),
    ("action_raw_potential_fixed", "simple_spread", "action_raw_potential_fixed"),
    ("v3_active_gsp_fixed", "simple_spread", "active_gsp_v3_fixed"),
]
EXPERIMENT_MAP = {
    name: (name, env_id, actor_model)
    for name, env_id, actor_model in EXPERIMENTS
}


def latest_run_dir(env_id, model_name):
    model_dir = Path("models") / env_id / model_name
    run_dirs = [
        path for path in model_dir.iterdir()
        if path.is_dir() and path.name.startswith("run")
    ]
    return sorted(run_dirs, key=lambda path: int(path.name.replace("run", "")))[-1]


def coverage_metrics(env, threshold=0.10):
    agents = env.world.agents
    landmarks = env.world.landmarks
    distances = np.asarray(
        [
            [
                np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for landmark in landmarks
            ]
            for agent in agents
        ],
        dtype=np.float32,
    )
    nearest = distances.min(axis=0)
    collisions = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            dist = np.linalg.norm(agents[i].state.p_pos - agents[j].state.p_pos)
            if dist < agents[i].size + agents[j].size:
                collisions += 1
    return int((nearest < threshold).sum()), float(nearest.mean()), collisions


def evaluate_model(env_id, model_path, episodes, episode_length, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    maddpg = MADDPG.init_from_save(str(model_path))
    maddpg.prep_rollouts(device="cpu")
    env = make_env(env_id, discrete_action=maddpg.discrete_action)
    env.seed(seed)

    returns = []
    final_coverages = []
    max_coverages = []
    final_distances = []
    collisions = []

    try:
        for episode in range(episodes):
            np.random.seed(seed + episode)
            obs = env.reset()
            episode_return = 0.0
            max_coverage = 0
            final_coverage = 0
            final_distance = 0.0
            final_collisions = 0
            for _ in range(episode_length):
                torch_obs = [
                    torch.as_tensor(obs[i], dtype=torch.float32).view(1, -1)
                    for i in range(maddpg.nagents)
                ]
                with torch.no_grad():
                    torch_actions = maddpg.step(torch_obs, explore=False)
                actions = [
                    action.detach().cpu().numpy().flatten()
                    for action in torch_actions
                ]
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards))
                final_coverage, final_distance, final_collisions = coverage_metrics(env)
                max_coverage = max(max_coverage, final_coverage)
                if all(dones):
                    break
            returns.append(episode_return)
            final_coverages.append(final_coverage)
            max_coverages.append(max_coverage)
            final_distances.append(final_distance)
            collisions.append(final_collisions)
    finally:
        env.close()

    return {
        "return": float(np.mean(returns)),
        "final_coverage": float(np.mean(final_coverages)),
        "max_coverage": float(np.mean(max_coverages)),
        "nearest_landmark_distance": float(np.mean(final_distances)),
        "collisions": float(np.mean(collisions)),
    }


def plot_results(rows, output_path):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["experiment"], []).append(row)
    names = list(grouped.keys())

    def mean_and_std(metric):
        means = []
        stds = []
        for name in names:
            values = np.asarray(
                [float(row[metric]) for row in grouped[name]],
                dtype=np.float32,
            )
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=0)))
        return means, stds

    final_coverage, final_coverage_std = mean_and_std("final_coverage")
    max_coverage, max_coverage_std = mean_and_std("max_coverage")
    returns, returns_std = mean_and_std("return")
    distances, distances_std = mean_and_std("nearest_landmark_distance")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.reshape(-1)
    axes[0].bar(names, final_coverage, yerr=final_coverage_std, color="#4c78a8")
    axes[0].set_ylim(0, 3)
    axes[0].set_title("Mean final coverage")
    axes[0].set_ylabel("landmarks covered")

    axes[1].bar(names, max_coverage, yerr=max_coverage_std, color="#72b7b2")
    axes[1].set_ylim(0, 3)
    axes[1].set_title("Mean max coverage")

    axes[2].bar(names, returns, yerr=returns_std, color="#f58518")
    axes[2].set_title("Mean evaluation return")

    axes[3].bar(names, distances, yerr=distances_std, color="#54a24b")
    axes[3].set_title("Mean nearest-landmark distance")
    axes[3].set_ylabel("lower is better")

    for axis in axes:
        axis.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_results(rows, csv_path, plot_path):
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, plot_path)


def write_readme(path, args, selected_experiments, rows):
    lines = [
        "# Active GSP v3 Smoke Experiment",
        "",
        f"Created: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "This run uses the production local-observation-only fast path.",
        "It does not modify or stop the background v2/passive-GSP training job.",
        "",
        "## Configuration",
        "",
        f"- Episodes: `{args.episodes}`",
        f"- Eval episodes: `{args.eval_episodes}`",
        f"- Episode length: `{args.episode_length}`",
        f"- Seeds: `{args.seeds or args.seed}`",
        f"- Rollout threads: `{args.n_rollout_threads}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Steps per update: `{args.steps_per_update}`",
        "",
        "## Variants",
        "",
    ]
    for exp_name, env_id, actor_model in selected_experiments:
        lines.append(f"- `{exp_name}`: env `{env_id}`, actor `{actor_model}`")
    if rows:
        lines.extend(["", "## Results", ""])
        lines.append("| experiment | seed | return | final_coverage | max_coverage | nearest_distance | collisions | train_sec |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['experiment']} | {row['seed']} | "
                f"{float(row['return']):.4f} | "
                f"{float(row['final_coverage']):.4f} | "
                f"{float(row['max_coverage']):.4f} | "
                f"{float(row['nearest_landmark_distance']):.4f} | "
                f"{float(row['collisions']):.4f} | "
                f"{float(row['train_wall_time_sec']):.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=4)
    parser.add_argument("--n_training_threads", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps_per_update", type=int, default=100)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--prefix", default="active_gsp_v3_smoke200")
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = Path("experiments") / f"{args.prefix}_{stamp}"
    result_dir.mkdir(parents=True, exist_ok=False)

    if args.experiments == "all":
        selected_experiments = EXPERIMENTS
    else:
        names = [name.strip() for name in args.experiments.split(",") if name.strip()]
        selected_experiments = [EXPERIMENT_MAP[name] for name in names]

    seeds = [args.seed]
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    rows = []
    csv_path = result_dir / "results.csv"
    plot_path = result_dir / "summary.png"
    readme_path = result_dir / "README.md"
    write_readme(readme_path, args, selected_experiments, rows)

    for seed in seeds:
        for exp_name, env_id, actor_model in selected_experiments:
            model_name = f"{args.prefix}_{exp_name}_seed{seed}_{stamp}"
            config = SimpleNamespace(
                env_id=env_id,
                model_name=model_name,
                seed=seed,
                n_rollout_threads=args.n_rollout_threads,
                n_training_threads=args.n_training_threads,
                buffer_length=int(1e6),
                n_episodes=args.episodes,
                episode_length=args.episode_length,
                steps_per_update=args.steps_per_update,
                batch_size=args.batch_size,
                n_exploration_eps=args.episodes,
                init_noise_scale=0.3,
                final_noise_scale=0.0,
                save_interval=max(args.episodes + args.n_rollout_threads, 100000),
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                tau=0.01,
                agent_alg="MADDPG",
                adversary_alg="MADDPG",
                discrete_action=True,
                actor_model=actor_model,
                print_interval=args.print_interval,
            )
            print(
                f"\n=== Training {exp_name}: env={env_id}, "
                f"actor={actor_model}, seed={seed} ===",
                flush=True,
            )
            started = time.perf_counter()
            train_run(config)
            train_wall_time_sec = time.perf_counter() - started
            run_dir = latest_run_dir(env_id, model_name)
            metrics = evaluate_model(
                env_id,
                run_dir / "model.pt",
                episodes=args.eval_episodes,
                episode_length=args.episode_length,
                seed=seed + 2039,
            )
            row = {
                "experiment": exp_name,
                "seed": seed,
                "env_id": env_id,
                "actor_model": actor_model,
                "model_name": model_name,
                "run_dir": str(run_dir),
                "train_wall_time_sec": train_wall_time_sec,
                **metrics,
            }
            rows.append(row)
            print(row, flush=True)
            write_results(rows, csv_path, plot_path)
            write_readme(readme_path, args, selected_experiments, rows)
            print(f"Updated results: {csv_path}", flush=True)
            print(f"Updated plot: {plot_path}", flush=True)

    print(f"\nSaved results: {csv_path}", flush=True)
    print(f"Saved plot: {plot_path}", flush=True)
    print(f"Saved README: {readme_path}", flush=True)


if __name__ == "__main__":
    main()
