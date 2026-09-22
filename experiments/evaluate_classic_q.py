"""Evaluate independent DQN checkpoints with the project's coordination metrics."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gym.spaces import Discrete


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.classic_q_baselines import IndependentDQNAgent
from experiments.evaluate_supervised_probe_policies import geometry
from utils.make_env import make_env


def load_agents(checkpoint_path, env, hidden_dim):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    algorithm = checkpoint["algorithm"]
    use_double_q = algorithm == "double_dqn"
    if algorithm not in {"dqn", "double_dqn"}:
        raise ValueError(f"Unsupported classic Q checkpoint: {algorithm}")
    agents = []
    for observation_space, action_space, state in zip(
        env.observation_space, env.action_space, checkpoint["agent_states"]
    ):
        if not isinstance(action_space, Discrete):
            raise ValueError("Classic Q evaluation requires discrete actions")
        agent = IndependentDQNAgent(
            observation_space.shape[0],
            action_space.n,
            hidden_dim=hidden_dim,
            use_double_q=use_double_q,
            device="cpu",
        )
        agent.q_net.load_state_dict(state["q_net"])
        agent.target_q_net.load_state_dict(state["target_q_net"])
        agent.set_exploration(0.0)
        agent.prep_rollouts()
        agents.append(agent)
    return algorithm, agents


def evaluate(model_path, episodes, horizon, seed_base, hidden_dim):
    env = make_env("simple_spread", discrete_action=True)
    rows = []
    try:
        algorithm, agents = load_agents(model_path, env, hidden_dim)
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            total_return = 0.0
            collision_steps = 0
            minimum_separation = float("inf")
            max_coverage = 0
            elapsed = 0
            for _ in range(horizon):
                actions = []
                for index, agent in enumerate(agents):
                    tensor = torch.as_tensor(obs[index:index + 1])
                    actions.append(agent.step(tensor, explore=False)[0].numpy())
                obs, rewards, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                elapsed += 1
                assignment, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision
                minimum_separation = min(minimum_separation, separation)
                max_coverage = max(max_coverage, coverage)
                if all(dones):
                    break
            rows.append({
                "algorithm": algorithm,
                "episode": episode,
                "test_seed": seed,
                "return": total_return,
                "hungarian_assignment_distance": assignment,
                "coverage_radius_auc": auc,
                "collision_step_rate": collision_steps / float(elapsed),
                "minimum_agent_separation": minimum_separation,
                "final_coverage": coverage,
                "max_coverage": max_coverage,
            })
    finally:
        env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=64_000_000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = evaluate(
        args.model_path,
        args.episodes,
        args.horizon,
        args.seed_base,
        args.hidden_dim,
    )
    with (output / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "algorithm": rows[0]["algorithm"],
        "episodes": len(rows),
        "model_path": str(Path(args.model_path)),
    }
    for metric in rows[0]:
        if metric in {"algorithm", "episode", "test_seed"}:
            continue
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric + "_mean"] = float(values.mean())
        summary[metric + "_std"] = float(values.std(ddof=1))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
