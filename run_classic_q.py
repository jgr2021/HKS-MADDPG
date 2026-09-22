import argparse
import os
from pathlib import Path

import numpy as np
import torch
from gym.spaces import Discrete
from tensorboardX import SummaryWriter

from algorithms.classic_q_baselines import IndependentDQNAgent
from utils.env_wrappers import DummyVecEnv, SubprocVecEnv
from utils.make_env import make_env
from utils.buffer import ReplayBuffer


USE_CUDA = torch.cuda.is_available()


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def _make_agents(env, config, device):
    if config.algorithm == "dqn":
        use_double_q = False
    elif config.algorithm == "double_dqn":
        use_double_q = True
    else:
        raise ValueError("Unknown algorithm %r" % config.algorithm)

    alg_hints = []
    for acsp, obsp in zip(env.action_space, env.observation_space):
        if not isinstance(acsp, Discrete):
            raise ValueError("Classic Q baselines require all action spaces to be discrete.")
        alg_hints.append(
            IndependentDQNAgent(
                num_in_obs=obsp.shape[0],
                num_out_acs=acsp.n,
                gamma=config.gamma,
                lr=config.lr,
                hidden_dim=config.hidden_dim,
                epsilon=config.init_epsilon,
                use_double_q=use_double_q,
                device=device,
            )
        )
    return alg_hints


def _compute_epsilon(step, config):
    if config.eps_decay <= 0:
        return float(config.init_epsilon)
    frac = min(1.0, float(step) / float(config.eps_decay))
    return float(config.init_epsilon + frac * (config.final_epsilon - config.init_epsilon))


def run(config):
    model_dir = Path("./models") / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = "run1"
    else:
        existing = [
            int(str(folder.name).split("run")[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        curr_run = "run%i" % (max(existing) + 1) if existing else "run1"

    run_dir = model_dir / curr_run
    log_dir = run_dir / "logs"
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    device = torch.device("cuda") if USE_CUDA else torch.device("cpu")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, True)
    agents = _make_agents(env, config, device)
    replay_buffer = ReplayBuffer(
        config.buffer_length,
        len(agents),
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.n for acsp in env.action_space],
    )

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print_interval = getattr(config, "print_interval", config.n_rollout_threads)
        if ep_i == 0 or ep_i % print_interval < config.n_rollout_threads:
            print(
                "Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes),
                flush=True,
            )

        obs = env.reset()
        for agent in agents:
            agent.prep_rollouts()

        for et_i in range(config.episode_length):
            torch_obs = [
                torch.Tensor(np.vstack(obs[:, i])).to(device)
                for i in range(len(agents))
            ]
            eps = _compute_epsilon(t, config)
            for agent in agents:
                agent.set_exploration(eps)

            agent_actions = [agent.step(torch_obs[i]).detach().cpu().numpy() for i, agent in enumerate(agents)]
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, _ = env.step(actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads

            if len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads:
                for agent in agents:
                    agent.prep_training()

                for _ in range(config.updates_per_step):
                    sample = replay_buffer.sample(
                        config.batch_size,
                        to_gpu=USE_CUDA,
                        norm_rews=False,
                    )
                    for a_i, agent in enumerate(agents):
                        agent.update(sample, a_i, logger=logger, niter=t)

                should_sync_target = t % config.target_update_interval < config.n_rollout_threads
                for agent in agents:
                    if should_sync_target:
                        agent.sync_target()
                    agent.prep_rollouts()

        ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar("agent%i/mean_episode_rewards" % a_i, a_ep_rew, ep_i)
            logger.add_scalar("agent%i/epsilon" % a_i, _compute_epsilon(t, config), ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            torch.save(
                {
                    "algorithm": config.algorithm,
                    "agent_states": [agent.state_dict() for agent in agents],
                    "seed": config.seed,
                },
                run_dir / "incremental" / ("model_ep%i.pt" % (ep_i + 1)),
            )
            torch.save(
                {
                    "algorithm": config.algorithm,
                    "agent_states": [agent.state_dict() for agent in agents],
                    "seed": config.seed,
                },
                run_dir / "model.pt",
            )

    torch.save(
        {
            "algorithm": config.algorithm,
            "agent_states": [agent.state_dict() for agent in agents],
            "seed": config.seed,
        },
        run_dir / "model.pt",
    )
    env.close()
    logger.export_scalars_to_json(str(log_dir / "summary.json"))
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name", help="Output directory for trained models")
    parser.add_argument("--algorithm", default="dqn", type=str, choices=["dqn", "double_dqn"])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--print_interval", default=1000, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--updates_per_step", default=1, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--target_update_interval", default=1000, type=int)
    parser.add_argument("--init_epsilon", default=1.0, type=float)
    parser.add_argument("--final_epsilon", default=0.05, type=float)
    parser.add_argument("--eps_decay", default=25000, type=int)
    config = parser.parse_args()
    run(config)
