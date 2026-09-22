import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG

def get_coverage_metrics(env, threshold=0.1):
    agents = env.world.agents
    landmarks = env.world.landmarks

    distances = np.array([
        [
            np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            for landmark in landmarks
        ]
        for agent in agents
    ])

    min_dist_per_landmark = distances.min(axis=0)
    occupied = int(np.sum(min_dist_per_landmark < threshold))
    min_dist_sum = float(min_dist_per_landmark.sum())

    collisions = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            distance = np.linalg.norm(
                agents[i].state.p_pos - agents[j].state.p_pos
            )
            if distance < agents[i].size + agents[j].size:
                collisions += 1

    return occupied, min_dist_sum, collisions

def run(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    env.seed(config.seed)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        max_occupied = 0
        final_occupied = 0
        final_min_dist_sum = 0.0
        final_collisions = 0
        episode_return = 0.0
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            episode_return += rewards[0]

            occupied, min_dist_sum, collisions = get_coverage_metrics(env)

            max_occupied = max(max_occupied, occupied)
            final_occupied = occupied
            final_min_dist_sum = min_dist_sum
            final_collisions = collisions
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)
        print(
            f"[Eval {ep_i + 1:02d}] "
            f"return={episode_return:.2f} | "
            f"max_coverage={max_occupied}/3 | "
            f"final_coverage={final_occupied}/3 | "
            f"final_min_dist_sum={final_min_dist_sum:.3f} | "
            f"collisions={final_collisions}"
        )

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    config = parser.parse_args()

    run(config)