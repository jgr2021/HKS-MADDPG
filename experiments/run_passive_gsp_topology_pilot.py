import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from tensorboardX import SummaryWriter
from torch.autograd import Variable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.buffer import ReplayBuffer
from utils.make_env import make_env


USE_CUDA = torch.cuda.is_available() and os.environ.get("MADDPG_FORCE_CPU") != "1"
STATUS_TITLE = "Passive GSP Topology Pilot Status"
PLOT_TITLE_PREFIX = "Passive GSP topology pilot"

METHODS = {
    "raw_mlp": {
        "env_id": "simple_spread",
        "actor_model": "mlp",
        "short": "raw",
        "actor_input_dim": 18,
    },
    "gsp_3node_aa_hks": {
        "env_id": "simple_spread",
        "actor_model": "passive_topology_3node_aa_hks",
        "short": "aa3",
        "actor_input_dim": 21,
    },
    "gsp_6node_al_hks": {
        "env_id": "simple_spread",
        "actor_model": "passive_topology_6node_al_hks",
        "short": "al6",
        "actor_input_dim": 21,
    },
    "gsp_6node_aal_hks": {
        "env_id": "simple_spread",
        "actor_model": "passive_topology_6node_aal_hks",
        "short": "aal6",
        "actor_input_dim": 21,
    },
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
    "collision_pair_step_mean",
    "time_to_first_full_coverage_step",
]


def command_output(command):
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


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
    rows = []
    try:
        for episode in range(episodes):
            test_seed = seed + episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = env.reset()
            episode_return = 0.0
            max_coverage = 0
            final_coverage = 0
            final_distance = 0.0
            final_collisions = 0
            collision_steps = 0
            collision_pair_steps = 0
            elapsed_steps = 0
            first_full_coverage_step = np.nan
            for step_index in range(episode_length):
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
                elapsed_steps += 1
                collision_pair_steps += final_collisions
                if final_collisions > 0:
                    collision_steps += 1
                if final_coverage == 3 and np.isnan(first_full_coverage_step):
                    first_full_coverage_step = step_index + 1
                if all(dones):
                    break
            collision_step_rate = (
                float(collision_steps) / float(elapsed_steps)
                if elapsed_steps else 0.0
            )
            collision_pair_step_mean = (
                float(collision_pair_steps) / float(elapsed_steps)
                if elapsed_steps else 0.0
            )
            rows.append(
                {
                    "eval_episode": episode,
                    "test_seed": test_seed,
                    "return": episode_return,
                    "final_coverage": final_coverage,
                    "max_coverage": max_coverage,
                    "final3": int(final_coverage == 3),
                    "max3": int(max_coverage == 3),
                    "nearest_landmark_distance": final_distance,
                    "collisions": final_collisions,
                    "collision_step_rate": collision_step_rate,
                    "collision_pair_step_mean": collision_pair_step_mean,
                    "time_to_first_full_coverage_step": first_full_coverage_step,
                }
            )
    finally:
        env.close()
    return rows


def summarize_episode_rows(rows):
    summary = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        if np.all(np.isnan(values)):
            summary[metric] = float("nan")
        else:
            summary[metric] = float(np.nanmean(values))
    summary["eval_episodes"] = len(rows)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_versions(output_dir):
    write_json(
        output_dir / "runtime_versions.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        },
    )
    freeze = command_output([sys.executable, "-m", "pip", "freeze"])
    (output_dir / "package_versions.txt").write_text(freeze + "\n", encoding="utf-8")


def create_model_run_dir(env_id, model_name):
    model_dir = Path("./models") / env_id / model_name
    if not model_dir.exists():
        curr_run = "run1"
    else:
        existing = [
            int(str(folder.name).split("run")[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        curr_run = "run1" if not existing else f"run{max(existing) + 1}"
    run_dir = model_dir / curr_run
    log_dir = run_dir / "logs"
    os.makedirs(log_dir)
    return run_dir, log_dir


def actor_augmented_shape(policy, obs_tensor):
    if hasattr(policy, "augment_observation"):
        with torch.no_grad():
            augmented = policy.augment_observation(obs_tensor)
        return list(augmented.shape)
    return list(obs_tensor.shape)


def capture_update_shapes(maddpg, sample, agent_i):
    obs, acs, _, next_obs, _ = sample
    critic_input = torch.cat((*obs, *acs), dim=1)
    target_actions = [
        policy(nobs)
        for policy, nobs in zip(maddpg.target_policies, next_obs)
    ]
    target_critic_input = torch.cat((*next_obs, *target_actions), dim=1)
    policy = maddpg.agents[agent_i].policy
    target_policy = maddpg.agents[agent_i].target_policy
    policy_mlp = getattr(policy, "mlp", policy)
    target_policy_mlp = getattr(target_policy, "mlp", target_policy)
    policy_fc1 = getattr(policy_mlp, "fc1", None)
    target_policy_fc1 = getattr(target_policy_mlp, "fc1", None)
    report = {
        "agent_i": agent_i,
        "raw_actor_tensor_shape": list(obs[agent_i].shape),
        "actor_augmented_tensor_shape": actor_augmented_shape(policy, obs[agent_i]),
        "target_actor_raw_tensor_shape": list(next_obs[agent_i].shape),
        "target_actor_augmented_tensor_shape": actor_augmented_shape(
            target_policy,
            next_obs[agent_i],
        ),
        "critic_input_tensor_shape": list(critic_input.shape),
        "target_critic_input_tensor_shape": list(target_critic_input.shape),
        "replay_obs_tensor_shapes": [list(item.shape) for item in obs],
        "replay_next_obs_tensor_shapes": [list(item.shape) for item in next_obs],
        "action_tensor_shapes": [list(item.shape) for item in acs],
        "policy_fc1_in_features": int(
            policy_fc1.in_features
            if policy_fc1 is not None else getattr(policy, "actor_input_dim", obs[agent_i].shape[1])
        ),
        "target_policy_fc1_in_features": int(
            target_policy_fc1.in_features
            if target_policy_fc1 is not None
            else getattr(target_policy, "actor_input_dim", next_obs[agent_i].shape[1])
        ),
        "critic_fc1_in_features": int(maddpg.agents[agent_i].critic.fc1.in_features),
        "target_critic_fc1_in_features": int(
            maddpg.agents[agent_i].target_critic.fc1.in_features
        ),
    }
    for attribute in [
        "last_graph_node_shape",
        "last_graph_embedding_shape",
        "last_action_feature_shape",
        "last_energy_gate_shape",
    ]:
        value = getattr(policy, attribute, None)
        if value is not None:
            report[attribute] = list(value)
    return report


def train_for_env_steps(config, total_env_steps, checkpoint_steps):
    if total_env_steps % config.n_rollout_threads != 0:
        raise ValueError(
            "total_env_steps must be divisible by n_rollout_threads for exact stopping."
        )
    for step in checkpoint_steps:
        if step % config.n_rollout_threads != 0:
            raise ValueError("checkpoint steps must be divisible by n_rollout_threads.")

    model_run_dir, log_dir = create_model_run_dir(config.env_id, config.model_name)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(
        config.env_id,
        config.n_rollout_threads,
        config.seed,
        config.discrete_action,
    )
    maddpg = MADDPG.init_from_env(
        env,
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        actor_lr=getattr(config, "actor_lr", None),
        critic_lr=getattr(config, "critic_lr", None),
        hidden_dim=config.hidden_dim,
        actor_model=config.actor_model,
        actor_anchor=getattr(config, "actor_anchor", None),
    )
    pretrained_policy_path = getattr(config, "pretrained_policy_path", None)
    if pretrained_policy_path:
        payload = torch.load(pretrained_policy_path, map_location="cpu")
        load_weights = bool(getattr(config, "pretrained_load_weights", True))
        for agent in maddpg.agents:
            if not hasattr(agent.policy, "load_probe_payload"):
                raise RuntimeError(
                    f"Actor {type(agent.policy).__name__} cannot load a probe payload."
                )
            agent.policy.load_probe_payload(payload, load_weights=load_weights)
            agent.target_policy.load_probe_payload(payload, load_weights=load_weights)
    maddpg.capture_policy_audit_references()
    if maddpg.requires_fixed_teacher:
        maddpg.capture_policy_anchors()
    replay_buffer = ReplayBuffer(
        config.buffer_length,
        maddpg.nagents,
        [obsp.shape[0] for obsp in env.observation_space],
        [
            acsp.shape[0] if isinstance(acsp, Box) else acsp.n
            for acsp in env.action_space
        ],
    )
    raw_obs_dims = [int(space.shape[0]) for space in env.observation_space]
    if any(
        buffer.shape[1] != raw_obs_dims[index]
        for index, buffer in enumerate(replay_buffer.obs_buffs)
    ):
        raise RuntimeError("Replay buffer observations must remain raw environment observations.")

    global_env_steps = 0
    episode_batches = 0
    env_episodes_started = 0
    saved_checkpoints = set()
    shape_report = None
    policy_audit_observations_captured = False
    checkpoint_steps = sorted(checkpoint_steps)

    try:
        while global_env_steps < total_env_steps:
            episode_batches += 1
            env_episodes_started += config.n_rollout_threads
            print(
                f"Episode batch {episode_batches} "
                f"(env episodes started {env_episodes_started}); "
                f"global_env_steps={global_env_steps}/{total_env_steps}",
                flush=True,
            )
            obs = env.reset()
            maddpg.prep_rollouts(device="cpu")
            remaining = max(0, total_env_steps - global_env_steps) / total_env_steps
            maddpg.scale_noise(
                config.final_noise_scale
                + (config.init_noise_scale - config.final_noise_scale) * remaining
            )
            maddpg.reset_noise()

            for _ in range(config.episode_length):
                if global_env_steps >= total_env_steps:
                    break
                torch_obs = [
                    Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                    for i in range(maddpg.nagents)
                ]
                torch_agent_actions = maddpg.step(torch_obs, explore=True)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                actions = [
                    [ac[i] for ac in agent_actions]
                    for i in range(config.n_rollout_threads)
                ]
                next_obs, rewards, dones, _ = env.step(actions)
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                obs = next_obs
                global_env_steps += config.n_rollout_threads

                if (
                    len(replay_buffer) >= config.batch_size
                    and (global_env_steps % config.steps_per_update) < config.n_rollout_threads
                ):
                    maddpg.prep_training(device="gpu" if USE_CUDA else "cpu")
                    if not policy_audit_observations_captured:
                        audit_sample = replay_buffer.sample(
                            min(config.policy_audit_batch_size, len(replay_buffer)),
                            to_gpu=USE_CUDA,
                            norm_rews=False,
                        )
                        maddpg.set_policy_audit_observations(audit_sample[0])
                        policy_audit_observations_captured = True
                    for _ in range(config.n_rollout_threads):
                        for a_i in range(maddpg.nagents):
                            sample = replay_buffer.sample(
                                config.batch_size,
                                to_gpu=USE_CUDA,
                            )
                            if shape_report is None:
                                shape_report = capture_update_shapes(maddpg, sample, a_i)
                                shape_report["global_env_steps_at_first_update"] = global_env_steps
                                shape_report["replay_buffer_obs_dims"] = [
                                    int(buffer.shape[1]) for buffer in replay_buffer.obs_buffs
                                ]
                            maddpg.update(
                                sample,
                                a_i,
                                logger=logger,
                                update_actor=actor_updates_enabled(
                                    config, global_env_steps
                                ),
                            )
                        maddpg.update_all_targets()
                    maddpg.prep_rollouts(device="cpu")

                for checkpoint_step in checkpoint_steps:
                    if (
                        global_env_steps >= checkpoint_step
                        and checkpoint_step not in saved_checkpoints
                    ):
                        os.makedirs(model_run_dir / "incremental", exist_ok=True)
                        maddpg.save(
                            model_run_dir / "incremental" / f"model_step{checkpoint_step}.pt"
                        )
                        maddpg.save(model_run_dir / "model.pt")
                        saved_checkpoints.add(checkpoint_step)

            ep_rews = replay_buffer.get_average_rewards(
                config.episode_length * config.n_rollout_threads
            )
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalar(
                    f"agent{a_i}/mean_episode_rewards",
                    a_ep_rew,
                    global_env_steps,
                )

        if global_env_steps != total_env_steps:
            raise RuntimeError(
                f"Training stopped at {global_env_steps}, expected {total_env_steps}."
            )
        maddpg.save(model_run_dir / "model.pt")
        counters = {
            "global_env_steps": global_env_steps,
            "total_env_steps": total_env_steps,
            "episode_batches": episode_batches,
            "env_episodes_started": env_episodes_started,
            "checkpoint_steps": checkpoint_steps,
            "saved_checkpoints": sorted(saved_checkpoints),
        }
        write_json(model_run_dir / "training_counters.json", counters)
        if shape_report is not None:
            write_json(model_run_dir / "tensor_shape_report.json", shape_report)
    finally:
        env.close()
        logger.export_scalars_to_json(str(log_dir / "summary.json"))
        logger.close()

    return model_run_dir, shape_report


def actor_updates_enabled(config, global_env_steps):
    """Whether the actor is inside its configured half-open update window."""
    start = int(getattr(config, "actor_update_start_step", 0))
    end = int(getattr(config, "actor_update_end_step", 2**63 - 1))
    if end < start:
        raise ValueError(
            "actor_update_end_step must be at least actor_update_start_step"
        )
    return start <= int(global_env_steps) < end


def train_one(method, seed, args, root_dir, stamp):
    spec = METHODS[method]
    run_dir = root_dir / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"pgt_{spec['short']}_s{seed}_{stamp}"
    config = SimpleNamespace(
        env_id=spec["env_id"],
        model_name=model_name,
        seed=seed,
        n_rollout_threads=args.n_rollout_threads,
        n_training_threads=args.n_training_threads,
        buffer_length=args.buffer_length,
        n_episodes=getattr(args, "budget_episodes", args.total_env_steps),
        episode_length=args.episode_length,
        steps_per_update=args.steps_per_update,
        batch_size=args.batch_size,
        n_exploration_eps=getattr(args, "budget_episodes", args.total_env_steps),
        init_noise_scale=0.3,
        final_noise_scale=0.0,
        save_interval=max(args.total_env_steps + args.n_rollout_threads, 100000),
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        actor_lr=float(spec.get("actor_lr", args.lr)),
        critic_lr=float(spec.get("critic_lr", args.lr)),
        tau=0.01,
        agent_alg="MADDPG",
        adversary_alg="MADDPG",
        discrete_action=True,
        actor_model=spec["actor_model"],
        pretrained_policy_path=spec.get("pretrained_policy_path"),
        pretrained_load_weights=spec.get("pretrained_load_weights", True),
        actor_anchor=spec.get("actor_anchor"),
        actor_update_start_step=int(spec.get("actor_update_start_step", 0)),
        actor_update_end_step=int(
            spec.get("actor_update_end_step", 2**63 - 1)
        ),
        policy_audit_batch_size=args.policy_audit_batch_size,
        print_interval=args.print_interval,
    )
    resolved_config = vars(config).copy()
    n_agents = int(spec.get("n_agents", 3))
    action_dim = int(spec.get("action_dim", 5))
    raw_observation_dim = int(spec.get("raw_observation_dim", 18))
    resolved_config.update(
        {
            "method": method,
            "actor_input_dim": spec["actor_input_dim"],
            "raw_observation_dim": raw_observation_dim,
            "critic_input_dim": raw_observation_dim * n_agents + action_dim * n_agents,
            "total_env_steps": args.total_env_steps,
            "checkpoint_steps": args.checkpoint_steps,
            "protocol": "actor_only_graph_augmentation_raw_critic",
        }
    )
    write_json(run_dir / "resolved_config.json", resolved_config)
    save_versions(run_dir)

    train_log = run_dir / "training.log"
    started = time.perf_counter()
    with train_log.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            model_run_dir, shape_report = train_for_env_steps(
                config,
                total_env_steps=args.total_env_steps,
                checkpoint_steps=args.checkpoint_steps,
            )
    train_wall_time_sec = time.perf_counter() - started

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    final_checkpoint_label = f"model_final_{args.total_env_steps}"
    shutil.copy2(model_run_dir / "model.pt", checkpoint_dir / f"{final_checkpoint_label}.pt")
    for checkpoint_step in args.checkpoint_steps:
        source = model_run_dir / "incremental" / f"model_step{checkpoint_step}.pt"
        if source.exists():
            shutil.copy2(source, checkpoint_dir / f"model_step{checkpoint_step}.pt")
    if shape_report is not None:
        write_json(run_dir / "tensor_shape_report.json", shape_report)
    for filename in ["training_counters.json", "tensor_shape_report.json"]:
        source = model_run_dir / filename
        if source.exists():
            shutil.copy2(source, run_dir / filename)
    if (model_run_dir / "logs" / "summary.json").exists():
        shutil.copy2(model_run_dir / "logs" / "summary.json", run_dir / "tensorboard_summary.json")

    final_eval_rows = evaluate_model(
        spec["env_id"],
        model_run_dir / "model.pt",
        episodes=args.eval_episodes,
        episode_length=args.episode_length,
        seed=args.eval_seed_base + seed * 10000,
    )
    for row in final_eval_rows:
        row.update({"method": method, "seed": seed, "checkpoint": final_checkpoint_label})
    write_csv(run_dir / "per_evaluation_episode_metrics.csv", final_eval_rows)

    curve_rows = []
    checkpoints = [
        (model_run_dir / "incremental" / f"model_step{step}.pt", f"model_step{step}")
        for step in args.checkpoint_steps
    ]
    for checkpoint, label in checkpoints:
        if not checkpoint.exists():
            continue
        checkpoint_eval = evaluate_model(
            spec["env_id"],
            checkpoint,
            episodes=args.curve_eval_episodes,
            episode_length=args.episode_length,
            seed=args.curve_eval_seed_base + seed * 10000,
        )
        checkpoint_summary = summarize_episode_rows(checkpoint_eval)
        checkpoint_summary.update(
            {
                "method": method,
                "seed": seed,
                "checkpoint": label,
                "checkpoint_path": str(checkpoint),
            }
        )
        curve_rows.append(checkpoint_summary)
    write_csv(run_dir / "learning_curve_eval.csv", curve_rows)

    summary = summarize_episode_rows(final_eval_rows)
    counters_path = run_dir / "training_counters.json"
    counters = json.loads(counters_path.read_text(encoding="utf-8")) if counters_path.exists() else {}
    summary.update(
        {
            "method": method,
            "seed": seed,
            "env_id": spec["env_id"],
            "actor_model": spec["actor_model"],
            "model_name": model_name,
            "source_model_run_dir": str(model_run_dir),
            "train_wall_time_sec": train_wall_time_sec,
            "env_steps_per_sec": (
                float(args.total_env_steps) / train_wall_time_sec
                if train_wall_time_sec > 0 else float("nan")
            ),
            "global_env_steps": counters.get("global_env_steps"),
            "status": "completed",
        }
    )
    write_json(run_dir / "summary.json", summary)
    return summary, curve_rows


def aggregate(seed_summaries):
    rows = []
    for method in METHODS:
        method_rows = [
            row for row in seed_summaries
            if row["method"] == method and row["status"] == "completed"
        ]
        if not method_rows:
            continue
        output = {"method": method, "n_seeds": len(method_rows)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in method_rows], dtype=np.float64)
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append(output)
    return rows


def plot_learning_curves(curve_rows, output_dir, checkpoint_steps):
    if not curve_rows:
        return
    checkpoint_order = [f"model_step{step}" for step in checkpoint_steps]
    x_lookup = {name: index for index, name in enumerate(checkpoint_order)}
    for metric in ["return", "final_coverage", "final3"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in METHODS:
            method_rows = [row for row in curve_rows if row["method"] == method]
            if not method_rows:
                continue
            xs = []
            means = []
            stds = []
            for checkpoint in checkpoint_order:
                values = [
                    row[metric]
                    for row in method_rows
                    if row["checkpoint"] == checkpoint
                ]
                if not values:
                    continue
                values = np.asarray(values, dtype=np.float64)
                xs.append(x_lookup[checkpoint])
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=method)
        ax.set_xticks(range(len(checkpoint_order)))
        ax.set_xticklabels(checkpoint_order, rotation=20)
        ax.set_title(f"{PLOT_TITLE_PREFIX}: {metric}")
        ax.set_xlabel("checkpoint")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"learning_curve_{metric}.png", dpi=180)
        plt.close(fig)


def write_status(root_dir, rows):
    lines = [
        f"# {STATUS_TITLE}",
        "",
        f"Updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| method | seed | status | steps | steps/sec | return | final | max | final3 | max3 | distance | collisions | collision-step-rate |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") == "completed":
            lines.append(
                f"| {row['method']} | {row['seed']} | {row['status']} | "
                f"{row.get('global_env_steps', '')} | "
                f"{row.get('env_steps_per_sec', float('nan')):.4f} | "
                f"{row['return']:.4f} | "
                f"{row['final_coverage']:.4f} | {row['max_coverage']:.4f} | "
                f"{row['final3']:.4f} | {row['max3']:.4f} | "
                f"{row['nearest_landmark_distance']:.4f} | {row['collisions']:.4f} | "
                f"{row.get('collision_step_rate', float('nan')):.4f} |"
            )
        else:
            lines.append(
                f"| {row.get('method', '')} | {row.get('seed', '')} | "
                f"{row.get('status', 'failed')} |  |  |  |  |  |  |  |  |  |  |"
            )
    (root_dir / "TRAINING_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/passive_gsp_topology_pilot")
    parser.add_argument("--total-env-steps", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--curve-eval-episodes", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--methods", default=",".join(METHODS.keys()))
    parser.add_argument("--n-rollout-threads", type=int, default=4)
    parser.add_argument("--n-training-threads", type=int, default=6)
    parser.add_argument("--buffer-length", type=int, default=int(1e6))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-per-update", type=int, default=100)
    parser.add_argument("--checkpoint-steps", default="5000,10000,15000,20000")
    parser.add_argument("--print-interval", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--policy-audit-batch-size", type=int, default=256)
    parser.add_argument("--eval-seed-base", type=int, default=990000)
    parser.add_argument("--curve-eval-seed-base", type=int, default=770000)
    args = parser.parse_args()
    args.checkpoint_steps = [
        int(step.strip()) for step in str(args.checkpoint_steps).split(",") if step.strip()
    ]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.output_dir) / f"run_{stamp}"
    root_dir.mkdir(parents=True, exist_ok=False)
    save_versions(root_dir)
    write_json(root_dir / "runner_config.json", vars(args))
    (root_dir / "launch_command.txt").write_text(
        subprocess.list2cmdline([sys.executable] + sys.argv) + "\n",
        encoding="utf-8",
    )

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
    method_boundary_specs = {}
    for method in methods:
        spec = METHODS[method]
        n_agents = int(spec.get("n_agents", 3))
        action_dim = int(spec.get("action_dim", 5))
        raw_observation_dim = int(spec.get("raw_observation_dim", 18))
        method_boundary_specs[method] = {
            "n_agents": n_agents,
            "action_dim": action_dim,
            "environment_observation_dim": raw_observation_dim,
            "raw_actor_tensor_dim": raw_observation_dim,
            "actor_model_input_dim": int(spec["actor_input_dim"]),
            "critic_input_dim": raw_observation_dim * n_agents + action_dim * n_agents,
            "target_critic_input_dim": raw_observation_dim * n_agents + action_dim * n_agents,
            "replay_obs_dims": [raw_observation_dim] * n_agents,
        }
    boundary_values = list(method_boundary_specs.values())
    homogeneous_boundaries = all(item == boundary_values[0] for item in boundary_values[1:])
    common_boundary = boundary_values[0] if homogeneous_boundaries else None
    write_json(
        root_dir / "protocol_metadata.json",
        {
            "protocol": "actor_only_graph_augmentation_raw_critic",
            "methods": methods,
            "method_specs": {method: METHODS[method] for method in methods},
            "seeds": seeds,
            "total_env_steps": args.total_env_steps,
            "checkpoint_steps": args.checkpoint_steps,
            "eval_episodes_final": args.eval_episodes,
            "eval_episodes_checkpoint": args.curve_eval_episodes,
            "evaluation_exploration_enabled": False,
            "method_boundary_specs": method_boundary_specs,
            "homogeneous_boundaries": homogeneous_boundaries,
            "environment_observation_dim": (
                common_boundary["environment_observation_dim"] if common_boundary else None
            ),
            "raw_actor_input_dim": (
                common_boundary["raw_actor_tensor_dim"] if common_boundary else None
            ),
            "gsp_actor_raw_tensor_dim": (
                common_boundary["raw_actor_tensor_dim"] if common_boundary else None
            ),
            "gsp_actor_augmented_tensor_dim": (
                common_boundary["actor_model_input_dim"] if common_boundary else None
            ),
            "critic_input_dim": (
                common_boundary["critic_input_dim"] if common_boundary else None
            ),
            "target_critic_input_dim": (
                common_boundary["target_critic_input_dim"] if common_boundary else None
            ),
            "replay_obs_dims": (
                common_boundary["replay_obs_dims"] if common_boundary else None
            ),
            "action_space_unchanged": True,
            "reward_function_unchanged": True,
            "explicit_communication": False,
            "handcrafted_action_prior": False,
            "reward_shaping": False,
            "global_state_leakage": False,
            "critic_side_gsp_features": False,
            "new_metric_columns": [
                "collision_step_rate",
                "collision_pair_step_mean",
                "time_to_first_full_coverage_step",
                "env_steps_per_sec",
            ],
        },
    )

    seed_summaries = []
    curve_rows = []
    for method in methods:
        for seed in seeds:
            try:
                summary, learning_curve = train_one(method, seed, args, root_dir, stamp)
                seed_summaries.append(summary)
                curve_rows.extend(learning_curve)
            except Exception as exc:
                failure_dir = root_dir / method / f"seed_{seed}"
                failure_dir.mkdir(parents=True, exist_ok=True)
                (failure_dir / "failure.txt").write_text(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    encoding="utf-8",
                )
                seed_summaries.append(
                    {
                        "method": method,
                        "seed": seed,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            write_status(root_dir, seed_summaries)
            write_json(root_dir / "seed_summaries.json", seed_summaries)
            write_csv(root_dir / "seed_summaries.csv", seed_summaries)
            write_csv(root_dir / "learning_curve_eval.csv", curve_rows)

    aggregate_rows = aggregate(seed_summaries)
    write_csv(root_dir / "aggregate_summary.csv", aggregate_rows)
    write_json(root_dir / "aggregate_summary.json", aggregate_rows)
    plot_learning_curves(curve_rows, root_dir, args.checkpoint_steps)
    write_status(root_dir, seed_summaries)
    print(root_dir)


if __name__ == "__main__":
    main()
