"""Read-only-model local motion fidelity audit on actual legacy MPE branches."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.analyze_d4_actor_gpu_screen import write_csv, write_json
from experiments.diagnose_active_gsp_counterfactual_masks import policy_digest
from experiments.probe_action_value_representations import snapshot_world, restore_world
from experiments.run_d4_actor_gpu_screen import sha256, source_hashes
from utils.active_gsp_v3_features import _state_quantities, compute_active_gsp_v3_from_local_obs_batch
from utils.local_contact_motion import MODES, predict_local_candidate_positions
from utils.make_env import make_env


FEATURE_NAMES = ("coverage_potential", "crowding_potential", "coverage_energy", "crowding_energy")
SEPARATION_BINS = (("contact_lt030", 0, .30), ("near_030_031", .30, .31),
                   ("mid_031_060", .31, .60), ("far_ge060", .60, np.inf))


def assert_environment_supported(env):
    world = env.world
    assert len(world.agents) == len(world.landmarks) == 3
    assert world.dt == .1 and world.damping == .25
    assert world.contact_force == 100 and world.contact_margin == .001
    assert not world.cache_dists and not world.walls
    assert not env.discrete_action_input and not env.force_discrete_action
    assert env.post_step_callback is None
    for agent in world.agents:
        assert agent.mass == 1 and agent.size == .15 and agent.max_speed is None
        assert agent.accel is None and not agent.u_noise and not agent.c_noise
        assert agent.silent and agent.collide and agent.action_callback is None
    assert all(not landmark.collide and not landmark.movable for landmark in world.landmarks)
    assert [space.shape for space in env.observation_space] == [(18,), (18,), (18,)]
    observed_controls = []
    for action in range(5):
        env._set_action(np.eye(5)[action], world.agents[0], env.action_space[0])
        observed_controls.append(world.agents[0].action.u.copy())
    np.testing.assert_array_equal(observed_controls, [[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]])
    return {"agent_mass": 1, "agent_radius": .15, "max_speed": None, "accel_attribute": None,
            "action_controls_after_environment_scaling": np.asarray(observed_controls).tolist(),
            "dt": .1, "damping": .25, "contact_force": 100, "contact_margin": .001,
            "noise": False, "walls": False, "distance_cache": False}


def branch_state(env, observations, base_actions):
    """World access is validation-only; predictors below receive one obs row."""
    snapshot = snapshot_world(env)
    initial = np.asarray([agent.state.p_pos for agent in env.world.agents]).copy()
    landmarks = np.asarray([landmark.state.p_pos for landmark in env.world.landmarks]).copy()
    velocities = np.asarray([agent.state.p_vel for agent in env.world.agents]).copy()
    cases, expected_base_next = [], None
    for focal in range(3):
        raw = np.asarray(observations[focal], dtype=np.float32)
        order = [focal] + [index for index in range(3) if index != focal]
        predicted = np.stack([predict_local_candidate_positions(raw[None], mode)[0] for mode in MODES])
        actual = []
        for action in range(5):
            restore_world(env, snapshot)
            actions = [value.copy() for value in base_actions]
            actions[focal] = np.eye(5)[action]
            env.step(actions)
            positions = np.asarray([agent.state.p_pos for agent in env.world.agents]).copy()
            actual.append(positions[order] - initial[focal])
            if np.array_equal(base_actions[focal], np.eye(5)[action]):
                if expected_base_next is not None:
                    np.testing.assert_allclose(positions, expected_base_next, rtol=0, atol=1e-12)
                expected_base_next = positions
        cases.append({"raw": raw, "focal": focal, "predicted": predicted, "actual": np.asarray(actual),
                      "landmarks": landmarks - initial[focal],
                      "validation_only_velocities": velocities[order],
                      "validation_only_base_actions": np.asarray(base_actions)[order]})
    restore_world(env, snapshot)
    return cases, expected_base_next


def quantities(agent_positions, landmarks):
    return np.asarray([_state_quantities(positions, landmarks) for positions in agent_positions])


def collect(checkpoints, episodes, seed_base):
    data = {key: [] for key in ("raw", "focal", "predicted", "actual", "landmarks",
                                "validation_only_velocities", "validation_only_base_actions",
                                "source", "episode", "step")}
    for source_index, (name, checkpoint) in enumerate(checkpoints.items()):
        model = MADDPG.init_from_save(checkpoint)
        model.prep_rollouts(device="cpu")
        before = policy_digest(model.policies)
        env = make_env("simple_spread", discrete_action=True)
        assert_environment_supported(env)
        try:
            for episode in range(episodes):
                seed = seed_base + episode
                torch.manual_seed(seed)
                np.random.seed(seed)
                env.seed(seed)
                obs = env.reset()
                for step in range(25):
                    inputs = [torch.as_tensor(value, dtype=torch.float32).view(1, 18) for value in obs]
                    with torch.no_grad():
                        actions = [value.cpu().numpy().ravel() for value in model.step(inputs, explore=False)]
                    if any(np.count_nonzero(action) != 1 for action in actions):
                        raise RuntimeError("A tied policy action is not a single candidate; stop and audit it")
                    expected = None
                    if step % 5 == 0:
                        cases, expected = branch_state(env, obs, actions)
                        for case in cases:
                            for key, value in case.items():
                                data[key].append(value)
                            data["source"].append(source_index)
                            data["episode"].append(episode)
                            data["step"].append(step)
                    obs, _, _, _ = env.step(actions)
                    if expected is not None:
                        actual = np.asarray([agent.state.p_pos for agent in env.world.agents])
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                if (episode + 1) % 25 == 0:
                    print(f"{name}: {episode + 1}/{episodes} trajectories", flush=True)
        finally:
            env.close()
        assert policy_digest(model.policies) == before
    return {key: np.asarray(value) for key, value in data.items()}


def analyze(data, source_names, output):
    predicted, actual = data["predicted"], data["actual"]
    n = len(actual)
    predicted_features = np.empty((n, 3, 5, 4))
    true_features = np.empty((n, 5, 4))
    for row in range(n):
        q = quantities(actual[row], data["landmarks"][row])
        true_features[row] = q - q[:1]
        for mode in range(3):
            q = quantities(predicted[row, mode], data["landmarks"][row])
            predicted_features[row, mode] = q - q[:1]
    legacy_features = compute_active_gsp_v3_from_local_obs_batch(data["raw"])
    np.testing.assert_allclose(predicted_features[:, 0], legacy_features, rtol=2e-3, atol=2e-6)
    assert np.isfinite(predicted_features).all() and np.isfinite(true_features).all()
    assert not predicted_features[:, :, 0].any() and not true_features[:, 0].any()
    separation = np.linalg.norm(data["raw"][:, 10:14].reshape(-1, 2, 2), axis=2).min(1)
    self_error = np.linalg.norm(predicted[:, :, :, 0] - actual[:, None, :, 0], axis=-1)
    predicted_delta = predicted[:, :, :, 0] - predicted[:, :, :1, 0]
    actual_delta = actual[:, :, 0] - actual[:, :1, 0]
    relative_error = np.linalg.norm(predicted_delta - actual_delta[:, None], axis=-1)
    graph_error = np.linalg.norm(predicted - actual[:, None], axis=-1).mean(-1)
    rows, feature_rows = [], []
    groups = [("all", np.ones(n, dtype=bool))] + [(name, data["source"] == index) for index, name in enumerate(source_names)]
    bins = [("all", 0, np.inf)] + list(SEPARATION_BINS)
    for source, source_mask in groups:
        for name, low, high in bins:
            selected = source_mask & (separation >= low) & (separation < high)
            if not selected.any():
                continue
            for index, mode in enumerate(MODES):
                error = self_error[selected, index].ravel()
                rows.append({"source": source, "separation_bin": name, "mode": mode,
                             "focal_cases": int(selected.sum()), "candidate_branches": int(selected.sum() * 5),
                             "self_error_mean": float(error.mean()), "self_error_p95": float(np.quantile(error, .95)),
                             "self_error_max": float(error.max()),
                             "action_relative_self_error_max": float(relative_error[selected, index].max()),
                             "whole_graph_node_error_mean": float(graph_error[selected, index].mean())})
                for feature, feature_name in enumerate(FEATURE_NAMES):
                    target = true_features[selected, 1:, feature]
                    estimated = predicted_features[selected, index, 1:, feature]
                    nontrivial = np.abs(target) >= 1e-5
                    error = estimated - target
                    feature_rows.append({"source": source, "separation_bin": name, "mode": mode,
                                         "feature": feature_name, "mae": float(np.abs(error).mean()),
                                         "rmse": float(np.sqrt(np.square(error).mean())),
                                         "target_rms": float(np.sqrt(np.square(target).mean())),
                                         "nontrivial_deltas": int(nontrivial.sum()),
                                         "sign_agreement": float((np.sign(estimated[nontrivial]) == np.sign(target[nontrivial])).mean()) if nontrivial.any() else None})
    write_csv(output / "motion_errors.csv", rows)
    write_csv(output / "feature_errors.csv", feature_rows)
    np.savez_compressed(output / "diagnostic_dataset.npz", **data, predicted_features=predicted_features,
                        actual_feature_deltas=true_features, min_focal_separation=separation)
    return {"focal_cases": n, "candidate_branches": n * 5, "contact_focal_cases": int((separation < .3).sum()),
            "max_contact_corrected_self_error": float(self_error[:, 1:].max()),
            "max_legacy_action_relative_self_error": float(relative_error[:, 0].max()),
            "legacy_feature_implementation_equivalence": True, "no_op_feature_deltas_zero": True,
            "all_metrics_finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    options = parser.parse_args()
    if options.episodes <= 0:
        parser.error("--episodes must be positive")
    root, output = options.run_dir.resolve(), options.output_dir.resolve()
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["source_hashes"] == source_hashes()
    checkpoints = {name: root / name / "seed_1/checkpoints/model_final_100000.pt"
                   for name in ("d4_geometric_control", "d4_dirichlet_gsp")}
    archived = json.loads((REPO_ROOT / "experiments/corrected_active_gsp_multiseed_20260711_124239/sources.json").read_text())
    checkpoints["archived_raw"] = REPO_ROOT / archived["1"]["raw_mlp"] / "checkpoints/model_final_100000.pt"
    hashes = {name: sha256(path) for name, path in checkpoints.items()}
    validation_env = make_env("simple_spread", discrete_action=True)
    try:
        environment_spec = assert_environment_supported(validation_env)
    finally:
        validation_env.close()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "new_training_steps": 0, "seed_base": 9100000,
                "source_training_seeds": [1], "episodes_per_source": options.episodes,
                "horizon": 25, "sample_every_steps": 5, "actions_per_focal": 5,
                "sources": {key: str(value) for key, value in checkpoints.items()}, "checkpoint_hashes": hashes,
                "prediction_input": "one raw local 18D observation only; no hidden velocity/action/world input",
                "device": "CPU: small geometry calculations and unchanged MPE validation, no parameter updates",
                "environment_spec": environment_spec,
                "diagnostic_source_hashes": {name: sha256(REPO_ROOT / name) for name in (
                    "utils/local_contact_motion.py", "experiments/diagnose_local_contact_motion.py",
                    "tests/test_local_contact_motion.py", "experiments/probe_action_value_representations.py",
                    "utils/active_gsp_v3_features.py")},
                "modes": MODES, "interpretation": "motion/feature fidelity, not reward or learning improvement"}
    write_json(output / "protocol.json", manifest)
    torch.set_num_threads(1)
    start = time.perf_counter()
    try:
        data = collect(checkpoints, options.episodes, 9100000)
        summary = analyze(data, list(checkpoints), output)
        assert hashes == {name: sha256(path) for name, path in checkpoints.items()}
        assert status["source_hashes"] == source_hashes()
        assert all(sha256(REPO_ROOT / name) == digest for name, digest in manifest["diagnostic_source_hashes"].items())
        manifest.update(status="completed", summary=summary, elapsed_seconds=time.perf_counter() - start,
                        trajectory_environment_transitions=len(checkpoints) * options.episodes * 25,
                        diagnostic_branch_environment_transitions=summary["candidate_branches"],
                        checkpoint_and_training_sources_unchanged=True,
                        dataset_sha256=sha256(output / "diagnostic_dataset.npz"))
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(output / "protocol.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
