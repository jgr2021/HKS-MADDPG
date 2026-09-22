import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.run_passive_gsp_topology_pilot as protocol


METHODS = {
    "raw_mlp": {
        "env_id": "simple_spread",
        "actor_model": "mlp",
        "short": "raw",
        "actor_input_dim": 18,
    },
    "local_geometry_residual": {
        "env_id": "simple_spread",
        "actor_model": "local_geometry_residual",
        "short": "lgeom",
        "actor_input_dim": 18,
    },
    "local_spectral_al_residual": {
        "env_id": "simple_spread",
        "actor_model": "local_spectral_al_residual",
        "short": "lspec",
        "actor_input_dim": 18,
    },
    "learned_raw_potential_residual": {
        "env_id": "simple_spread",
        "actor_model": "learned_raw_potential_residual",
        "short": "lraw",
        "actor_input_dim": 18,
    },
    "learned_active_gsp_residual": {
        "env_id": "simple_spread",
        "actor_model": "learned_active_gsp_residual",
        "short": "lagsp",
        "actor_input_dim": 18,
    },
    "learned_multiscale_spectral_residual": {
        "env_id": "simple_spread",
        "actor_model": "learned_multiscale_spectral_residual",
        "short": "lmsgsp",
        "actor_input_dim": 18,
    },
    "learned_crowding_gsp_residual": {
        "env_id": "simple_spread",
        "actor_model": "learned_crowding_gsp_residual",
        "short": "lcgsp",
        "actor_input_dim": 18,
    },
    "learned_rwse_potential_control": {
        "env_id": "simple_spread",
        "actor_model": "learned_rwse_potential_control",
        "short": "lrwsec",
        "actor_input_dim": 18,
    },
    "learned_active_rwse_residual": {
        "env_id": "simple_spread",
        "actor_model": "learned_active_rwse_residual",
        "short": "lrwse",
        "actor_input_dim": 18,
    },
    "learned_scf_potential_control": {
        "env_id": "simple_spread",
        "actor_model": "learned_scf_potential_control",
        "short": "lscfc",
        "actor_input_dim": 18,
    },
    "learned_active_scf_residual": {
        "env_id": "simple_spread",
        "actor_model": "learned_active_scf_residual",
        "short": "lscf",
        "actor_input_dim": 18,
    },
    "learned_gated_dirichlet_control": {
        "env_id": "simple_spread",
        "actor_model": "learned_gated_dirichlet_control",
        "short": "lgdc",
        "actor_input_dim": 18,
    },
    "learned_gated_dirichlet_gsp": {
        "env_id": "simple_spread",
        "actor_model": "learned_gated_dirichlet_gsp",
        "short": "lgdgsp",
        "actor_input_dim": 18,
    },
    "learned_gated_dirichlet_gsp025": {
        "env_id": "simple_spread",
        "actor_model": "learned_gated_dirichlet_gsp025",
        "short": "lgdgsp025",
        "actor_input_dim": 18,
    },
    "learned_matching_control": {
        "env_id": "simple_spread",
        "actor_model": "learned_matching_control",
        "short": "lmatchc",
        "actor_input_dim": 18,
    },
    "learned_spectral_matching_residual": {
        "env_id": "simple_spread",
        "actor_model": "learned_spectral_matching_residual",
        "short": "lsmatch",
        "actor_input_dim": 18,
    },
    "linear_matching_control": {
        "env_id": "simple_spread",
        "actor_model": "linear_matching_control",
        "short": "linmatchc",
        "actor_input_dim": 18,
    },
    "linear_spectral_matching": {
        "env_id": "simple_spread",
        "actor_model": "linear_spectral_matching",
        "short": "linsmatch",
        "actor_input_dim": 18,
    },
    "matching_only_control": {
        "env_id": "simple_spread",
        "actor_model": "matching_only_control",
        "short": "matchonlyc",
        "actor_input_dim": 18,
    },
    "spectral_matching_only": {
        "env_id": "simple_spread",
        "actor_model": "spectral_matching_only",
        "short": "smatchonly",
        "actor_input_dim": 18,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/local_spectral_actor_pilot")
    parser.add_argument("--total-env-steps", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--curve-eval-episodes", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--methods", default=",".join(METHODS.keys()))
    parser.add_argument("--n-rollout-threads", type=int, default=4)
    parser.add_argument("--n-training-threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-per-update", type=int, default=100)
    parser.add_argument("--checkpoint-steps", default="20000")
    parser.add_argument("--print-interval", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval-seed-base", type=int, default=990000)
    parser.add_argument("--curve-eval-seed-base", type=int, default=770000)
    args = parser.parse_args()
    args.checkpoint_steps = [
        int(step.strip()) for step in args.checkpoint_steps.split(",") if step.strip()
    ]

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")

    protocol.METHODS = METHODS
    protocol.STATUS_TITLE = "Local Graph Actor Pilot Status"
    protocol.PLOT_TITLE_PREFIX = "Local graph actor pilot"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.output_dir) / f"run_{stamp}"
    root_dir.mkdir(parents=True, exist_ok=False)
    protocol.save_versions(root_dir)
    protocol.write_json(root_dir / "runner_config.json", vars(args))
    (root_dir / "launch_command.txt").write_text(
        subprocess.list2cmdline([sys.executable] + sys.argv) + "\n",
        encoding="utf-8",
    )
    protocol.write_json(
        root_dir / "protocol_metadata.json",
        {
            "protocol": "actor_only_local_spectral_graph_raw_critic",
            "methods": methods,
            "seeds": seeds,
            "total_env_steps": args.total_env_steps,
            "checkpoint_steps": args.checkpoint_steps,
            "eval_episodes_final": args.eval_episodes,
            "eval_episodes_checkpoint": args.curve_eval_episodes,
            "evaluation_exploration_enabled": False,
            "environment_observation_dim": 18,
            "actor_raw_tensor_dim": 18,
            "critic_input_dim": 69,
            "target_critic_input_dim": 69,
            "replay_obs_dims": [18, 18, 18],
            "method_definitions": {
                "raw_mlp": "unchanged raw 18D MLP actor",
                "local_geometry_residual": "relative-node geometry residual without diffusion",
                "local_spectral_al_residual": (
                    "relative 6-node AL graph; polynomial powers [0,1,2,4] of S=I-L_sym"
                ),
                "learned_raw_potential_residual": (
                    "learned candidate-action residual using non-spectral potential deltas"
                ),
                "learned_active_gsp_residual": (
                    "learned candidate-action residual using potential and Dirichlet-energy deltas"
                ),
                "learned_multiscale_spectral_residual": (
                    "learned residual using potentials and normalized heat-Rayleigh deltas at t=0,0.5,2"
                ),
                "learned_crowding_gsp_residual": (
                    "learned candidate-action residual using potentials and crowding Dirichlet energy"
                ),
                "learned_rwse_potential_control": (
                    "five-channel learned residual with three RWSE channels masked"
                ),
                "learned_active_rwse_residual": (
                    "learned residual using potentials and focal return deltas at P^2,P^4,P^6"
                ),
                "learned_scf_potential_control": (
                    "six-channel learned residual with directional graph-filter channels masked"
                ),
                "learned_active_scf_residual": (
                    "learned residual using potentials and focal S/S^2 coordinate responses"
                ),
                "learned_gated_dirichlet_control": (
                    "separated potential residual with two zeroed adaptive energy channels"
                ),
                "learned_gated_dirichlet_gsp": (
                    "separated potential residual with bounded state-action adaptive signed Dirichlet gates"
                ),
                "learned_gated_dirichlet_gsp025": (
                    "same adaptive Dirichlet gate with spectral logit correction scale 0.25"
                ),
                "learned_matching_control": (
                    "learned Sinkhorn assignment-cost residual with matching spectrum masked"
                ),
                "learned_spectral_matching_residual": (
                    "learned Sinkhorn assignment residual with entropy and sigma2/sigma3 spectrum"
                ),
                "linear_matching_control": (
                    "zero-initialized learned linear assignment-cost readout with spectrum masked"
                ),
                "linear_spectral_matching": (
                    "zero-initialized six-weight linear assignment and matching-spectrum readout"
                ),
                "matching_only_control": (
                    "minimal learned assignment-feature actor with matching spectrum masked"
                ),
                "spectral_matching_only": (
                    "minimal six-weight actor over assignment costs and matching spectrum"
                ),
            },
            "graph_coordinates": "focal-agent-relative",
            "action_conditioned_residual": (
                "learned only; no fixed feature weights or handcrafted action prior"
            ),
            "environment_unchanged": True,
            "reward_unchanged": True,
            "action_space_unchanged": True,
            "critic_unchanged": True,
            "replay_buffer_unchanged": True,
            "explicit_communication": False,
            "handcrafted_action_prior": False,
            "global_state_leakage": False,
            "critic_side_graph_features": False,
        },
    )

    seed_summaries = []
    curve_rows = []
    for method in methods:
        for seed in seeds:
            try:
                summary, learning_curve = protocol.train_one(
                    method, seed, args, root_dir, stamp
                )
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
            protocol.write_status(root_dir, seed_summaries)
            protocol.write_json(root_dir / "seed_summaries.json", seed_summaries)
            protocol.write_csv(root_dir / "seed_summaries.csv", seed_summaries)
            protocol.write_csv(root_dir / "learning_curve_eval.csv", curve_rows)

    aggregate_rows = protocol.aggregate(seed_summaries)
    protocol.write_csv(root_dir / "aggregate_summary.csv", aggregate_rows)
    protocol.write_json(root_dir / "aggregate_summary.json", aggregate_rows)
    protocol.plot_learning_curves(curve_rows, root_dir, args.checkpoint_steps)
    protocol.write_status(root_dir, seed_summaries)
    print(root_dir)


if __name__ == "__main__":
    main()
