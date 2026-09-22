"""Locked optimizer-control study for the distilled 3x3 hybrid actor.

Stage A is a single fresh-seed screen.  It tests whether the previously
observed actor collapse can be explained or prevented by simpler optimizer
controls before committing to a multi-seed 100k comparison.

The environment, reward, replay, centralized critic, discrete actions,
distilled initialization, exploration schedule, and update frequency are
identical across arms.  Only actor-update handling differs.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol


PAYLOAD = "experiments/hybrid_actor_dagger_pilot_20260713/hybrid_distilled_actor.pt"


def fixed_teacher_projection():
    return {
        "mode": "fixed_teacher_kl",
        "max_kl": 0.002,
        "temperature": 1.0,
        "bisection_steps": 16,
    }


def old_policy_projection():
    return {
        "mode": "old_policy_kl",
        "max_kl": 0.002,
        "temperature": 1.0,
        "bisection_steps": 16,
    }


def fixed_teacher_penalty(coefficient):
    return {
        "mode": "fixed_teacher_kl_penalty",
        "coefficient": float(coefficient),
        "temperature": 1.0,
    }


BASE = {
    "env_id": "simple_spread",
    "actor_model": "counterfactual_active_probe",
    "actor_input_dim": 18,
    "pretrained_policy_path": PAYLOAD,
    "pretrained_load_weights": True,
    "critic_lr": 0.01,
}

METHODS = {
    "hybrid_frozen": {
        **BASE,
        "short": "hybfrz",
        "actor_lr": 0.01,
        "actor_update_start_step": 1_000_000_000,
    },
    "hybrid_unanchored": {
        **BASE,
        "short": "hybun",
        "actor_lr": 0.01,
    },
    "hybrid_actor_lr1e3": {
        **BASE,
        "short": "hyblr3",
        "actor_lr": 1e-3,
    },
    "hybrid_actor_lr1e4": {
        **BASE,
        "short": "hyblr4",
        "actor_lr": 1e-4,
    },
    "hybrid_delayed_actor5k": {
        **BASE,
        "short": "hybdly",
        "actor_lr": 0.01,
        "actor_update_start_step": 5000,
    },
    "hybrid_teacher_penalty1": {
        **BASE,
        "short": "hybpen1",
        "actor_lr": 0.01,
        "actor_anchor": fixed_teacher_penalty(1.0),
    },
    "hybrid_teacher_penalty10": {
        **BASE,
        "short": "hybpen10",
        "actor_lr": 0.01,
        "actor_anchor": fixed_teacher_penalty(10.0),
    },
    "hybrid_old_policy_kl002": {
        **BASE,
        "short": "hybold",
        "actor_lr": 0.01,
        "actor_anchor": old_policy_projection(),
    },
    "hybrid_fixed_teacher_kl002": {
        **BASE,
        "short": "hybfix",
        "actor_lr": 0.01,
        "actor_anchor": fixed_teacher_projection(),
    },
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/hybrid_optimizer_controls_screen",
        "--total-env-steps": "5000",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "200",
        "--seeds": "51",
        "--checkpoint-steps": "64,100,1000,4996,5000",
        "--methods": ",".join(METHODS),
        "--eval-seed-base": "5100000",
        "--curve-eval-seed-base": "5200000",
        "--policy-audit-batch-size": "100",
    }
    present = {item.split("=", 1)[0] for item in argv}
    output = list(argv)
    for flag, value in defaults.items():
        if flag not in present:
            output += [flag, value]
    return output


def main():
    if not Path(PAYLOAD).exists():
        raise FileNotFoundError(PAYLOAD)
    protocol.METHODS = METHODS
    protocol.METRICS = vector_protocol.METRICS
    protocol.evaluate_model = vector_protocol.evaluate_model
    protocol.STATUS_TITLE = "Hybrid Optimizer Controls Screen"
    protocol.PLOT_TITLE_PREFIX = "Distilled actor optimizer-control screen"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
