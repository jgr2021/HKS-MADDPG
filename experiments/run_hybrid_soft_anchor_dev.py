"""Locked development sweep for fixed-reference KL-loss and parameter L2."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol


PAYLOAD = "experiments/hybrid_actor_dagger_pilot_20260713/hybrid_distilled_actor.pt"


def kl_penalty(coefficient):
    return {
        "mode": "fixed_teacher_kl_penalty",
        "coefficient": float(coefficient),
        "temperature": 1.0,
    }


def l2_penalty(coefficient):
    return {
        "mode": "fixed_parameter_l2_penalty",
        "coefficient": float(coefficient),
    }


BASE = {
    "env_id": "simple_spread",
    "actor_model": "counterfactual_active_probe",
    "actor_input_dim": 18,
    "pretrained_policy_path": PAYLOAD,
    "pretrained_load_weights": True,
    "actor_lr": 0.01,
    "critic_lr": 0.01,
}


METHODS = {
    "hybrid_frozen": {
        **BASE, "short": "sdevfrz",
        "actor_update_start_step": 1_000_000_000,
    },
    "hybrid_soft_kl_0p1": {
        **BASE, "short": "sdevk01", "actor_anchor": kl_penalty(0.1),
    },
    "hybrid_soft_kl_1": {
        **BASE, "short": "sdevk1", "actor_anchor": kl_penalty(1.0),
    },
    "hybrid_soft_kl_10": {
        **BASE, "short": "sdevk10", "actor_anchor": kl_penalty(10.0),
    },
    "hybrid_soft_l2_1": {
        **BASE, "short": "sdevl1", "actor_anchor": l2_penalty(1.0),
    },
    "hybrid_soft_l2_10": {
        **BASE, "short": "sdevl10", "actor_anchor": l2_penalty(10.0),
    },
    "hybrid_soft_l2_100": {
        **BASE, "short": "sdevl100", "actor_anchor": l2_penalty(100.0),
    },
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/hybrid_soft_anchor_dev_20260811",
        "--total-env-steps": "20000",
        "--eval-episodes": "200",
        "--curve-eval-episodes": "50",
        "--seeds": "75,76",
        "--checkpoint-steps": "64,100,1000,5000,10000,20000",
        "--methods": ",".join(METHODS),
        "--eval-seed-base": "42_000_000".replace("_", ""),
        "--curve-eval-seed-base": "42_100_000".replace("_", ""),
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
    protocol.STATUS_TITLE = "Hybrid Soft-Anchor Development Sweep"
    protocol.PLOT_TITLE_PREFIX = "Soft-anchor development only"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
