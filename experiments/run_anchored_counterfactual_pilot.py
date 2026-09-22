"""Locked pilot for fixed-teacher KL trust-region actor updates.

The trust-region radius is fixed at KL=0.002 before looking at results.  By
Pinsker's inequality this bounds empirical categorical-policy total variation
by sqrt(0.002 / 2) ~= 3.2 percent on each actor update minibatch.  Environment,
reward, replay, critic inputs, actions, and decentralized execution are reused
unchanged from the locked protocol runner.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol


PAYLOAD = "experiments/action_value_representation_probe_20260712/active_hungarian_probe_seed1.pt"
def anchor(max_kl):
    return {
        "mode": "fixed_teacher_kl",
        "max_kl": max_kl,
        "temperature": 1.0,
        "bisection_steps": 16,
    }


ANCHOR = anchor(0.002)
METHODS = {
    "counterfactual_pretrained_unanchored": {
        "env_id": "simple_spread",
        "actor_model": "counterfactual_active_probe",
        "short": "cfpre",
        "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD,
        "pretrained_load_weights": True,
    },
    "counterfactual_pretrained_kl002": {
        "env_id": "simple_spread",
        "actor_model": "counterfactual_active_probe",
        "short": "cfkl002",
        "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD,
        "pretrained_load_weights": True,
        "actor_anchor": ANCHOR,
    },
    "counterfactual_pretrained_kl1e4": {
        "env_id": "simple_spread", "actor_model": "counterfactual_active_probe",
        "short": "cfkl1e4", "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
        "actor_anchor": anchor(1e-4),
    },
    "counterfactual_pretrained_kl1e5": {
        "env_id": "simple_spread", "actor_model": "counterfactual_active_probe",
        "short": "cfkl1e5", "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
        "actor_anchor": anchor(1e-5),
    },
    "counterfactual_pretrained_kl1e6": {
        "env_id": "simple_spread", "actor_model": "counterfactual_active_probe",
        "short": "cfkl1e6", "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
        "actor_anchor": anchor(1e-6),
    },
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/anchored_counterfactual_pilot",
        "--total-env-steps": "1000",
        "--eval-episodes": "200",
        "--curve-eval-episodes": "200",
        "--seeds": "31,32,33",
        "--checkpoint-steps": "4,64,100,500,1000",
        "--methods": ",".join(METHODS),
        "--eval-seed-base": "2000000",
        "--curve-eval-seed-base": "2100000",
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
    protocol.STATUS_TITLE = "Anchored Counterfactual Pilot Status"
    protocol.PLOT_TITLE_PREFIX = "Fixed-teacher KL=0.002 pilot"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
