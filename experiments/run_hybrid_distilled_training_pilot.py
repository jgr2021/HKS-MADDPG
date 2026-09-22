"""First-update stress test for the DAgger-distilled hybrid actor."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol


PAYLOAD = "experiments/hybrid_actor_dagger_pilot_20260713/hybrid_distilled_actor.pt"
METHODS = {
    "hybrid_distilled_unanchored": {
        "env_id": "simple_spread", "actor_model": "counterfactual_active_probe",
        "short": "hybd", "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
    },
    "hybrid_distilled_kl002": {
        "env_id": "simple_spread", "actor_model": "counterfactual_active_probe",
        "short": "hybdkl", "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
        "actor_anchor": {
            "mode": "fixed_teacher_kl", "max_kl": 0.002,
            "temperature": 1.0, "bisection_steps": 16,
        },
    },
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/hybrid_distilled_training_pilot",
        "--total-env-steps": "1000", "--eval-episodes": "200",
        "--curve-eval-episodes": "200", "--seeds": "41,42,43",
        "--checkpoint-steps": "4,64,100,500,1000",
        "--methods": ",".join(METHODS),
        "--eval-seed-base": "3500000", "--curve-eval-seed-base": "3600000",
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
    protocol.STATUS_TITLE = "Hybrid-Distilled Training Pilot Status"
    protocol.PLOT_TITLE_PREFIX = "Hybrid-distilled first-update stress test"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
