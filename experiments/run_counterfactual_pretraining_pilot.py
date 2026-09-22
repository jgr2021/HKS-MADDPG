"""Matched architecture-control pilot for counterfactual actor pretraining."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol


PAYLOAD = "experiments/action_value_representation_probe_20260712/active_hungarian_probe_seed1.pt"
METHODS = {
    "action_aware_geometric_control": {
        "env_id": "simple_spread", "actor_model": "learned_raw_potential_residual",
        "short": "geom", "actor_input_dim": 18,
    },
    "counterfactual_architecture_control": {
        "env_id": "simple_spread", "actor_model": "counterfactual_active_probe",
        "short": "cfctrl", "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": False,
    },
    "counterfactual_pretrained": {
        "env_id": "simple_spread", "actor_model": "counterfactual_active_probe",
        "short": "cfpre", "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
    },
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/counterfactual_pretraining_pilot",
        "--total-env-steps": "20000", "--eval-episodes": "500",
        "--curve-eval-episodes": "100", "--seeds": "31,32,33",
        "--checkpoint-steps": "5000,10000,15000,20000",
        "--methods": ",".join(METHODS),
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
    protocol.STATUS_TITLE = "Counterfactual Pretraining Pilot Status"
    protocol.PLOT_TITLE_PREFIX = "Counterfactual pretraining pilot"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
