"""Fresh five-seed confirmation of the first-update-then-freeze control."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol


PAYLOAD = "experiments/hybrid_actor_dagger_pilot_20260713/hybrid_distilled_actor.pt"
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
        **BASE,
        "short": "fu_frz",
        "actor_update_start_step": 1_000_000_000,
    },
    "hybrid_first_update_then_freeze": {
        **BASE,
        "short": "fu_one",
        # The only update event occurs at transition 100. The half-open end
        # would disable every later event if training continued.
        "actor_update_end_step": 200,
    },
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/hybrid_first_update_confirmation_20260716",
        "--total-env-steps": "100",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "100",
        "--seeds": "60,61,62,63,64",
        "--checkpoint-steps": "64,100",
        "--methods": ",".join(METHODS),
        "--eval-seed-base": "8000000",
        "--curve-eval-seed-base": "8100000",
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
    protocol.STATUS_TITLE = "Hybrid First-Update Confirmation"
    protocol.PLOT_TITLE_PREFIX = "First-update-then-freeze confirmation"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
