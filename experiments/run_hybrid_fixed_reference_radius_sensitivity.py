"""Locked five-seed sensitivity study for the fixed-reference KL radius."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_hybrid_optimizer_controls as base
from experiments import run_passive_gsp_topology_pilot as protocol


def projection(max_kl):
    return {
        "mode": "fixed_teacher_kl",
        "max_kl": float(max_kl),
        "temperature": 1.0,
        "bisection_steps": 16,
    }


METHODS = {
    "hybrid_frozen": dict(base.METHODS["hybrid_frozen"]),
    "hybrid_fixed_teacher_kl001": {
        **base.BASE,
        "short": "hybe1",
        "actor_lr": 0.01,
        "actor_anchor": projection(0.001),
    },
    "hybrid_fixed_teacher_kl002": {
        **base.BASE,
        "short": "hybe2",
        "actor_lr": 0.01,
        "actor_anchor": projection(0.002),
    },
    "hybrid_fixed_teacher_kl005": {
        **base.BASE,
        "short": "hybe5",
        "actor_lr": 0.01,
        "actor_anchor": projection(0.005),
    },
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/hybrid_fixed_reference_radius_sensitivity_20260811",
        "--total-env-steps": "20000",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "100",
        "--seeds": "65,66,67,68,69",
        "--checkpoint-steps": "64,100,1000,4996,5000,10000,20000",
        "--methods": ",".join(METHODS),
        "--eval-seed-base": "8100000",
        "--curve-eval-seed-base": "8200000",
        "--policy-audit-batch-size": "100",
    }
    present = {item.split("=", 1)[0] for item in argv}
    output = list(argv)
    for flag, value in defaults.items():
        if flag not in present:
            output += [flag, value]
    return output


def main():
    if not Path(base.PAYLOAD).exists():
        raise FileNotFoundError(base.PAYLOAD)
    protocol.METHODS = METHODS
    protocol.METRICS = vector_protocol.METRICS
    protocol.evaluate_model = vector_protocol.evaluate_model
    protocol.STATUS_TITLE = "Hybrid Fixed-Reference Radius Sensitivity"
    protocol.PLOT_TITLE_PREFIX = "Fixed-reference KL radius sensitivity"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
