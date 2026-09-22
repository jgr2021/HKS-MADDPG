"""Five-seed 100k durability test for low actor LR and fixed reference."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol
from experiments.run_hybrid_optimizer_controls import METHODS as ALL_METHODS


METHOD_NAMES = (
    "hybrid_frozen",
    "hybrid_actor_lr1e4",
    "hybrid_fixed_teacher_kl002",
)
METHODS = {name: ALL_METHODS[name] for name in METHOD_NAMES}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/hybrid_optimizer_long_horizon_20260716",
        "--total-env-steps": "100000",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "100",
        "--seeds": "55,56,57,58,59",
        "--checkpoint-steps": "64,100,1000,20000,40000,60000,80000,100000",
        "--methods": ",".join(METHODS),
        "--eval-seed-base": "5600000",
        "--curve-eval-seed-base": "5700000",
        "--policy-audit-batch-size": "100",
    }
    present = {item.split("=", 1)[0] for item in argv}
    output = list(argv)
    for flag, value in defaults.items():
        if flag not in present:
            output += [flag, value]
    return output


def main():
    protocol.METHODS = METHODS
    protocol.METRICS = vector_protocol.METRICS
    protocol.evaluate_model = vector_protocol.evaluate_model
    protocol.STATUS_TITLE = "Hybrid Optimizer Long-Horizon Status"
    protocol.PLOT_TITLE_PREFIX = "Distilled actor 100k durability"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
