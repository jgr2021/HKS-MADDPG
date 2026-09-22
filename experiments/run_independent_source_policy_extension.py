"""Train new source policies for counterfactual data folds 24 and 25."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MADDPG_FORCE_CPU", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol


METHODS = {
    "action_aware_geometric_control": vector_protocol.METHODS[
        "action_aware_geometric_control"
    ]
}


def add_defaults(argv):
    defaults = {
        "--output-dir": "experiments/independent_source_policy_extension_20260811d",
        "--total-env-steps": "100000",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "100",
        "--seeds": "24,25",
        "--checkpoint-steps": "20000,40000,60000,80000,100000",
        "--methods": "action_aware_geometric_control",
        "--eval-seed-base": "44000000",
        "--curve-eval-seed-base": "44100000",
        "--buffer-length": "200000",
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
    protocol.STATUS_TITLE = "Independent Source Policy Extension"
    protocol.PLOT_TITLE_PREFIX = "New source-policy folds 24 and 25"
    sys.argv = add_defaults(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
