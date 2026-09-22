"""Fresh-seed 20k confirmation for Stage-A optimizer-control survivors."""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol
from experiments.run_hybrid_optimizer_controls import METHODS as STAGE_A_METHODS


DECISIONS = (
    "experiments/hybrid_optimizer_controls_screen_20260716/"
    "run_20260716_211049/matched_stage_a_analysis/stage_a_decisions.csv"
)
DIAGNOSTIC_CONTROLS = {
    "hybrid_frozen",
    "hybrid_unanchored",
    "hybrid_old_policy_kl002",
}


def selected_methods():
    with Path(DECISIONS).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = {
        row["method"]
        for row in rows
        if str(row["eligible"]).strip().lower() == "true"
    }
    selected = {
        method: spec
        for method, spec in STAGE_A_METHODS.items()
        if method in eligible or method in DIAGNOSTIC_CONTROLS
    }
    if "hybrid_frozen" not in selected:
        raise ValueError("Frozen control must be present")
    if not (eligible - {"hybrid_frozen"}):
        raise ValueError("No non-frozen Stage-A method is eligible")
    return selected


def add_defaults(argv, methods):
    defaults = {
        "--output-dir": "experiments/hybrid_optimizer_controls_stage_b_20260716",
        "--total-env-steps": "20000",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "100",
        "--seeds": "52,53,54",
        "--checkpoint-steps": "64,100,1000,4996,5000,10000,20000",
        "--methods": ",".join(methods),
        "--eval-seed-base": "5400000",
        "--curve-eval-seed-base": "5500000",
        "--policy-audit-batch-size": "100",
    }
    present = {item.split("=", 1)[0] for item in argv}
    output = list(argv)
    for flag, value in defaults.items():
        if flag not in present:
            output += [flag, value]
    return output


def main():
    methods = selected_methods()
    protocol.METHODS = methods
    protocol.METRICS = vector_protocol.METRICS
    protocol.evaluate_model = vector_protocol.evaluate_model
    protocol.STATUS_TITLE = "Hybrid Optimizer Controls Stage B"
    protocol.PLOT_TITLE_PREFIX = "Fresh-seed optimizer-control confirmation"
    print("Stage-B methods:", ",".join(methods), flush=True)
    sys.argv = add_defaults(sys.argv, methods)
    protocol.main()


if __name__ == "__main__":
    main()
