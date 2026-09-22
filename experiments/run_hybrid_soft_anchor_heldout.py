"""Held-out 100k test for development-selected soft anchors."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol
from experiments.run_hybrid_soft_anchor_dev import kl_penalty, l2_penalty


PAYLOAD = "experiments/hybrid_actor_dagger_pilot_20260713/hybrid_distilled_actor.pt"


def methods_for(selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection["protocol_pass"] or selection["heldout_outcomes_viewed"]:
        raise RuntimeError("Development selection is not eligible for held-out use")
    kl = selection["selected"]["kl_loss"]
    l2 = selection["selected"]["l2"]
    base = {
        "env_id": "simple_spread",
        "actor_model": "counterfactual_active_probe",
        "actor_input_dim": 18,
        "pretrained_policy_path": PAYLOAD,
        "pretrained_load_weights": True,
        "actor_lr": 0.01,
        "critic_lr": 0.01,
    }
    return {
        "hybrid_soft_kl_selected": {
            **base,
            "short": "hldk",
            "actor_anchor": kl_penalty(kl["coefficient"]),
            "development_source_method": kl["method"],
            "development_failed": kl["development_failed"],
        },
        "hybrid_soft_l2_selected": {
            **base,
            "short": "hldl",
            "actor_anchor": l2_penalty(l2["coefficient"]),
            "development_source_method": l2["method"],
            "development_failed": l2["development_failed"],
        },
    }


def add_defaults(argv, methods):
    defaults = {
        "--output-dir": "experiments/hybrid_soft_anchor_heldout_20260811",
        "--total-env-steps": "100000",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "100",
        "--seeds": "55,56,57,58,59",
        "--checkpoint-steps": "64,100,1000,20000,40000,60000,80000,100000",
        "--methods": ",".join(methods),
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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    methods = methods_for(known.selection)
    protocol.METHODS = methods
    protocol.METRICS = vector_protocol.METRICS
    protocol.evaluate_model = vector_protocol.evaluate_model
    protocol.STATUS_TITLE = "Held-Out Soft-Anchor 100k Status"
    protocol.PLOT_TITLE_PREFIX = "Held-out development-selected soft anchors"
    sys.argv = [sys.argv[0], *add_defaults(remaining, methods)]
    protocol.main()


if __name__ == "__main__":
    main()
