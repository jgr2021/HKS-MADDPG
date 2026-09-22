"""Run frozen, unanchored, and fixed-reference updates for one new student."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_vector_signal_gsp_experiment as vector_protocol
from experiments import run_passive_gsp_topology_pilot as protocol
from experiments.run_hybrid_optimizer_controls import fixed_teacher_projection


REPLICATES = {
    "fold21_t101_s101": {
        "seed": 201,
        "eval_seed_base": 6_400_000,
        "curve_eval_seed_base": 6_500_000,
        "artifact_date": "20260716",
    },
    "fold22_t102_s102": {
        "seed": 202,
        "eval_seed_base": 6_600_000,
        "curve_eval_seed_base": 6_700_000,
        "artifact_date": "20260716",
    },
    "fold23_t103_s103": {
        "seed": 203,
        "eval_seed_base": 6_800_000,
        "curve_eval_seed_base": 6_900_000,
        "artifact_date": "20260716",
    },
    "fold24_t104_s104": {
        "seed": 204,
        "eval_seed_base": 46_000_000,
        "curve_eval_seed_base": 46_100_000,
        "artifact_date": "20260811",
    },
    "fold25_t105_s105": {
        "seed": 205,
        "eval_seed_base": 46_200_000,
        "curve_eval_seed_base": 46_300_000,
        "artifact_date": "20260811",
    },
}


def payload_path(replicate):
    artifact_date = REPLICATES[replicate]["artifact_date"]
    return (
        REPO_ROOT
        / "experiments"
        / f"independent_hybrid_distillation_{replicate}_{artifact_date}"
        / "hybrid_distilled_actor.pt"
    )


def methods_for(replicate):
    payload = str(payload_path(replicate))
    suffix = replicate[4:6]
    base = {
        "env_id": "simple_spread",
        "actor_model": "counterfactual_active_probe",
        "actor_input_dim": 18,
        "pretrained_policy_path": payload,
        "pretrained_load_weights": True,
        "critic_lr": 0.01,
    }
    return {
        "hybrid_frozen": {
            **base,
            "short": f"ifrz{suffix}",
            "actor_lr": 0.01,
            "actor_update_start_step": 1_000_000_000,
        },
        "hybrid_unanchored": {
            **base,
            "short": f"iun{suffix}",
            "actor_lr": 0.01,
        },
        "hybrid_fixed_teacher_kl002": {
            **base,
            "short": f"ifix{suffix}",
            "actor_lr": 0.01,
            "actor_anchor": fixed_teacher_projection(),
        },
    }


def add_defaults(argv, replicate, methods):
    config = REPLICATES[replicate]
    defaults = {
        "--output-dir": (
            "experiments/independent_hybrid_online_"
            f"{replicate}_{config['artifact_date']}"
        ),
        "--total-env-steps": "20000",
        "--eval-episodes": "500",
        "--curve-eval-episodes": "100",
        "--seeds": str(config["seed"]),
        "--checkpoint-steps": "64,100,1000,5000,10000,20000",
        "--methods": ",".join(methods),
        "--eval-seed-base": str(config["eval_seed_base"]),
        "--curve-eval-seed-base": str(config["curve_eval_seed_base"]),
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
    parser.add_argument("--replicate", required=True, choices=tuple(REPLICATES))
    known, remaining = parser.parse_known_args()
    replicate = known.replicate
    payload = payload_path(replicate)
    if not payload.exists():
        raise FileNotFoundError(payload)

    methods = methods_for(replicate)
    protocol.METHODS = methods
    protocol.METRICS = vector_protocol.METRICS
    protocol.evaluate_model = vector_protocol.evaluate_model
    protocol.STATUS_TITLE = (
        f"Independent Hybrid Online Status: {replicate}"
    )
    protocol.PLOT_TITLE_PREFIX = (
        f"Independent distilled actor online replication: {replicate}"
    )
    sys.argv = [sys.argv[0], *add_defaults(remaining, replicate, methods)]
    protocol.main()


if __name__ == "__main__":
    main()
