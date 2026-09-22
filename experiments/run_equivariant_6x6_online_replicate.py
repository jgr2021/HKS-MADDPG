"""Run the locked 5k online test for one 6x6 distilled payload."""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import run_passive_gsp_topology_pilot as protocol
from experiments.run_equivariant_6x6_online_stress import (
    METRICS,
    evaluate_model,
)
from experiments.run_hybrid_optimizer_controls import fixed_teacher_projection


SCREEN_DIR = Path("experiments/equivariant_6x6_sinkhorn_screen_20260716")
PAYLOADS = {
    1: (
        "experiments/equivariant_6x6_sinkhorn64_seed1_20260713/"
        "equivariant_6x6_actor.pt"
    ),
    2: (
        "experiments/equivariant_6x6_sinkhorn64_seed2_20260713/"
        "equivariant_6x6_actor.pt"
    ),
    3: (
        "experiments/equivariant_6x6_sinkhorn64_seed3_20260713/"
        "equivariant_6x6_actor.pt"
    ),
}
ONLINE = {
    1: {"seed": 72, "eval": 7_400_000, "curve": 7_500_000},
    2: {"seed": 73, "eval": 7_600_000, "curve": 7_700_000},
    3: {"seed": 74, "eval": 7_800_000, "curve": 7_900_000},
}
ACTOR_MODELS = {
    8: "equivariant_matching_safety_6x6_sinkhorn8",
    16: "equivariant_matching_safety_6x6_sinkhorn16",
    32: "equivariant_matching_safety_6x6_sinkhorn32",
    64: "equivariant_matching_safety_6x6",
}


def selected_iterations():
    metadata = json.loads(
        (SCREEN_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata["payload_ids"] != ["seed1", "seed2", "seed3"]:
        raise RuntimeError("The Sinkhorn screen must include all three payloads")
    if int(metadata["episodes"]) < 200:
        raise RuntimeError("The formal Sinkhorn screen requires 200 episodes")
    selected = int(metadata["selected_iterations"])
    with (SCREEN_DIR / "decisions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        decisions = list(csv.DictReader(handle))
    row = next(
        item
        for item in decisions
        if int(item["sinkhorn_iterations"]) == selected
    )
    if (
        row["eligible"].strip().lower() != "true"
        or int(row["eligible_payloads"]) != 3
        or int(row["total_payloads"]) != 3
    ):
        raise RuntimeError("Selected Sinkhorn iterations did not pass all payloads")
    return selected


def methods_for(payload_seed, actor_model):
    payload = PAYLOADS[payload_seed]
    base = {
        "env_id": "simple_spread_6x6",
        "actor_model": actor_model,
        "actor_input_dim": 36,
        "raw_observation_dim": 36,
        "n_agents": 6,
        "action_dim": 5,
        "pretrained_policy_path": payload,
        "pretrained_load_weights": True,
        "critic_lr": 0.01,
    }
    return {
        "equivariant_6x6_frozen": {
            **base,
            "short": f"e6f{payload_seed}",
            "actor_lr": 0.01,
            "actor_update_start_step": 1_000_000_000,
        },
        "equivariant_6x6_unanchored": {
            **base,
            "short": f"e6u{payload_seed}",
            "actor_lr": 0.01,
        },
        "equivariant_6x6_actor_lr1e4": {
            **base,
            "short": f"e6l{payload_seed}",
            "actor_lr": 1e-4,
        },
        "equivariant_6x6_fixed_teacher_kl002": {
            **base,
            "short": f"e6k{payload_seed}",
            "actor_lr": 0.01,
            "actor_anchor": fixed_teacher_projection(),
        },
    }


def add_defaults(argv, payload_seed, methods):
    config = ONLINE[payload_seed]
    defaults = {
        "--output-dir": (
            f"experiments/equivariant_6x6_online_payload"
            f"{payload_seed}_20260716"
        ),
        "--total-env-steps": "5000",
        "--eval-episodes": "200",
        "--curve-eval-episodes": "50",
        "--seeds": str(config["seed"]),
        "--checkpoint-steps": "64,100,1000,5000",
        "--methods": ",".join(methods),
        "--eval-seed-base": str(config["eval"]),
        "--curve-eval-seed-base": str(config["curve"]),
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
    parser.add_argument("--payload-seed", type=int, choices=(1, 2, 3), required=True)
    known, remaining = parser.parse_known_args()
    payload_seed = known.payload_seed
    if not Path(PAYLOADS[payload_seed]).exists():
        raise FileNotFoundError(PAYLOADS[payload_seed])
    iterations = selected_iterations()
    actor_model = ACTOR_MODELS[iterations]
    methods = methods_for(payload_seed, actor_model)

    protocol.METHODS = methods
    protocol.METRICS = METRICS
    protocol.evaluate_model = evaluate_model
    protocol.STATUS_TITLE = (
        f"Equivariant 6x6 Online Status: payload {payload_seed}"
    )
    protocol.PLOT_TITLE_PREFIX = (
        f"Equivariant 6x6, Sinkhorn {iterations}, payload {payload_seed}"
    )
    sys.argv = [
        sys.argv[0],
        *add_defaults(remaining, payload_seed, methods),
    ]
    protocol.main()


if __name__ == "__main__":
    main()
