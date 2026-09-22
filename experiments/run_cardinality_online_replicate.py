"""Run the locked online arms for one eligible NxN distilled payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import run_passive_gsp_topology_pilot as protocol
from experiments.cardinality_coordination import (
    ONLINE_METRICS,
    env_id_for_cardinality,
    evaluate_model,
    observation_dim,
)
from experiments.run_hybrid_optimizer_controls import fixed_teacher_projection


DISTILL_ROOT = Path("experiments/cardinality_distillations_20260811")


def payload_path(n_agents, replicate):
    return DISTILL_ROOT / f"n{n_agents}_actor{replicate}" / "cardinality_actor.pt"


def methods_for(n_agents, replicate):
    payload = str(payload_path(n_agents, replicate))
    suffix = f"n{n_agents}a{replicate}"
    base = {
        "env_id": env_id_for_cardinality(n_agents),
        "actor_model": "equivariant_matching_safety_nxn_sinkhorn8",
        "actor_input_dim": observation_dim(n_agents),
        "raw_observation_dim": observation_dim(n_agents),
        "n_agents": n_agents,
        "action_dim": 5,
        "pretrained_policy_path": payload,
        "pretrained_load_weights": True,
        "actor_lr": 0.01,
        "critic_lr": 0.01,
    }
    return {
        "cardinality_frozen": {
            **base,
            "short": f"cfrz{suffix}",
            "actor_update_start_step": 1_000_000_000,
        },
        "cardinality_first_update_then_freeze": {
            **base,
            "short": f"cone{suffix}",
            "actor_update_end_step": 200,
        },
        "cardinality_unanchored": {
            **base,
            "short": f"cun{suffix}",
        },
        "cardinality_fixed_teacher_kl002": {
            **base,
            "short": f"cfix{suffix}",
            "actor_anchor": fixed_teacher_projection(),
        },
    }


def add_defaults(argv, n_agents, replicate, methods):
    online_seed = 4000 + n_agents * 10 + replicate
    offset = n_agents * 1_000_000 + replicate * 100_000
    defaults = {
        "--output-dir": (
            f"experiments/cardinality_online_n{n_agents}_actor"
            f"{replicate}_20260811"
        ),
        "--total-env-steps": "20000",
        "--eval-episodes": "300",
        "--curve-eval-episodes": "100",
        "--seeds": str(online_seed),
        "--checkpoint-steps": "64,100,1000,5000,10000,20000",
        "--methods": ",".join(methods),
        "--eval-seed-base": str(20_000_000 + offset),
        "--curve-eval-seed-base": str(20_030_000 + offset),
        "--policy-audit-batch-size": "100",
        "--n-training-threads": "1",
        "--buffer-length": "200000",
    }
    present = {item.split("=", 1)[0] for item in argv}
    output = list(argv)
    for flag, value in defaults.items():
        if flag not in present:
            output += [flag, value]
    return output


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n-agents", type=int, choices=(3, 4, 5, 6), required=True)
    parser.add_argument("--replicate", type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument(
        "--allow-ineligible",
        action="store_true",
        help="Run the locked intention-to-treat payload despite offline gate status.",
    )
    known, remaining = parser.parse_known_args()
    metadata_path = (
        DISTILL_ROOT / f"n{known.n_agents}_actor{known.replicate}"
        / "metadata.json"
    )
    if not payload_path(known.n_agents, known.replicate).exists():
        raise FileNotFoundError(payload_path(known.n_agents, known.replicate))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata["offline_gate"]["eligible"] and not known.allow_ineligible:
        raise RuntimeError(
            f"Payload N={known.n_agents}, replicate={known.replicate} "
            "did not pass the locked offline gate"
        )
    methods = methods_for(known.n_agents, known.replicate)
    protocol.METHODS = methods
    protocol.METRICS = ONLINE_METRICS
    protocol.evaluate_model = evaluate_model
    protocol.STATUS_TITLE = (
        f"Cardinality Online N={known.n_agents}, actor {known.replicate}"
    )
    protocol.PLOT_TITLE_PREFIX = (
        f"Cardinality N={known.n_agents}, actor {known.replicate}"
    )
    sys.argv = [
        sys.argv[0],
        *add_defaults(
            remaining, known.n_agents, known.replicate, methods
        ),
    ]
    protocol.main()


if __name__ == "__main__":
    main()
