"""Generic fixed-state margin/KL audit for one NxN online run."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.analyze_hybrid_policy_margin_drift import (
    flatten_agent_values,
    pinsker_argmax_certificate,
    policy_probabilities,
)
from utils.make_env import make_env


FROZEN = "cardinality_frozen"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initial_margin_quartile_masks(margins, mode):
    """Return four deterministic masks defined only by initial margins.

    ``threshold`` retains the original numeric-quantile analysis. ``rank``
    stably orders equal margins by their fixed flattened-state index and then
    assigns equal-sample quartiles. The latter avoids empty strata when a large
    probability-margin tie lands on a quantile boundary.
    """
    margins = np.asarray(margins, dtype=np.float64)
    if mode == "threshold":
        thresholds = np.quantile(margins, [0.25, 0.50, 0.75])
        edges = [-np.inf, *thresholds.tolist(), np.inf]
        masks = [
            (margins > edges[index]) & (margins <= edges[index + 1])
            for index in range(4)
        ]
        return masks, thresholds.tolist()
    if mode != "rank":
        raise ValueError(f"Unknown quartile mode: {mode}")
    order = np.argsort(margins, kind="stable")
    labels = np.empty(len(margins), dtype=np.int8)
    labels[order] = np.minimum(4 * np.arange(len(margins)) // len(margins), 3)
    return [labels == index for index in range(4)], []


def collect_fixed_observations(checkpoint, env_id, episodes, horizon, seed_base):
    model = MADDPG.init_from_save(str(checkpoint))
    model.prep_rollouts(device="cpu")
    env = make_env(env_id, discrete_action=model.discrete_action)
    observations = [[] for _ in model.agents]
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            for _ in range(horizon):
                for agent_index, value in enumerate(obs):
                    observations[agent_index].append(value.copy())
                tensors = [
                    torch.from_numpy(value).view(1, -1) for value in obs
                ]
                with torch.no_grad():
                    actions = [
                        action.cpu().numpy().ravel()
                        for action in model.step(tensors, explore=False)
                    ]
                obs, _, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                if all(dones):
                    break
    finally:
        env.close()
    return [
        torch.from_numpy(np.asarray(values, dtype=np.float32))
        for values in observations
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--output-name", default="cardinality_policy_drift")
    parser.add_argument(
        "--quartile-mode", choices=("threshold", "rank"),
        default="threshold",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = run_dir / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    methods = metadata["methods"]
    seeds = [int(value) for value in metadata["seeds"]]
    checkpoints = [int(value) for value in metadata["checkpoint_steps"]]
    if len(seeds) != 1:
        raise ValueError("Cardinality replicate audit expects one online seed")
    seed = seeds[0]
    env_ids = {
        metadata["method_specs"][method]["env_id"] for method in methods
    }
    if len(env_ids) != 1:
        raise RuntimeError("Every method in one cardinality run must share an env")
    env_id = env_ids.pop()
    initial_checkpoint = (
        run_dir / FROZEN / f"seed_{seed}" / "checkpoints" / "model_step64.pt"
    )
    observations = collect_fixed_observations(
        initial_checkpoint, env_id, args.episodes, args.horizon, args.seed_base
    )
    initial_model = MADDPG.init_from_save(str(initial_checkpoint))
    initial_model.prep_rollouts(device="cpu")
    initial_probs = policy_probabilities(initial_model, observations)
    initial_actions = [values.argmax(dim=1) for values in initial_probs]
    initial_margins = []
    for values in initial_probs:
        top_two = values.topk(2, dim=1).values
        initial_margins.append(top_two[:, 0] - top_two[:, 1])
    all_margins = flatten_agent_values(initial_margins)
    quartile_masks, quartiles = initial_margin_quartile_masks(
        all_margins, args.quartile_mode
    )

    rows = []
    for method in methods:
        for step in checkpoints:
            checkpoint = (
                run_dir / method / f"seed_{seed}" / "checkpoints"
                / f"model_step{step}.pt"
            )
            model = MADDPG.init_from_save(str(checkpoint))
            model.prep_rollouts(device="cpu")
            candidate_probs = policy_probabilities(model, observations)
            kl_values, agreement_values = [], []
            for reference, candidate, action in zip(
                initial_probs, candidate_probs, initial_actions
            ):
                kl_values.append((reference * (
                    reference.clamp_min(1e-8).log()
                    - candidate.clamp_min(1e-8).log()
                )).sum(dim=1))
                agreement_values.append(
                    (candidate.argmax(dim=1) == action).float()
                )
            kl = flatten_agent_values(kl_values)
            agreement = flatten_agent_values(agreement_values)
            certified = pinsker_argmax_certificate(all_margins, kl)
            flipped = agreement < 0.5
            row = {
                "method": method,
                "training_seed": seed,
                "checkpoint_step": step,
                "states": len(agreement),
                "initial_reference_kl_mean": float(kl.mean()),
                "initial_action_agreement": float(agreement.mean()),
                "action_flip_rate": float(flipped.mean()),
                "pinsker_argmax_certified_fraction": float(certified.mean()),
                "pinsker_certificate_violation_count": int(
                    (certified & flipped).sum()
                ),
            }
            for index in range(4):
                selected = quartile_masks[index]
                row[f"agreement_initial_margin_q{index + 1}"] = float(
                    agreement[selected].mean()
                )
                row[f"kl_initial_margin_q{index + 1}"] = float(
                    kl[selected].mean()
                )
            rows.append(row)
    write_csv(output / "policy_drift.csv", rows)
    (output / "metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "env_id": env_id,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed_base": args.seed_base,
        "states_per_agent": len(observations[0]),
        "quartile_mode": args.quartile_mode,
        "quartile_tie_break": (
            "stable flattened-state index" if args.quartile_mode == "rank"
            else "numeric thresholds"
        ),
        "initial_probability_margin_quartiles": quartiles,
        "quartile_sample_counts": [int(mask.sum()) for mask in quartile_masks],
        "checkpoints": checkpoints,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
