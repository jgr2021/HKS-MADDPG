"""Fixed-state action-margin and policy-drift diagnostics for hybrid runs."""

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
from utils.make_env import make_env


FROZEN = "hybrid_frozen"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_fixed_observations(checkpoint, episodes, horizon, seed_base):
    model = MADDPG.init_from_save(str(checkpoint))
    model.prep_rollouts(device="cpu")
    env = make_env("simple_spread", discrete_action=model.discrete_action)
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


def policy_probabilities(model, observations):
    output = []
    with torch.no_grad():
        for agent, values in zip(model.agents, observations):
            batches = []
            for start in range(0, len(values), 2048):
                logits = agent.policy(values[start:start + 2048])
                batches.append(torch.softmax(logits, dim=1).cpu())
            output.append(torch.cat(batches, dim=0))
    return output


def flatten_agent_values(values):
    return torch.cat(values, dim=0).numpy()


def pinsker_argmax_certificate(initial_margin, reference_kl):
    """Return states whose initial argmax is guaranteed to be unchanged."""
    initial_margin = np.asarray(initial_margin, dtype=np.float64)
    reference_kl = np.asarray(reference_kl, dtype=np.float64)
    return initial_margin > np.sqrt(
        2.0 * np.maximum(reference_kl, 0.0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=7_300_000)
    parser.add_argument("--output-name", default="policy_margin_drift")
    parser.add_argument(
        "--reference-method",
        default=FROZEN,
        help="Method whose step-64 actor defines the frozen initial policy.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        help=(
            "Checkpoint steps to audit. Defaults to checkpoint_steps from "
            "protocol_metadata.json."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = run_dir / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    methods = metadata["methods"]
    seeds = [int(seed) for seed in metadata["seeds"]]
    checkpoints = tuple(
        int(step)
        for step in (
            args.checkpoints
            if args.checkpoints is not None
            else metadata["checkpoint_steps"]
        )
    )
    if not checkpoints:
        raise ValueError("No checkpoints were selected")
    initial_checkpoint = (
        run_dir
        / args.reference_method
        / f"seed_{seeds[0]}"
        / "checkpoints"
        / "model_step64.pt"
    )
    observations = collect_fixed_observations(
        initial_checkpoint, args.episodes, args.horizon, args.seed_base
    )
    initial_model = MADDPG.init_from_save(str(initial_checkpoint))
    initial_model.prep_rollouts(device="cpu")
    initial_probs = policy_probabilities(initial_model, observations)
    initial_actions = [value.argmax(dim=1) for value in initial_probs]
    initial_margins = []
    for value in initial_probs:
        top_two = value.topk(2, dim=1).values
        initial_margins.append(top_two[:, 0] - top_two[:, 1])
    all_margins = flatten_agent_values(initial_margins)
    quartiles = np.quantile(all_margins, [0.25, 0.50, 0.75])
    bin_edges = [-np.inf, *quartiles.tolist(), np.inf]

    rows = []
    for method in methods:
        for seed in seeds:
            seed_dir = run_dir / method / f"seed_{seed}"
            for step in checkpoints:
                checkpoint = (
                    seed_dir / "checkpoints" / f"model_step{step}.pt"
                )
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                model = MADDPG.init_from_save(str(checkpoint))
                model.prep_rollouts(device="cpu")
                candidate_probs = policy_probabilities(model, observations)
                candidate_actions = [
                    value.argmax(dim=1) for value in candidate_probs
                ]
                kl_values = []
                agreement_values = []
                entropy_values = []
                candidate_margin_values = []
                for reference, candidate, reference_action in zip(
                    initial_probs, candidate_probs, initial_actions
                ):
                    kl_values.append(
                        (
                            reference
                            * (
                                reference.clamp_min(1e-8).log()
                                - candidate.clamp_min(1e-8).log()
                            )
                        ).sum(dim=1)
                    )
                    agreement_values.append(
                        (candidate.argmax(dim=1) == reference_action).float()
                    )
                    entropy_values.append(
                        -(
                            candidate
                            * candidate.clamp_min(1e-8).log()
                        ).sum(dim=1)
                    )
                    top_two = candidate.topk(2, dim=1).values
                    candidate_margin_values.append(
                        top_two[:, 0] - top_two[:, 1]
                    )
                kl = flatten_agent_values(kl_values)
                agreement = flatten_agent_values(agreement_values)
                entropy = flatten_agent_values(entropy_values)
                candidate_margin = flatten_agent_values(
                    candidate_margin_values
                )
                # If p is the initial action distribution and q is the
                # candidate, Pinsker gives ||p-q||_1 <= sqrt(2 KL(p||q)).
                # An initial top-two gap larger than this bound guarantees
                # that q retains p's argmax action.
                certified = pinsker_argmax_certificate(all_margins, kl)
                flipped = agreement < 0.5
                certificate_violations = certified & flipped
                base = {
                    "method": method,
                    "training_seed": seed,
                    "checkpoint_step": step,
                    "states": int(len(agreement)),
                    "initial_reference_kl_mean": float(kl.mean()),
                    "initial_action_agreement": float(agreement.mean()),
                    "candidate_entropy_mean": float(entropy.mean()),
                    "candidate_probability_margin_mean": float(
                        candidate_margin.mean()
                    ),
                    "action_flip_rate": float(flipped.mean()),
                    "pinsker_argmax_certified_fraction": float(
                        certified.mean()
                    ),
                    "pinsker_certificate_violation_count": int(
                        certificate_violations.sum()
                    ),
                    "certified_action_agreement": (
                        float(agreement[certified].mean())
                        if np.any(certified) else float("nan")
                    ),
                    "uncertified_action_agreement": (
                        float(agreement[~certified].mean())
                        if np.any(~certified) else float("nan")
                    ),
                    "flipped_initial_margin_mean": (
                        float(all_margins[flipped].mean())
                        if np.any(flipped) else float("nan")
                    ),
                    "agreed_initial_margin_mean": (
                        float(all_margins[~flipped].mean())
                        if np.any(~flipped) else float("nan")
                    ),
                    "flipped_kl_mean": (
                        float(kl[flipped].mean())
                        if np.any(flipped) else float("nan")
                    ),
                    "agreed_kl_mean": (
                        float(kl[~flipped].mean())
                        if np.any(~flipped) else float("nan")
                    ),
                }
                margin_array = all_margins
                for bin_index in range(4):
                    selected = (
                        (margin_array > bin_edges[bin_index])
                        & (margin_array <= bin_edges[bin_index + 1])
                    )
                    base[f"agreement_initial_margin_q{bin_index + 1}"] = (
                        float(agreement[selected].mean())
                    )
                    base[
                        f"pinsker_certified_fraction_q{bin_index + 1}"
                    ] = float(certified[selected].mean())
                rows.append(base)

    write_csv(output / "margin_drift_summary.csv", rows)
    (output / "metadata.json").write_text(
        json.dumps({
            "run_dir": str(run_dir),
            "episodes": args.episodes,
            "horizon": args.horizon,
            "seed_base": args.seed_base,
            "states_per_agent": len(observations[0]),
            "initial_probability_margin_quartiles": quartiles.tolist(),
            "checkpoints": list(checkpoints),
            "reference_method": args.reference_method,
            "argmax_certificate": (
                "initial top-two probability margin > sqrt(2 * "
                "KL(initial || candidate))"
            ),
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
