"""Evaluate frozen local actors against archived same-state action outcomes.

Labels are used after action selection, never as actor inputs. No environment
is stepped, no loss is fitted, and checkpoint files are never written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.audit_active_gsp_compression import residual_with_features, sha256
from experiments.diagnose_learned_active_gsp_masks import MASKS, write_csv
from utils.networks import _active_gsp_v3_features_tensor

METHOD = "learned_active_gsp_residual"
MASK_NAMES = tuple(MASKS)
OUTCOME_NAMES = ("hungarian_delta", "radius_auc_delta", "min_separation_delta",
                 "reward_delta", "collision")


def group_candidates(data):
    """Validate five-action groups and infer focal IDs from audited triples."""
    group = np.asarray(data["group"])
    count = len(group)
    if count == 0 or count % 15:
        raise ValueError("Expected complete three-agent, five-action state groups")
    for key in ("raw", "action", "active", "targets", "reward_delta", "collision",
                "train_seed", "episode"):
        if len(data[key]) != count or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid or non-finite dataset field: {key}")
    if data["raw"].shape != (count, 18) or data["action"].shape != (count, 5):
        raise ValueError("Expected raw 18D observations and five one-hot actions")
    action_id = data["action"].argmax(axis=1)
    np.testing.assert_array_equal(data["action"], np.eye(5)[action_id])
    order = np.lexsort((action_id, group))
    grouped_ids = group[order].reshape(-1, 5)
    groups = len(grouped_ids)
    np.testing.assert_array_equal(grouped_ids, np.repeat(np.arange(groups)[:, None], 5, axis=1))
    np.testing.assert_array_equal(action_id[order].reshape(-1, 5),
                                  np.tile(np.arange(5), (groups, 1)))
    raw = data["raw"][order].reshape(groups, 5, 18)
    np.testing.assert_array_equal(raw, np.repeat(raw[:, :1], 5, axis=1))
    result = {"raw": raw[:, 0].astype(np.float32), "group": grouped_ids[:, 0]}
    for key in ("train_seed", "episode"):
        values = data[key][order].reshape(groups, 5)
        np.testing.assert_array_equal(values, np.repeat(values[:, :1], 5, axis=1))
        triples = values[:, 0].reshape(-1, 3)
        np.testing.assert_array_equal(triples, np.repeat(triples[:, :1], 3, axis=1))
        result[key] = values[:, 0]

    # The historical collector emits focal agents 0,1,2 for every state.
    # Cross-view geometry verifies this provenance assumption before IDs are used.
    triples = result["raw"].reshape(-1, 3, 18)
    positions = triples[:, :, 2:4]
    landmarks = triples[:, :, 4:10].reshape(-1, 3, 3, 2) + positions[:, :, None, :]
    np.testing.assert_allclose(landmarks, np.repeat(landmarks[:, :1], 3, axis=1),
                               atol=1e-6, rtol=1e-6)
    for focal in range(3):
        others = [agent for agent in range(3) if agent != focal]
        expected = positions[:, others] - positions[:, focal, None]
        np.testing.assert_allclose(triples[:, focal, 10:14].reshape(-1, 2, 2),
                                   expected, atol=1e-6, rtol=1e-6)
    result["focal"] = np.arange(groups) % 3
    result["active"] = data["active"][order].reshape(groups, 5, 4)
    targets = data["targets"][order].reshape(groups, 5, 3)
    rewards = data["reward_delta"][order].reshape(groups, 5)
    np.testing.assert_allclose(targets[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(rewards[:, 0], 0.0, atol=1e-12)
    result["outcomes"] = np.concatenate([
        targets, rewards[:, :, None],
        data["collision"][order].reshape(groups, 5, 1),
    ], axis=2)
    if not np.isin(result["outcomes"][:, :, -1], [0, 1]).all():
        raise ValueError("Collision labels must be binary")
    return result


def score_masks(policy, local_obs):
    """Return full and masked logits using only one focal actor's raw rows."""
    if local_obs.ndim != 2 or local_obs.shape[1] != 18:
        raise ValueError("Actor inputs must be [batch,18]")
    if policy.training or policy.feature_mode != "active_gsp_v3":
        raise ValueError("Expected a frozen full Active GSP actor in eval mode")
    with torch.no_grad():
        raw_logits = policy.mlp(local_obs)
        features = policy.action_features(local_obs)
        logits = torch.stack([
            raw_logits + residual_with_features(
                policy, local_obs, features * features.new_tensor(MASKS[name])
            ) for name in MASK_NAMES
        ], dim=1)
        torch.testing.assert_close(logits[:, 0], policy(local_obs), rtol=1e-5, atol=1e-6)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite policy logits")
    return logits, raw_logits, features


def selected_outcomes(outcomes, actions):
    if outcomes.ndim != 3 or outcomes.shape[1] != 5:
        raise ValueError("Expected [groups,5,outcomes]")
    if actions.shape != (len(outcomes),) or not np.isin(actions, np.arange(5)).all():
        raise ValueError("Invalid selected action indices")
    return outcomes[np.arange(len(outcomes)), actions]


def policy_digest(policies):
    digest = hashlib.sha256()
    for index, policy in enumerate(policies):
        for key, value in sorted(policy.state_dict().items()):
            digest.update(f"{index}:{key}".encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def policy_sources():
    root = REPO_ROOT / "experiments/corrected_active_gsp_multiseed_20260711_124239"
    mapping = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    result = [
        {"seed": seed, "cohort": "initial_1_2_3", "checkpoint": str(
            REPO_ROOT / mapping[str(seed)][METHOD] / "checkpoints/model_final_100000.pt")}
        for seed in (1, 2, 3)
    ]
    path = REPO_ROOT / "experiments/multiscale_spectral_screening_eval_20260711/sources.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        result.extend({"seed": int(row["train_seed"]), "cohort": "later_11_12_13",
                       "checkpoint": row["checkpoint"]}
                      for row in csv.DictReader(handle) if row["method"] == METHOD)
    if sorted(row["seed"] for row in result) != [1, 2, 3, 11, 12, 13]:
        raise ValueError("Missing or duplicate frozen policy source")
    return result


def summarize(data, logits, raw_logits, source):
    actions = logits.argmax(axis=2)
    full_actions = actions[:, 0]
    full_outcomes = selected_outcomes(data["outcomes"], full_actions)
    raw = data["raw"]
    separation = np.linalg.norm(raw[:, 10:14].reshape(-1, 2, 2), axis=2).min(axis=1)
    landmark = np.linalg.norm(raw[:, 4:10].reshape(-1, 3, 2), axis=2).min(axis=1)
    strata = {"all": np.ones(len(raw), dtype=bool)}
    strata.update({f"source_{seed}": data["train_seed"] == seed
                   for seed in np.unique(data["train_seed"])})
    strata.update({"agent_sep_lt030": separation < 0.30,
                   "agent_sep_030_060": (separation >= 0.30) & (separation < 0.60),
                   "agent_sep_ge060": separation >= 0.60,
                   "landmark_lt030": landmark < 0.30,
                   "landmark_030_060": (landmark >= 0.30) & (landmark < 0.60),
                   "landmark_ge060": landmark >= 0.60})
    rows = []
    for index, name in enumerate(MASK_NAMES):
        outcomes = selected_outcomes(data["outcomes"], actions[:, index])
        delta = outcomes - full_outcomes
        changed = actions[:, index] != full_actions
        logit_effect = logits[:, index] - logits[:, 0]
        logit_effect -= logit_effect.mean(axis=1, keepdims=True)
        margin = np.sort(logits[:, index], axis=1)[:, -1] - np.sort(logits[:, index], axis=1)[:, -2]
        residual = logits[:, index] - raw_logits
        residual -= residual.mean(axis=1, keepdims=True)
        ties = (logits[:, index] == logits[:, index].max(axis=1, keepdims=True)).sum(axis=1) > 1
        for stratum, selected in strata.items():
            count = int(selected.sum())
            if not count:
                continue
            changed_selected = selected & changed
            row = {"train_seed": source["seed"], "cohort": source["cohort"],
                   "mask": name, "stratum": stratum, "groups": count,
                   "changed_groups": int(changed_selected.sum()),
                   "action_flip_rate": float(changed[selected].mean()),
                   "exact_tie_rate": float(ties[selected].mean()),
                   "mean_logit_margin": float(margin[selected].mean()),
                   "mean_abs_centered_effect": float(np.abs(logit_effect[selected]).mean()),
                   "mean_abs_centered_residual": float(np.abs(residual[selected]).mean())}
            for metric_index, metric in enumerate(OUTCOME_NAMES):
                row[f"{metric}_mask_minus_full"] = float(delta[selected, metric_index].mean())
                row[f"{metric}_changed_only"] = (
                    float(delta[changed_selected, metric_index].mean())
                    if changed_selected.any() else float("nan"))
            rows.append(row)
    return rows, actions


def aggregate_cohorts(rows):
    output = []
    keys = ("action_flip_rate", "mean_abs_centered_effect") + tuple(
        f"{name}_mask_minus_full" for name in OUTCOME_NAMES)
    for cohort in ("initial_1_2_3", "later_11_12_13"):
        for mask in MASK_NAMES:
            selected = [row for row in rows if row["cohort"] == cohort
                        and row["mask"] == mask and row["stratum"] == "all"]
            if len(selected) != 3:
                raise ValueError("Each cohort must contain exactly three model seeds")
            for metric in keys:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                half = 4.3026527297 * values.std(ddof=1) / math.sqrt(3)
                output.append({"cohort": cohort, "mask": mask, "metric": metric,
                               "n_policy_seeds": 3, "mean": float(values.mean()),
                               "ci95_low": float(values.mean() - half),
                               "ci95_high": float(values.mean() + half),
                               "per_seed_values": ";".join(f"{value:.9g}" for value in values)})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT /
                        "experiments/counterfactual_dataset_fivefold_20260811/counterfactual_dataset.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = policy_sources()
    for source in sources:
        source["sha256_before"] = sha256(Path(source["checkpoint"]))
    dataset_hash = sha256(args.dataset)
    metadata_path = args.dataset.parent / "metadata.json"
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if dataset_hash != dataset_metadata["combined_sha256"]:
        raise ValueError("Dataset differs from its archived combined hash")
    protocol = {"status": "started", "dataset": str(args.dataset.resolve()),
                "dataset_sha256_before": dataset_hash, "sources": sources,
                "masks": MASKS, "actor_input": "one focal raw 18D observation only",
                "label_use": "lookup only after frozen action selection; no fitting",
                "new_environment_steps": 0, "new_training_steps": 0,
                "comparison": "mask minus full; same weights, same state, same non-focal actions",
                "cohorts": "report seeds 1-3 and 11-13 separately; no post-hoc promotion",
                "limits": "conditional one-step effects on archived geometric-control states, not full-policy returns",
                "tie_policy": "numpy first argmax for diagnostic lookup; report exact ties",
                "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
                "batch_size": args.batch_size}
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with np.load(args.dataset, allow_pickle=False) as loaded:
        data = group_candidates({key: loaded[key] for key in
                                 ("raw", "action", "active", "targets", "reward_delta",
                                  "collision", "group", "train_seed", "episode")})
    features = []
    for start in range(0, len(data["raw"]), args.batch_size):
        features.append(_active_gsp_v3_features_tensor(
            torch.from_numpy(data["raw"][start:start + args.batch_size])).numpy())
    np.testing.assert_allclose(np.concatenate(features), data["active"], rtol=2e-4, atol=2e-6)
    rows = []
    for source in sources:
        model = MADDPG.init_from_save(source["checkpoint"])
        if model.actor_anchor is not None:
            raise ValueError("Expected an unanchored original-protocol checkpoint")
        model.prep_rollouts(device="cpu")
        before = policy_digest(model.policies)
        logits = np.empty((len(data["raw"]), len(MASK_NAMES), 5), dtype=np.float32)
        raw_logits = np.empty((len(data["raw"]), 5), dtype=np.float32)
        for focal, policy in enumerate(model.policies):
            indices = np.flatnonzero(data["focal"] == focal)
            for start in range(0, len(indices), args.batch_size):
                batch = indices[start:start + args.batch_size]
                scored, raw, _ = score_masks(policy, torch.from_numpy(data["raw"][batch]))
                logits[batch] = scored.numpy()
                raw_logits[batch] = raw.numpy()
        summary, actions = summarize(data, logits, raw_logits, source)
        rows.extend(summary)
        source["policy_state_unchanged"] = before == policy_digest(model.policies)
        source["sha256_after"] = sha256(Path(source["checkpoint"]))
        if not source["policy_state_unchanged"] or source["sha256_after"] != source["sha256_before"]:
            raise RuntimeError("Frozen policy changed during diagnostic")
        np.savez_compressed(args.output_dir / f"seed_{source['seed']}_decisions.npz",
                            group=data["group"], focal=data["focal"],
                            source_seed=data["train_seed"], episode=data["episode"],
                            mask_names=np.asarray(MASK_NAMES), logits=logits,
                            raw_mlp_subnetwork_logits=raw_logits, selected_actions=actions)
        print(f"Completed frozen policy seed {source['seed']}: {len(actions)} groups", flush=True)
    write_csv(args.output_dir / "per_policy_stratum.csv", rows)
    write_csv(args.output_dir / "cohort_intervals.csv", aggregate_cohorts(rows))
    protocol.update({"status": "completed", "groups": len(data["raw"]),
                     "archived_candidate_rows": len(data["raw"]) * 5,
                     "sources": sources, "dataset_sha256_after": sha256(args.dataset),
                     "group_mapping_and_feature_equivalence": "pass"})
    if protocol["dataset_sha256_after"] != dataset_hash:
        raise RuntimeError("Input dataset changed during diagnostic")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
