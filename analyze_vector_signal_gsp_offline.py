"""Offline feature diagnostics on existing deterministic policy trajectories."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from utils.active_gsp_v3_features import compute_active_gsp_v3_from_local_obs_batch
from utils.make_env import make_env
from utils.vector_signal_gsp_features import compute_vector_signal_features_from_local_obs_batch


DEFAULT_GEOM = Path("experiments/learned_active_gsp_100k_seed1_corrected/run_20260711_073619/learned_raw_potential_residual/seed_1/checkpoints/model_final_100000.pt")
DEFAULT_GSP = Path("experiments/learned_active_gsp_100k_seed1_corrected/run_20260711_073619/learned_active_gsp_residual/seed_1/checkpoints/model_final_100000.pt")


def corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.std() < 1e-12 or y.std() < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def geometry(env):
    a = np.asarray([x.state.p_pos for x in env.world.agents])
    l = np.asarray([x.state.p_pos for x in env.world.landmarks])
    d = np.linalg.norm(a[:, None] - l[None], axis=2)
    separation = min(np.linalg.norm(a[i] - a[j]) for i in range(3) for j in range(i + 1, 3))
    return d, separation


def collect(checkpoint, method, episodes, horizon, seed_base):
    model = MADDPG.init_from_save(str(checkpoint)); model.prep_rollouts(device="cpu")
    env = make_env("simple_spread", discrete_action=True); records = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode; torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32); episode_records = []
            for step in range(horizon):
                tensors = [torch.as_tensor(x).view(1, -1) for x in obs]
                with torch.no_grad(): actions = model.step(tensors, explore=False)
                indices = [int(x.argmax(dim=1).item()) for x in actions]
                active = compute_active_gsp_v3_from_local_obs_batch(obs)
                vector = compute_vector_signal_features_from_local_obs_batch(obs)
                selected = []
                for agent, action in enumerate(indices):
                    selected.append(np.concatenate([active[agent, action], vector[agent, action]]))
                obs, _, dones, _ = env.step([x.cpu().numpy().ravel() for x in actions])
                obs = np.asarray(obs, dtype=np.float32)
                distances, separation = geometry(env)
                collision = separation < 0.10
                same_agent_multi = np.unique(distances.argmin(axis=0)).size < 3
                low_unique = np.unique(distances.argmin(axis=1)).size < 3
                for agent, values in enumerate(selected):
                    episode_records.append({"method": method, "episode": episode, "step": step,
                                            "agent": agent, "values": values,
                                            "same_agent_multi": same_agent_multi,
                                            "low_unique": low_unique, "collision": collision,
                                            "min_separation": separation})
                if all(dones): break
            final_distances, _ = geometry(env)
            final_full = bool(np.all(final_distances.min(axis=0) < 0.10))
            for record in episode_records: record["low_final_full_coverage"] = not final_full
            records.extend(episode_records)
    finally: env.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometric-checkpoint", type=Path, default=DEFAULT_GEOM)
    parser.add_argument("--active-gsp-checkpoint", type=Path, default=DEFAULT_GSP)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("VECTOR_SIGNAL_GSP_DIAGNOSTICS.md"))
    args = parser.parse_args()
    records = collect(args.geometric_checkpoint, "geometric", args.episodes, args.horizon, 880000)
    records += collect(args.active_gsp_checkpoint, "active_gsp_6node", args.episodes, args.horizon, 880000)
    values = np.asarray([r["values"] for r in records])
    # [old geom 0:2, old energy 2:4, mu 4:7, M0 7:13, M1 13:19]
    geom_corr = np.asarray([[corr(values[:, i], values[:, j]) for j in range(4, 13)] for i in range(2)])
    energy_corr = np.asarray([[corr(values[:, i], values[:, j]) for j in range(13, 19)] for i in range(2, 4)])
    m1 = values[:, 13:19]; diag = m1[:, [0, 3, 5]].ravel(); off = m1[:, [1, 2, 4]]
    labels = ("same_agent_multi", "low_unique", "low_final_full_coverage", "collision", "min_separation")
    assignment_corr = {label: [corr(off[:, i], [r[label] for r in records]) for i in range(3)] for label in labels}
    max_geom = float(np.nanmax(np.abs(geom_corr))); max_energy = float(np.nanmax(np.abs(energy_corr)))
    max_assignment = float(np.nanmax(np.abs([assignment_corr[x] for x in labels[:3]])))
    lines = ["# Vector-Signal GSP Diagnostics", "",
             f"Offline only: {args.episodes} matched deterministic episodes per existing checkpoint, "
             f"{len(records):,} focal-agent action samples. No policy was changed.", "",
             "## Correlation summary", "",
             f"- Maximum absolute correlation of existing geometric deltas with delta_mu/delta_M0: **{max_geom:.3f}**.",
             f"- Maximum absolute correlation of current 6-node energy channels with delta_M1: **{max_energy:.3f}**.",
             f"- Maximum absolute off-diagonal M1 correlation with assignment-failure labels: **{max_assignment:.3f}**.", "",
             "## M1 distribution", "",
             "| entries | mean | std | p05 | median | p95 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in (("diagonal", diag), ("off-diagonal", off.ravel())):
        lines.append(f"| {name} | {data.mean():.6g} | {data.std():.6g} | {np.quantile(data,.05):.6g} | {np.median(data):.6g} | {np.quantile(data,.95):.6g} |")
    lines += ["", "## Off-diagonal M1 outcome correlations", "",
              "Channels are landmark pairs (L0,L1), (L0,L2), (L1,L2).", "",
              "| outcome | L0-L1 | L0-L2 | L1-L2 |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in labels:
        c = assignment_corr[label]; lines.append(f"| {label} | {c[0]:.3f} | {c[1]:.3f} | {c[2]:.3f} |")
    lines += ["", "## Interpretation", "",
              ("The new descriptors are strongly redundant with at least one existing geometric channel." if max_geom >= .9 else
               "The new descriptors are not a simple duplicate of the existing geometric deltas."),
              ("Off-diagonal M1 terms show assignment-relevant association in this diagnostic." if max_assignment >= .1 else
               "Off-diagonal M1 terms show weak assignment association; training benefit is uncertain."),
              ("M1 is less directly redundant with the current scalar energies and has non-degenerate diagonal/off-diagonal spread." if max_energy < .9 and off.std() > 1e-6 else
               "M1 overlaps strongly with the current energy branch or remains comparatively degenerate."), "",
              "These are observational correlations on policy-selected actions, not causal evidence. The locked screen determines promotion."]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    matrix_path = args.output.with_name("vector_signal_gsp_correlation_matrices.csv")
    rows = []
    for i in range(2):
        for j in range(9): rows.append({"group": "geometry_vs_mu_m0", "row": i, "column": j, "correlation": geom_corr[i,j]})
    for i in range(2):
        for j in range(6): rows.append({"group": "energy_vs_m1", "row": i, "column": j, "correlation": energy_corr[i,j]})
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(args.output)


if __name__ == "__main__": main()
