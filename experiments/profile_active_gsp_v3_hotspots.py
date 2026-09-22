import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.active_gsp_v3_features import (  # noqa: E402
    AGENT_AGENT_SCALE,
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_SENSITIVITY,
    N_ACTIONS,
    SIGMA_AGENT_AGENT,
    SIGMA_AGENT_LANDMARK,
    _coverage_potential,
    _coverage_signal,
    _crowding_potential,
    _crowding_signal,
    _dirichlet_energy,
    _gaussian,
    predict_self_position_after_action,
)
from utils.gsp_features import (  # noqa: E402
    N_AGENTS,
    N_LANDMARKS,
    reconstruct_geometry_from_raw_observation,
)
from utils.make_env import make_env  # noqa: E402


COMPONENTS = [
    "local observation parsing",
    "candidate action next-state construction",
    "agent-agent distance",
    "agent-landmark distance",
    "graph edge / weight calculation",
    "coverage potential",
    "crowding potential",
    "Dirichlet energy",
]


def collect_local_observations(count=1200, seed=9200):
    rng = np.random.RandomState(seed)
    env = make_env("simple_spread", discrete_action=True)
    env.seed(seed)
    observations = []
    try:
        obs = env.reset()
        while len(observations) < count:
            for agent_index, local_obs in enumerate(obs):
                observations.append((np.asarray(local_obs, dtype=np.float32), agent_index))
                if len(observations) >= count:
                    break
            actions = []
            for _ in range(3):
                action = np.zeros(5, dtype=np.float32)
                action[int(rng.randint(5))] = 1.0
                actions.append(action)
            obs, _, _, _ = env.step(actions)
    finally:
        env.close()
    return observations


def timed_reference_like_call(local_obs, agent_index, totals):
    call_start = time.perf_counter()

    start = time.perf_counter()
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        local_obs,
        agent_index,
    )
    totals["local observation parsing"] += time.perf_counter() - start

    states = []
    for action_index in range(N_ACTIONS):
        start = time.perf_counter()
        candidate_agents = agent_positions.copy()
        candidate_agents[int(agent_index)] = predict_self_position_after_action(
            local_obs,
            action_index,
            dt=DEFAULT_DT,
            damping=DEFAULT_DAMPING,
            sensitivity=DEFAULT_SENSITIVITY,
        )
        totals["candidate action next-state construction"] += time.perf_counter() - start
        states.append(candidate_agents)

    all_agent_states = [agent_positions] + states
    for candidate_agents in all_agent_states:
        start = time.perf_counter()
        aa_dist = np.linalg.norm(
            candidate_agents[:, None, :] - candidate_agents[None, :, :],
            axis=2,
        )
        totals["agent-agent distance"] += time.perf_counter() - start

        start = time.perf_counter()
        al_dist = np.linalg.norm(
            candidate_agents[:, None, :] - landmark_positions[None, :, :],
            axis=2,
        )
        totals["agent-landmark distance"] += time.perf_counter() - start

        start = time.perf_counter()
        adjacency = np.zeros((N_AGENTS + N_LANDMARKS, N_AGENTS + N_LANDMARKS), dtype=np.float64)
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                adjacency[i, j] = _gaussian(aa_dist[i, j], SIGMA_AGENT_AGENT, AGENT_AGENT_SCALE)
                adjacency[j, i] = adjacency[i, j]
        for i in range(N_AGENTS):
            for j in range(N_LANDMARKS):
                node_j = N_AGENTS + j
                adjacency[i, node_j] = _gaussian(al_dist[i, j], SIGMA_AGENT_LANDMARK)
                adjacency[node_j, i] = adjacency[i, node_j]
        degree = adjacency.sum(axis=1)
        inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        normalized_adjacency = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
        laplacian = np.eye(N_AGENTS + N_LANDMARKS, dtype=np.float64) - normalized_adjacency
        laplacian = 0.5 * (laplacian + laplacian.T)
        totals["graph edge / weight calculation"] += time.perf_counter() - start

        start = time.perf_counter()
        _coverage_potential(candidate_agents, landmark_positions)
        totals["coverage potential"] += time.perf_counter() - start

        start = time.perf_counter()
        _crowding_potential(candidate_agents)
        totals["crowding potential"] += time.perf_counter() - start

        start = time.perf_counter()
        _dirichlet_energy(laplacian, _coverage_signal(candidate_agents, landmark_positions))
        _dirichlet_energy(laplacian, _crowding_signal(candidate_agents))
        totals["Dirichlet energy"] += time.perf_counter() - start

    return time.perf_counter() - call_start


def main():
    observations = collect_local_observations()
    totals = {name: 0.0 for name in COMPONENTS}
    total_elapsed = 0.0

    for local_obs, agent_index in observations:
        total_elapsed += timed_reference_like_call(local_obs, agent_index, totals)

    measured = sum(totals.values())
    overhead = max(total_elapsed - measured, 0.0)
    rows = []
    for name in COMPONENTS:
        rows.append(
            {
                "component": name,
                "ms_per_call": totals[name] * 1000.0 / len(observations),
                "percentage": 100.0 * totals[name] / total_elapsed,
                "calls_per_env_step": 3.0,
                "optimization_decision": "",
            }
        )
    rows.append(
        {
            "component": "Python loop / object creation / array conversion",
            "ms_per_call": overhead * 1000.0 / len(observations),
            "percentage": 100.0 * overhead / total_elapsed,
            "calls_per_env_step": 3.0,
            "optimization_decision": "",
        }
    )
    rows.append(
        {
            "component": "CPU-GPU transfer",
            "ms_per_call": 0.0,
            "percentage": 0.0,
            "calls_per_env_step": 0.0,
            "optimization_decision": "No transfer in feature path.",
        }
    )

    decisions = {
        "local observation parsing": "Vectorize parsing over [M, 18] and avoid per-agent reconstruct calls.",
        "candidate action next-state construction": "Broadcast all five candidate actions at once.",
        "agent-agent distance": "Compute [M, actions, A, A] distances in one broadcast.",
        "agent-landmark distance": "Compute [M, actions, A, L] distances in one broadcast.",
        "graph edge / weight calculation": "Avoid graph/Laplacian objects; use pairwise weights directly.",
        "coverage potential": "Compute nearest-agent landmark distances batched.",
        "crowding potential": "Compute upper-triangle AA proximity batched.",
        "Dirichlet energy": "Use direct weighted differences; no 6x6 Laplacian or eigendecomposition.",
        "Python loop / object creation / array conversion": "Remove action/env Python loops from hot path.",
    }
    for row in rows:
        if not row["optimization_decision"]:
            row["optimization_decision"] = decisions[row["component"]]

    output_dir = Path("experiments") / f"active_gsp_v3_hotspot_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "hotspots.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "ms_per_call",
                "percentage",
                "calls_per_env_step",
                "optimization_decision",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(output_dir)
    for row in rows:
        print(
            f"{row['component']}: {row['ms_per_call']:.6f} ms/call, "
            f"{row['percentage']:.2f}%"
        )


if __name__ == "__main__":
    main()
