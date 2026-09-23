# HKS-MADDPG

Research code for multi-agent reinforcement learning with heat-kernel-signature (HKS) features in the MPE `simple_spread` task.

## HKS actor features

The `fixed_hks` configuration reconstructs three agent positions and three landmark positions from each actor's raw 18-dimensional observation. It builds a complete agent–landmark bipartite graph with Gaussian edge weights:

```text
W_uv = B_uv * exp(-||x_u - x_v||^2 / (2 * 0.6^2))
```

Here `B_uv` is one for agent–landmark pairs and zero otherwise. From the normalized graph Laplacian, the actor takes its focal heat-kernel diagonal at diffusion times 0.5, 1, and 2. Appending these three values gives a 21-dimensional actor input. Replay and centralized critic inputs retain raw observations and actions.

The repository also contains other feature configurations and experimental scripts.

## Repository layout

| Path | Contents |
| --- | --- |
| `algorithms/` | Multi-agent learning implementations |
| `experiments/` | Training, evaluation, and analysis scripts |
| `utils/` | Graph features and policy modules |
| `multiagent/` | Particle environment |
| `tests/` | Code checks |

## Installation

```bash
pip install -r requirements.txt
```
