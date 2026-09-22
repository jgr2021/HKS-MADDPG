# HKS-MADDPG

**HKS** (Heat Kernel Signature) feature augmentation for Multi-Agent Reinforcement Learning, built on **MADDPG** ([Lowe et al., 2017](https://arxiv.org/abs/1706.02275)).

This repository contains the official experiment code for the HKS paper. The focal method augments each agent's local observation with a **graph-signal-processing (GSP) descriptor** before feeding it to the policy network, improving cooperative behavior on the multi-agent particle environment (MPE) `simple_spread` task.

## Method

Given the agents' and landmarks' positions, we construct a weighted geometric graph:

- **Agent–agent** edges: Gaussian kernel with bandwidth `σ = 0.8`
- **Agent–landmark** edges: Gaussian kernel with bandwidth `σ = 0.6`

The normalized graph Laplacian is then spectrally decomposed, and the **heat kernel signature** (HKS) is computed by applying the heat operator `exp(-t L)` to the graph signal across a set of diffusion times. The resulting per-node descriptor is concatenated to the raw 18-dim observation and fed into a standard MLP actor.

Three observation descriptors are supported by the `LocalActor`:

| descriptor    | meaning                                              |
|---------------|------------------------------------------------------|
| `raw`         | unmodified 18-dim observation (baseline)             |
| `fixed_hks`   | HKS with a fixed Gaussian bandwidth (the focal method)|
| `adaptive_hks`| HKS with an adaptive, median-distance bandwidth       |

## Environment

The experiments run on **MPE `simple_spread`** with discrete actions. Three agents must cooperatively cover three landmarks without colliding.

## Repository layout

```
algorithms/       MADDPG and the MAPPO/HKS policy implementations
experiments/      training, evaluation, and analysis scripts
utils/            HKS feature builders, environment wrappers, networks
multiagent/       the MPE multi-agent environment and scenarios
tests/            unit tests for features and policies
main.py           MADDPG training entry point
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements include PyTorch, NumPy, gym, and tensorboardX.

## Usage

Train MADDPG on `simple_spread`:

```bash
python main.py simple_spread maddpg_hks --discrete_action --seed 41 --n_episodes 25000
```

## Citation

```bibtex
@misc{hks-maddpg,
  title        = {HKS: Heat-Kernel-Signature Feature Augmentation for Multi-Agent Reinforcement Learning},
  author       = {Gurui Jin},
  year         = {2026},
  howpublished = {\url{https://github.com/jgr2021/hks-maddpg}},
}
```

## License

MIT
