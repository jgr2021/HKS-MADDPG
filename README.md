# HKS-MADDPG

**Task-aware graph construction and heat-kernel-signature features for cooperative multi-agent reinforcement learning.**

This repository implements an actor observation augmentation for **MADDPG** on
the three-agent, three-landmark MPE `simple_spread` task. Each agent reconstructs
an agent–landmark graph from its local observation, computes three focal HKS
values, and appends them to the observation before choosing an action.

The method combines a task-specific **weighted bipartite graph** with the
existing **heat kernel signature (HKS)** descriptor. The graph represents
agent–landmark proximity and shared-landmark relationships. HKS provides a
compact multiscale description of that graph.

```text
Local observation (18D)
    ├── agent–landmark graph → normalized Laplacian → focal HKS (3D)
    └────────────────────────── concatenate ────────────────┘
                                      ↓
                              MADDPG actor (21D)
                                      ↓
                              five discrete actions
```

The centralized critic and replay buffer use raw observations. The released
comparison is **Raw MADDPG versus HKS-augmented MADDPG**.
See [method and equations](docs/METHOD.md) for the graph, descriptor, and observation layout.

## Start here

| What you want to do | Entry point |
| --- | --- |
| Understand the method | [Method](docs/METHOD.md) |
| Train Raw or HKS | [`scripts/train.py`](scripts/train.py) |
| Evaluate a checkpoint or select the best checkpoint of a run | [`scripts/evaluate.py`](scripts/evaluate.py) |
| Summarize all five training seeds | [`scripts/summarize.py`](scripts/summarize.py) |
| Check settings, metrics, and output files | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Locate the implementation | [Code map](code/README.md) |

## Installation

The verified environment is **Python 3.9, PyTorch 2.5.1, NumPy 1.23.5, Gym 0.21.0**.
Training requires CUDA. The MPE source is included in `code/multiagent/`.

```bash
git clone https://github.com/jgr2021/HKS-MADDPG.git
cd HKS-MADDPG
python -m pip install -r requirements.txt
```

Gym 0.21 uses an older build system; see the
[installation notes](docs/REPRODUCIBILITY.md#installation) if installation fails.

## Training

Run commands from the repository root:

```bash
python scripts/train.py --method raw --seed 41 --output outputs
python scripts/train.py --method hks --seed 41 --output outputs
```

Each command trains one seed for **2,000,000 environment transitions**, using
**four rollout environments** and CUDA optimization. For the five-seed comparison,
complete Raw seeds **41–45**, then HKS seeds **41–45**. Existing run directories
are never overwritten.

## Evaluation

Select the checkpoint with highest mean return on the run's 100-episode
learning-curve bank, then evaluate it on the common 500-episode final bank:

```bash
python scripts/evaluate.py --run-dir outputs/hks/seed_41 --output outputs/evaluation/hks_41
```

Repeat for each method and seed. Aggregate the five seed summaries as described
in [Reproducibility](docs/REPRODUCIBILITY.md#five-seed-summary).
To evaluate a particular checkpoint instead, use `--checkpoint path/to/model.pt`.
Evaluation defaults to CPU; `--device cuda` enables CUDA inference.

## Repository structure

```text
HKS-MADDPG/
├── code/          MADDPG, HKS features, MPE, and retained training implementation
├── scripts/       train, evaluate, and summarize commands
├── docs/          method, evaluation protocol, and archive information
├── tests/         feature, checkpoint, and evaluation checks
├── requirements.txt
└── README.md
```

Training creates `outputs/` locally. This is a code release; manuscript files,
trained checkpoints, and experiment result tables are not bundled.

## Verification

```bash
python -m unittest discover -s tests -v
```

## References

[1] R. Lowe et al., “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,” in *Advances in Neural Information Processing Systems*, 2017. [Online](https://arxiv.org/abs/1706.02275).

[2] J. Sun, M. Ovsjanikov, and L. Guibas, “A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion,” *Computer Graphics Forum*, 2009. [Online](https://doi.org/10.1111/j.1467-8659.2009.01515.x).

[3] shariqiqbal2810, “MADDPG-PyTorch,” GitHub repository. [Online](https://github.com/shariqiqbal2810/maddpg-pytorch).

[4] OpenAI, “Multi-Agent Particle Environment,” GitHub repository. [Online](https://github.com/openai/multiagent-particle-envs).
