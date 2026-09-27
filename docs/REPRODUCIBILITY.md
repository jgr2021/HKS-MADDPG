# Training and evaluation

## Installation

The code was checked in Python 3.9 with PyTorch 2.5.1+cu124, NumPy 1.23.5,
and Gym 0.21.0. Install a CUDA-enabled PyTorch build compatible with your GPU
before training. The runtime checks CUDA availability and records the devices
used for optimization in `gpu_device_audit.json`.

The requirements retain the legacy Gym API used by the bundled MPE. If modern
packaging tools reject Gym 0.21's metadata, use a separate Python 3.9 environment
with the older build tools:

```bash
python -m pip install "pip==23.0.1" "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install -r requirements.txt
```

The dependency versions are inherited from the experiment environment. Only
the existing local environment was used for the repository checks; a fresh
installation is not claimed to have been tested.

## Default training settings

| Setting | Value |
| --- | --- |
| Algorithm / task | MADDPG / MPE `simple_spread` |
| Agents / landmarks / actions | 3 / 3 / 5 discrete actions |
| Seeds | 41, 42, 43, 44, 45 |
| Budget per seed | 2,000,000 environment transitions |
| Episode length | 25 steps |
| Rollout environments | 4 |
| Hidden width | 64 |
| Replay capacity / batch size | 1,000,000 / 1,024 |
| Learning rate / discount / target rate | 0.01 / 0.95 / 0.01 |
| Update interval | Every 100 collected transitions after replay warmup |
| Updates at each interval | 4 rounds, updating all three agents |
| Exploration schedule | 0.3 to 0 over the transition budget |
| Saved checkpoint interval | 20,000 transitions |
| Curve / final evaluation | 100 / 500 episodes, each of length 25 |

Four rollout environments contribute four transitions per environment step.
The budget remains 2M transitions (80,000 episodes). The update loop also uses
the rollout count for its update rounds; changing it is therefore a training
protocol change even when the transition budget is held fixed.

The public entry point calls the existing experiment trainer, registers the
original Raw/HKS actors, and preserves their parameter names. Environment
simulation and training action collection use CPU; optimization uses CUDA.

## Output files

```text
outputs/
├── raw/seed_41/
│   ├── checkpoints/                 portable checkpoint copies
│   ├── learning_curve_eval.csv      curve-bank metrics for checkpoint selection
│   ├── per_evaluation_episode_metrics.csv  final-step checkpoint evaluation
│   ├── summary.json                 final-step summary
│   ├── protocol.json                budgets and evaluation bank settings
│   ├── resolved_config.json         actor, critic, and training settings
│   ├── training_counters.json       actual transition/checkpoint counts
│   ├── gpu_device_audit.json        optimization device evidence
│   └── training.log                 progress and diagnostics
├── hks/seed_41/                     same layout
├── models/                         trainer's original checkpoints / TensorBoard logs
└── evaluation/                     separately requested checkpoint evaluations
```

The trainer retains its original model outputs and copies checkpoints into each
run directory. These generated files stay outside Git.

## Best-checkpoint evaluation

1. Evaluate each checkpoint on the same 100-episode bank, seeds
   `91000000` through `91000099`.
2. For each method and each training seed, select the checkpoint with highest
   mean episode return. Ties use the earlier row in the chronological curve CSV.
3. Evaluate that checkpoint on the separate 500-episode bank, seeds
   `92000000` through `92000499`.
4. Compute the mean and sample standard deviation across all five training seeds.

```bash
python scripts/evaluate.py --run-dir outputs/raw/seed_41 --output outputs/evaluation/raw_41
python scripts/evaluate.py --run-dir outputs/hks/seed_41 --output outputs/evaluation/hks_41
```

Repeat for seeds 42–45. Each output contains `episodes.csv` and `summary.json`,
including the selected checkpoint, curve return, and evaluation bank. The
script reads checkpoint copies inside the run directory, so it does not depend
on the original machine's absolute checkpoint paths.

Final-step results from training remain separate. Selection is within each
seed's training trajectory; it never selects only one training seed.

For an explicit checkpoint:

```bash
python scripts/evaluate.py --checkpoint outputs/hks/seed_41/checkpoints/model_final_2000000.pt --output outputs/evaluation/hks_41_final
```

The public tools load Raw/HKS checkpoints. Other historical actor types may
require the additional registrations in the archived experiment scripts.

## Five-seed summary

```bash
python scripts/summarize.py outputs/evaluation/hks_41/summary.json outputs/evaluation/hks_42/summary.json outputs/evaluation/hks_43/summary.json outputs/evaluation/hks_44/summary.json outputs/evaluation/hks_45/summary.json --output outputs/evaluation/hks_summary.csv
```

Repeat for Raw, or supply both methods' ten summary files in one command. The
script requires all five seeds and matching 100/500-episode banks and devices.
The reported standard deviation uses `ddof=1` across the five seed means.

## Metric definitions

| Metric | Definition |
| --- | --- |
| `return` | Sum over 25 steps of the mean reward across the three agents |
| `final_coverage` | Number of landmarks within distance **< 0.10** of at least one agent at the final step; 0–3 |
| `coverage_fraction` | `final_coverage / 3`; multiply by 100 to display a percentage |
| `full_coverage_rate` | Fraction of evaluation episodes with all three landmarks covered at the final step |
| `collision_step_rate` | Fraction of steps containing at least one colliding agent pair |

Coverage is a landmark coverage fraction; it is distinct from the separately
recorded coverage-radius AUC. Do not multiply coverage by episode length. Return
is already summed over steps by this evaluator.

## Verification scope

The public tests check the HKS formula against a matrix exponential, descriptor
permutation behavior, actor/critic dimensions, checkpoint round trips,
deterministic environment evaluation, and five-seed aggregation. They do not
retrain the published experiment or claim to reproduce its numerical results.
