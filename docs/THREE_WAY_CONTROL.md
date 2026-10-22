# Raw / Adjacency Statistics / HKS controlled comparison

## Scope

Base revision: `c2cb3676332bfb082589288afc138b3fa97114a0`.
This change exposes the existing `AdjacencyStatisticsPolicy` as
`--method fixed_adjacency`. It does not change MADDPG, the environment, replay,
critic, graph weights, or either feature implementation. Evaluation already
calls the shared registration function, so the saved actor type is restored
without a separate method switch. No communication degradation is introduced.

| CLI method | Registered actor | Actor input | Critic input |
|---|---|---:|---:|
| raw | mlp | 18 | 69 |
| fixed_adjacency | confirm_fixed_adjacency | 21 | 69 |
| hks | confirm_fixed_hks | 21 | 69 |

All actors have two hidden layers of width 64 and five action logits. With input
BatchNorm enabled, a raw actor has 5,737 trainable parameters and either augmented
actor has 5,935 (198 more: 3×64 input weights and 6 BatchNorm affine parameters).
Each critic has 8,843 parameters. There are three independent agents; targets
are copies, not extra independently optimized capacity. Adjacency and HKS have
identical parameter names/shapes and, under the same Torch seed, identical
initial weights. Raw does not have equal capacity or identical RNG consumption.
No trainable parameters are added by either descriptor.

Both descriptors construct the same six-node Gaussian weighted bipartite graph
(default sigma 0.6, no agent–agent or landmark–landmark edges). Adjacency reads
`W[:,0,3:6]` and returns mean, max, population standard deviation. HKS reads the
focal diagonal of the normalized-Laplacian heat kernel at t=0.5,1,2. The raw
18-vector is preserved in both augmented observations; replay remains raw.

## Run the fixed protocol

Use the documented CUDA training environment in REPRODUCIBILITY.md. From the
repository root, first run:

```bash
python -m unittest discover -s tests -v
```

Then run all fifteen jobs (30 million training transitions in total):

```bash
for method in raw fixed_adjacency hks; do
  for seed in 41 42 43 44 45; do
    python scripts/train.py --method "$method" --seed "$seed" --output outputs/three_way
  done
done
for method in raw fixed_adjacency hks; do
  for seed in 41 42 43 44 45; do
    python scripts/evaluate.py --run-dir "outputs/three_way/$method/seed_$seed" \
      --output "outputs/three_way/evaluation/${method}_${seed}" --device cpu
  done
done
python scripts/summarize.py outputs/three_way/evaluation/*/summary.json \
  --output outputs/three_way/evaluation/three_way_summary.csv
```

Keep defaults for every job: 2M transitions, four rollout workers, replay 1M,
batch 1024, 100-transition update interval, four update rounds per interval after
warmup, same optimizer settings, exploration schedule, 25-step horizon, and
100 checkpoints spaced every 20,000 transitions. With batch 1024 and four
workers, the first update occurs at transition 1100: 19,990 update intervals,
79,960 target-update rounds and 239,880 individual agent updates per run. Equal transitions do not imply
equal wall time: HKS has eigendecomposition overhead. Do not compare a tuned HKS
run against a default Adjacency run; freeze this protocol before final evaluation.

Selection uses mean return over seeds 91000000–91000099, independently for each
training seed; ties select the first chronological CSV row. Final evaluation
uses seeds 92000000–92000499, shared across methods and training seeds and disjoint
from selection. These are shared initial conditions, not identical trajectories:
actions and visited states differ between methods. Use the same evaluation
device for all methods. Do not tune using the final bank; the inherited trainer
also writes final-step evaluation on that bank, which should remain sealed until
all settings and selections are fixed. Use `--run-dir`, not manually chosen
`--checkpoint`, for the primary comparison.

## Recording and auditing results

Retain every run's protocol.json, resolved_config.json, training_counters.json,
tensor_shape_report.json, gpu_device_audit.json, versions.json (if emitted),
training.log, curve CSV and checkpoints. Record the Git commit, dirty diff,
Python/dependency versions, hardware, and elapsed training time with the campaign.
Check actual transitions, update counts and saved checkpoint lists, not just CLI
budgets. Do not interpret mocked dispatch tests as evidence that training ran.

The final evaluation directories contain episode-level data and summary.json
with the chosen checkpoint, selection score, bank and training seed. Preserve
checkpoint hashes alongside the final results. The summary script reports mean
and sample SD across five training seeds, not across 2,500 independent runs.
Coverage fraction = final covered-landmark count / 3. Retain count in the JSON;
use the fraction consistently in comparisons. Report return, coverage fraction,
full coverage rate, and collision step rate, including all five seed values.

Record missing/failed runs as such, never as zero. Aggregate only complete five-
seed sets. For HKS minus Adjacency, also report the five paired seed differences
and their mean/SD (positive is better except collision rate). Five seeds give
limited statistical precision; episode-level replication does not replace
independent training seeds. Predeclare return as the primary metric, and report
all secondary metrics even when they disagree. Do not select seeds or checkpoints
using final coverage. The existing aggregator checks evaluation banks/devices
and seed completeness, but does NOT certify that training budgets match: inspect
the saved training configurations/counters before including results.

## What a positive result can establish

If replicated HKS improvements over Adjacency occur under this protocol, they
support an advantage of this fixed multiscale diffusion descriptor over these
three focal affinity statistics, at matched actor capacity and training budget,
on this task and seed distribution. Raw gives a useful unaugmented baseline;
HKS-versus-Raw alone cannot isolate the effect from increased input capacity.

This does not establish that spectral computation itself is necessary or that
HKS beats all non-spectral graph descriptors. In particular:

* Adjacency summarizes only the focal node's three edges; HKS also reflects other
  agents' edges, landmark degrees and multi-hop structure. Both actors see the
  same raw observation, but their engineered summaries have different receptive
  fields. This is a major unresolved mechanism confound.
* Feature scales, correlations and BatchNorm running statistics differ, as do
  conditioning and optimization difficulty. Same dimension is not same content.
* A heat-kernel diagonal can be computed by matrix exponential or approximated
  by powers/message passing without explicitly computing an eigenbasis. This
  experiment tests a descriptor, not the necessity of a spectral implementation.
* Gaussian bandwidth, time scales, graph topology and past HKS tuning are fixed
  design choices. Historical tuning may favor HKS; five common seeds do not
  erase that selection bias. A held-out repeat would strengthen the evidence.
* Same seed does not force the same exploration/replay trajectory; Raw's different
  initialization shapes also consume random numbers differently. CUDA numerical
  nondeterminism and unequal compute cost remain possible differences.

Stronger attribution would require a matched whole-graph non-spectral summary
and/or finite-hop diffusion controls, with equal tuning budgets. Zero-padding
Raw to 21 inputs could separately test the capacity difference. These are future
controls, not part of this three-arm implementation or any claimed result.
