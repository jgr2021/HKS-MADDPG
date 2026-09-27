# Implementation map

Use the commands in [`../scripts/`](../scripts/) from the repository root.
They add this directory to the Python import path and register the HKS actor.

| File / symbol | Role |
| --- | --- |
| `utils/adaptive_hks_controls.py::FixedHKSPolicy` | The paper's HKS actor; registered as `confirm_fixed_hks` |
| `utils/catalog76_policies.py::graph` | Gaussian agent–landmark graph; the HKS recipe is `{}` |
| `utils/catalog76_policies.py::FeaturePolicy` | Concatenates the descriptor with the raw observation |
| `utils/exploration_topology_policies.py::local_positions` | Reconstructs focal-agent-relative node positions |
| `utils/exploration_topology_policies.py::focal_hks` | Normalized Laplacian eigendecomposition and focal heat-kernel diagonal |
| `utils/networks.py::MLPNetwork` | Actor / critic MLP |
| `algorithms/maddpg.py::MADDPG` | Centralized critics, decentralized actors, updates, and checkpoint loading |
| `experiments/run_passive_gsp_topology_pilot.py` | Existing transition-counted trainer and output writers |
| `experiments/run_d4_actor_gpu_screen.py` | Existing evaluation device adapter and training defaults |
| `run_vector_signal_gsp_experiment.py::evaluate_model` | Deterministic episode evaluation and coverage metrics |
| `utils/cuda_protocol_audit.py` | Checks optimization tensors, gradients, and device records |
| `multiagent/scenarios/simple_spread.py` | Task observations, rewards, and world initialization |

The implementation files were moved without changing their contents. Their
historical module names remain so the model registry and saved parameter keys
stay compatible. Some shared modules contain additional policy classes; the
public commands select only `raw` and `hks`.

`main.py` is retained because it supplies vectorized environment creation to the
trainer. It is not the public HKS training command: its original command-line
parser does not register the HKS actor.

| Public name | Historical method key | Checkpoint actor key | Actor input |
| --- | --- | --- | --- |
| `raw` | `raw` | `mlp` | 18D |
| `hks` | `fixed_hks` | `confirm_fixed_hks` | 21D |

No MAPPO or adaptive-bandwidth configuration is dispatched by the public tools.
