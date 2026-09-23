# HKS-MADDPG

Code and manuscript for **Heat-Kernel Structural Features for Cooperative Policy Learning**.

## Paper

- [Read the five-page paper](paper/icassp_hks/main.pdf)
- [LaTeX source](paper/icassp_hks/main.tex)

The paper studies cooperative navigation in MPE `simple_spread` with three agents and three landmarks. Its experimental comparison is **MADDPG with raw observations vs. MADDPG with HKS features**, trained with seeds 41–45.

## Method in the paper

Each actor reconstructs the agent and landmark positions from its own 18-dimensional observation. The six nodes form a complete agent–landmark bipartite graph. For an agent–landmark pair of nodes (u,v), the edge weight is

\[
W_{uv}=\exp\!\left(-\frac{\|x_u-x_v\|_2^2}{2(0.6)^2}\right),
\]

and all other entries of (W) are zero. The normalized Laplacian (L=I-D^{-1/2}WD^{-1/2}) gives three focal heat-kernel values, ([e^{-\tau L}]_{ii}) at \(\tau\in\{0.5,1,2\}\). These values are appended to the raw observation, giving the actor a 21-dimensional input. The centralized critic continues to use raw joint observations and actions.

## Evaluation reported in the paper

Each seed is trained for two million environment transitions. The checkpoint with the highest mean return on a common 100-episode selection bank is evaluated on a separate common 500-episode bank. The paper reports means and sample standard deviations over all five seeds, along with paired uncertainty. HKS has lower mean assignment distance and higher mean coverage-radius AUC, while collision-step frequency is higher. The paired return confidence interval includes zero.

The repository also contains code for other experimental configurations. Their results are outside this paper's Raw-vs-HKS comparison.

## Repository layout

| Path | Contents |
| --- | --- |
| `paper/icassp_hks/` | Manuscript PDF, LaTeX source, bibliography, and included figures |
| `algorithms/` | Multi-agent learning implementations |
| `experiments/` | Training, evaluation, and analysis scripts |
| `utils/` | Feature construction and policy modules |
| `multiagent/` | Particle environment |
| `tests/` | Code checks |

To build the manuscript, run `latexmk -pdf main.tex` inside `paper/icassp_hks/`. The LaTeX style, bibliography style, and figure files are included in that directory.
