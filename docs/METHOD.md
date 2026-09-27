# Graph construction and HKS actor features

## Task and local observation

The task has three agents and three landmarks. Each actor receives one
18-dimensional MPE observation:

| Slice | Information |
| --- | --- |
| `0:2` | Focal agent velocity |
| `2:4` | Focal agent absolute position |
| `4:10` | Three landmark positions relative to the focal agent |
| `10:14` | Two other agent positions relative to the focal agent |
| `14:18` | Communication entries from the other agents |

For graph construction, node 0 is the focal agent at the origin, nodes 1–2 are
the other agents, and nodes 3–5 are the landmarks. All graph positions come from
that actor's local observation.

## Weighted agent–landmark graph

Connect each agent to each landmark with an undirected Gaussian-weighted edge:

$$W_{uv}=B_{uv}\exp\!\left(-\frac{\|x_u-x_v\|^2}{2\sigma^2}\right),\qquad \sigma=0.6.$$

Here $B_{uv}=1$ for agent–landmark pairs and zero otherwise. There are no
agent–agent edges, landmark–landmark edges, or self-loops. Agents relate through
their connections to shared landmarks.

## Focal heat kernel signature

With $D_{uu}=\sum_v W_{uv}$, form the normalized Laplacian
$L=I-D^{-1/2}WD^{-1/2}$. For eigenpairs $(\lambda_k,\phi_k)$, the focal descriptor is

$$h_i(t)=\sum_k e^{-t\lambda_k}\phi_k(i)^2,\qquad t\in\{0.5,1,2\}.$$

The implementation clamps tiny degrees to $10^{-12}$, symmetrizes the
Laplacian, and clamps numerical eigenvalues to $[0,2]$ before exponentiation.
The three descriptor values describe the focal node at three diffusion scales.

## Integration with MADDPG

Concatenate $[o_i,h_i(0.5),h_i(1),h_i(2)]$ to form the 21D actor input.
HKS is computed without gradients from the raw observation; the policy MLP is
trained by MADDPG. Raw MADDPG uses the same actor architecture with an 18D input.
Both use a centralized critic with 69 inputs: three 18D observations plus three
five-dimensional action vectors. Replay stores the original observations.

HKS is an existing descriptor. This implementation uses it with the task-aware
agent–landmark graph and actor-side observation augmentation described above.
See the [implementation map](../code/README.md) for the corresponding functions.
