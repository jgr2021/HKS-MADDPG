# Verification record — 2026-10-22

Base: c2cb3676332bfb082589288afc138b3fa97114a0.
Command: `python -m unittest discover -s tests -v`.
Outcome: **13 tests passed**, 27.920 seconds on this run.
`git diff --check` and Python compilation also passed.

Environment: Python 3.12, PyTorch 2.5.1+cu124, NumPy 1.26.4, Gym 0.23.1;
CUDA unavailable. This is a CPU verification environment, not a reproduction
of the documented Python 3.9 / NumPy 1.23.5 / Gym 0.21 training environment.
The bundled scenario loader was migrated from removed `imp.load_source` to
`importlib` to run on Python 3.12. Scenario dynamics/rewards were not edited.
The original pinned environment has not been re-tested in this session.

Passed checks:

* Both augmented actors produce 21D input and preserve the raw 18D prefix.
* Actual graph calls yield identical Gaussian bipartite weights; adjacency
  values equal focal mean/max/population SD; landmark permutation invariance.
* HKS agrees with a normalized-Laplacian matrix exponential reference.
* Instantiated parameter counts: Raw 5,737, Adjacency/HKS 5,935 per actor;
  8,843 per critic; critic input 69. Augmented initial state dictionaries match.
* All three actor checkpoints save/load with matching actions.
* Adjacency evaluation is repeatable, and the actual evaluation CLI loads an
  untrained adjacency checkpoint in a fresh process and completes two episodes.
* Mocked public training dispatch for all three methods × seeds 41–45 yields
  identical normalized budgets, checkpoint schedules, and 100/500 episode banks.
* Selection maximizes return; ties retain earlier CSV row; portable checkpoint
  paths work; overlapping selection/final banks are rejected before evaluation.
* Three-method aggregation accepts complete seed sets, uses sample SD, and
  rejects duplicate/missing seeds, inconsistent banks and non-finite metrics.

Not run: CUDA optimization, end-to-end training, any of the fifteen 2M-transition
jobs, or evaluation of trained policies. The mocked dispatch prints training
messages but never calls the real trainer. Random-weight smoke evaluations test
execution only: their scores are not experimental performance results and are
not reported. No empirical claim that HKS improves performance is made.

Full commands, fairness caveats and result recording rules are in
[THREE_WAY_CONTROL.md](THREE_WAY_CONTROL.md).
