# Previous repository layout

The complete repository before the directory cleanup is preserved at:

- Branch: [`archive/research-20260927`](https://github.com/jgr2021/HKS-MADDPG/tree/archive/research-20260927)
- Commit: `8590acb` (`Remove manuscript files and keep repository code-only`)

The main branch focuses on Raw/HKS MADDPG. Dated queue supervisors, interrupted
run recovery, exploratory graph/operator/control scripts, MAPPO experiments,
and their tests are retained in the archive branch. Local research folders and
their experimental outputs are separate from this public checkout.

To inspect the old code without replacing the current checkout:

```bash
git worktree add ../HKS-MADDPG-research origin/archive/research-20260927
```

The runtime modules needed by the public commands moved from the old root into
`code/` with their contents unchanged. `code/README.md` identifies the relevant
symbols, including the historical names behind the public `hks` option.

This cleanup adds public command wrappers and documentation. It does not
recompute results, change model formulas, or upload manuscript material.
