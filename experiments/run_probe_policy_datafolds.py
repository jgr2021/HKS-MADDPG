"""Leave-one-source-policy-seed-out confirmation for Hungarian probes."""

import subprocess
import sys


def main():
    for heldout_seed in (21, 22):  # fold 23 is the already completed confirmation
        subprocess.run([
            sys.executable, "experiments/evaluate_supervised_probe_policies.py",
            "--output-dir", f"experiments/supervised_probe_policy_datafold_hold{heldout_seed}_20260712",
            "--episodes", "500", "--horizon", "25", "--seed-base", "2300000",
            "--model-seed", "1", "--heldout-seed", str(heldout_seed),
            "--representations", "raw_action_geom,raw_action_active",
            "--objectives", "hungarian",
        ], check=True)


if __name__ == "__main__": main()
