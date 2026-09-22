"""Run three locked supervised-probe policy initializations sequentially."""

import subprocess
import sys


def main():
    for model_seed in (1, 2, 3):
        subprocess.run([
            sys.executable,
            "experiments/evaluate_supervised_probe_policies.py",
            "--output-dir", f"experiments/supervised_probe_policy_confirmation_seed{model_seed}_20260712",
            "--episodes", "500",
            "--horizon", "25",
            "--epochs", "50",
            "--threads", "6",
            "--model-seed", str(model_seed),
            "--seed-base", "1700000",
            "--representations", "raw_action_geom,raw_action_active,raw_action_vector_gsp",
            "--objectives", "hungarian,hungarian_safe",
        ], check=True)


if __name__ == "__main__":
    main()
