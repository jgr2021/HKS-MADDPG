"""Three-initialization confirmation for reward-counterfactual local policies."""

import subprocess
import sys


def main():
    for model_seed in (1, 2, 3):
        subprocess.run([
            sys.executable, "experiments/probe_and_evaluate_reward_value.py",
            "--output-dir", f"experiments/reward_probe_confirmation_seed{model_seed}_20260712",
            "--episodes", "500", "--horizon", "25", "--seed-base", "2200000",
            "--representations", "raw_action_geom,raw_action_active",
            "--probe-model-seeds", str(model_seed),
            "--deployment-model-seed", str(model_seed),
        ], check=True)


if __name__ == "__main__": main()
