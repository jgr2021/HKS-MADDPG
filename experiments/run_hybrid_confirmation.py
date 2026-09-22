"""Three-initialization confirmation of the Active hybrid controller."""

import subprocess
import sys


def main():
    for model_seed in (1, 2, 3):
        subprocess.run([
            sys.executable, "experiments/evaluate_hybrid_counterfactual_controller.py",
            "--output-dir", f"experiments/hybrid_counterfactual_confirmation_seed{model_seed}_20260712",
            "--episodes", "500", "--horizon", "25", "--seed-base", "2700000",
            "--model-seed", str(model_seed), "--representations", "raw_action_active",
        ], check=True)


if __name__ == "__main__": main()
