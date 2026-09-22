"""Create new full-pipeline distilled actors from source folds 24 and 25."""

import subprocess
import sys


DATASET = (
    "experiments/counterfactual_dataset_fivefold_20260811/"
    "counterfactual_dataset.npz"
)
REPLICATES = (
    {
        "name": "fold24_t104_s104",
        "heldout": 24,
        "teacher_seed": 104,
        "student_seed": 104,
        "train_seed_base": 45_000_000,
        "dagger_seed_base": 45_100_000,
        "eval_seed_base": 45_200_000,
    },
    {
        "name": "fold25_t105_s105",
        "heldout": 25,
        "teacher_seed": 105,
        "student_seed": 105,
        "train_seed_base": 45_300_000,
        "dagger_seed_base": 45_400_000,
        "eval_seed_base": 45_500_000,
    },
)


def main():
    for replicate in REPLICATES:
        subprocess.run([
            sys.executable,
            "experiments/distill_hybrid_counterfactual_actor.py",
            "--dataset", DATASET,
            "--output-dir",
            (
                "experiments/independent_hybrid_distillation_"
                f"{replicate['name']}_20260811"
            ),
            "--train-episodes", "300",
            "--dagger-episodes", "200",
            "--eval-episodes", "500",
            "--teacher-seed", str(replicate["teacher_seed"]),
            "--student-seed", str(replicate["student_seed"]),
            "--heldout-source-seed", str(replicate["heldout"]),
            "--train-seed-base", str(replicate["train_seed_base"]),
            "--dagger-seed-base", str(replicate["dagger_seed_base"]),
            "--eval-seed-base", str(replicate["eval_seed_base"]),
        ], check=True)


if __name__ == "__main__":
    main()
