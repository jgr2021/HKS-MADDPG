"""Create three independent hybrid teacher/student actor payloads.

Each replicate changes the held-out source-policy fold and both network/training
seeds.  Settings are fixed before the resulting actors are evaluated online.
"""

import subprocess
import sys


REPLICATES = (
    {
        "name": "fold21_t101_s101",
        "heldout": 21,
        "teacher_seed": 101,
        "student_seed": 101,
        "train_seed_base": 6_100_000,
        "dagger_seed_base": 6_110_000,
        "eval_seed_base": 6_120_000,
    },
    {
        "name": "fold22_t102_s102",
        "heldout": 22,
        "teacher_seed": 102,
        "student_seed": 102,
        "train_seed_base": 6_200_000,
        "dagger_seed_base": 6_210_000,
        "eval_seed_base": 6_220_000,
    },
    {
        "name": "fold23_t103_s103",
        "heldout": 23,
        "teacher_seed": 103,
        "student_seed": 103,
        "train_seed_base": 6_300_000,
        "dagger_seed_base": 6_310_000,
        "eval_seed_base": 6_320_000,
    },
)


def main():
    for replicate in REPLICATES:
        subprocess.run([
            sys.executable,
            "experiments/distill_hybrid_counterfactual_actor.py",
            "--output-dir",
            f"experiments/independent_hybrid_distillation_{replicate['name']}_20260716",
            "--train-episodes",
            "300",
            "--dagger-episodes",
            "200",
            "--eval-episodes",
            "500",
            "--teacher-seed",
            str(replicate["teacher_seed"]),
            "--student-seed",
            str(replicate["student_seed"]),
            "--heldout-source-seed",
            str(replicate["heldout"]),
            "--train-seed-base",
            str(replicate["train_seed_base"]),
            "--dagger-seed-base",
            str(replicate["dagger_seed_base"]),
            "--eval-seed-base",
            str(replicate["eval_seed_base"]),
        ], check=True)


if __name__ == "__main__":
    main()
