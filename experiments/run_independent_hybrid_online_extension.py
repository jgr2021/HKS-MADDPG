"""Run the preregistered online protocol for new folds 24 and 25 only."""

import subprocess
import sys


REPLICATES = (
    "fold24_t104_s104",
    "fold25_t105_s105",
)


def main():
    for replicate in REPLICATES:
        subprocess.run(
            [
                sys.executable,
                "experiments/run_independent_hybrid_online_replicate.py",
                "--replicate",
                replicate,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
