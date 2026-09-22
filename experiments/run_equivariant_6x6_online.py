"""Launch the three locked 6x6 online payload replications."""

import subprocess
import sys


def main():
    for payload_seed in (1, 2, 3):
        subprocess.run(
            [
                sys.executable,
                "experiments/run_equivariant_6x6_online_replicate.py",
                "--payload-seed",
                str(payload_seed),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
