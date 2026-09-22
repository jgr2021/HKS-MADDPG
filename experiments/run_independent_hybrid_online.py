"""Launch online tests only for independently distilled actors that pass."""

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_independent_hybrid_distillations import REPLICATES


DECISIONS = Path(
    "experiments/independent_hybrid_distillation_analysis_20260716/"
    "decisions.csv"
)


def main():
    if not DECISIONS.exists():
        raise FileNotFoundError(
            f"Run the independent-distillation analysis first: {DECISIONS}"
        )
    with DECISIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = {
        row["replicate"]
        for row in rows
        if row["eligible"].strip().lower() == "true"
    }
    expected = {replicate["name"] for replicate in REPLICATES}
    if eligible != expected:
        raise RuntimeError(
            "All three independently distilled actors must pass the frozen "
            f"offline gate before online testing; eligible={sorted(eligible)}"
        )

    for replicate in REPLICATES:
        name = replicate["name"]
        subprocess.run(
            [
                sys.executable,
                "experiments/run_independent_hybrid_online_replicate.py",
                "--replicate",
                name,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
