"""Run the locked three-payload-per-cardinality offline pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


OUTPUT_ROOT = Path("experiments/cardinality_distillations_20260811")
CARDINALITIES = (3, 4, 5, 6)
REPLICATES = (1, 2, 3)


def config_for(n_agents, replicate):
    offset = n_agents * 1_000_000 + replicate * 100_000
    return {
        "n_agents": n_agents,
        "replicate": replicate,
        "student_seed": 3000 + n_agents * 10 + replicate,
        "train_seed_base": 10_000_000 + offset,
        "dagger_seed_base": 10_030_000 + offset,
        "eval_seed_base": 10_060_000 + offset,
    }


def write_status(rows):
    with (OUTPUT_ROOT / "status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "n_agents", "replicate", "status", "eligible",
            "validation_accuracy", "wall_time_sec", "output_dir", "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "status.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def row_from_metadata(config, output_dir, status="completed", error=""):
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    return {
        "n_agents": config["n_agents"],
        "replicate": config["replicate"],
        "status": status,
        "eligible": bool(metadata["offline_gate"]["eligible"]),
        "validation_accuracy": metadata["final"]["validation_accuracy"],
        "wall_time_sec": metadata["wall_time_sec"],
        "output_dir": str(output_dir),
        "error": error,
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "launch_command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "protocol.json").write_text(json.dumps({
        "locked_date": "2026-08-11",
        "cardinalities": list(CARDINALITIES),
        "replicates": list(REPLICATES),
        "train_episodes": 300,
        "dagger_episodes": 200,
        "eval_episodes": 300,
        "horizon": 25,
        "epochs_max_per_round": 50,
        "actor": "equivariant_matching_safety_nxn_sinkhorn8",
        "teacher": "independent_safe_matching_teacher",
        "configs": [
            config_for(n_agents, replicate)
            for n_agents in CARDINALITIES for replicate in REPLICATES
        ],
    }, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        stdout=(OUTPUT_ROOT / "package_versions.txt").open("w", encoding="utf-8"),
    )
    rows = []
    for n_agents in CARDINALITIES:
        for replicate in REPLICATES:
            config = config_for(n_agents, replicate)
            output_dir = OUTPUT_ROOT / f"n{n_agents}_actor{replicate}"
            command = [
                sys.executable,
                "experiments/distill_cardinality_actor.py",
                "--n-agents", str(n_agents),
                "--output-dir", str(output_dir),
                "--train-episodes", "300",
                "--dagger-episodes", "200",
                "--eval-episodes", "300",
                "--horizon", "25",
                "--epochs", "50",
                "--student-seed", str(config["student_seed"]),
                "--train-seed-base", str(config["train_seed_base"]),
                "--dagger-seed-base", str(config["dagger_seed_base"]),
                "--eval-seed-base", str(config["eval_seed_base"]),
            ]
            log_path = OUTPUT_ROOT / f"n{n_agents}_actor{replicate}.log"
            started = time.perf_counter()
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
                rows.append(row_from_metadata(config, output_dir))
            except Exception as exc:
                failure = {
                    "n_agents": n_agents,
                    "replicate": replicate,
                    "status": "failed",
                    "eligible": False,
                    "validation_accuracy": "",
                    "wall_time_sec": time.perf_counter() - started,
                    "output_dir": str(output_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                rows.append(failure)
                (OUTPUT_ROOT / f"n{n_agents}_actor{replicate}.failure.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            write_status(rows)
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
