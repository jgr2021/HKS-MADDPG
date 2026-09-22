"""Add predeclared attempts 4/5 only where three eligible payloads are missing."""

import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from experiments.run_cardinality_distillations import OUTPUT_ROOT, config_for


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows):
    with (OUTPUT_ROOT / "supplement_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "n_agents", "replicate", "status", "eligible",
            "validation_accuracy", "wall_time_sec", "output_dir", "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "supplement_status.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def eligible_count(rows, n_agents):
    return sum(
        int(row["n_agents"]) == n_agents
        and row["status"] == "completed"
        and str(row["eligible"]).lower() == "true"
        for row in rows
    )


def main():
    if not (OUTPUT_ROOT / "status.csv").exists():
        raise FileNotFoundError("Primary cardinality distillation batch is incomplete")
    primary = read_rows(OUTPUT_ROOT / "status.csv")
    supplement = []
    for n_agents in (3, 4, 5, 6):
        for replicate in (4, 5):
            combined = [*primary, *supplement]
            if eligible_count(combined, n_agents) >= 3:
                break
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
            started = time.perf_counter()
            try:
                with (OUTPUT_ROOT / f"n{n_agents}_actor{replicate}.log").open(
                    "w", encoding="utf-8"
                ) as log:
                    subprocess.run(
                        command, check=True, stdout=log,
                        stderr=subprocess.STDOUT
                    )
                metadata = json.loads(
                    (output_dir / "metadata.json").read_text(encoding="utf-8")
                )
                row = {
                    "n_agents": n_agents,
                    "replicate": replicate,
                    "status": "completed",
                    "eligible": bool(metadata["offline_gate"]["eligible"]),
                    "validation_accuracy": metadata["final"]["validation_accuracy"],
                    "wall_time_sec": metadata["wall_time_sec"],
                    "output_dir": str(output_dir),
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "n_agents": n_agents,
                    "replicate": replicate,
                    "status": "failed",
                    "eligible": False,
                    "validation_accuracy": "",
                    "wall_time_sec": time.perf_counter() - started,
                    "output_dir": str(output_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (OUTPUT_ROOT / f"n{n_agents}_actor{replicate}.failure.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            supplement.append(row)
            write_rows(supplement)
            print(json.dumps(row), flush=True)
    combined = [*primary, *supplement]
    deficits = {
        str(n_agents): 3 - eligible_count(combined, n_agents)
        for n_agents in (3, 4, 5, 6)
    }
    (OUTPUT_ROOT / "eligibility_summary.json").write_text(json.dumps({
        "target_eligible_per_cardinality": 3,
        "max_attempts_per_cardinality": 5,
        "deficits_after_supplement": deficits,
    }, indent=2) + "\n", encoding="utf-8")
    if any(value > 0 for value in deficits.values()):
        raise RuntimeError(f"Offline eligibility deficits remain: {deficits}")


if __name__ == "__main__":
    main()
