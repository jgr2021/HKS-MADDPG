"""Run fixed-state cardinality policy audits with bounded concurrency."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


def only_run_dir(n_agents, replicate):
    parent = Path(
        f"experiments/cardinality_online_n{n_agents}_actor"
        f"{replicate}_20260811"
    )
    runs = sorted(parent.glob("run_*"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run under {parent}, got {runs}")
    return runs[0]


def write_status(output, rows):
    rows = sorted(rows, key=lambda row: (row["n_agents"], row["replicate"]))
    with (output / "status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "status.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def run_one(n_agents, replicate, output, output_name, quartile_mode):
    started = time.perf_counter()
    status, error = "pending", ""
    try:
        run_dir = only_run_dir(n_agents, replicate)
        command = [
            sys.executable,
            "experiments/analyze_cardinality_policy_drift.py",
            "--run-dir", str(run_dir),
            "--episodes", "100",
            "--horizon", "25",
            "--seed-base",
            str(30_000_000 + n_agents * 1_000_000 + replicate * 100_000),
            "--output-name", output_name,
            "--quartile-mode", quartile_mode,
        ]
        with (output / f"n{n_agents}_actor{replicate}.log").open(
            "w", encoding="utf-8"
        ) as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT
            )
        status = "completed"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        (output / f"n{n_agents}_actor{replicate}.failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    return {
        "n_agents": n_agents,
        "replicate": replicate,
        "status": status,
        "wall_time_sec": time.perf_counter() - started,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiments/cardinality_policy_drift_batch_20260811",
    )
    parser.add_argument(
        "--analysis-output-name", default="cardinality_policy_drift"
    )
    parser.add_argument(
        "--quartile-mode", choices=("threshold", "rank"),
        default="threshold",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    jobs = [
        (n_agents, replicate)
        for n_agents in (3, 4, 5, 6)
        for replicate in (1, 2, 3)
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                run_one, *job, output,
                args.analysis_output_name, args.quartile_mode,
            ): job for job in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_status(output, rows)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
