"""Launch the amended intention-to-treat cardinality online batch."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


DISTILL_ROOT = Path("experiments/cardinality_distillations_20260811")
OUTPUT_ROOT = Path("experiments/cardinality_online_batch_20260811")


def write_rows(rows):
    rows = sorted(rows, key=lambda row: (row["n_agents"], row["replicate"]))
    with (OUTPUT_ROOT / "status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "status.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


def run_one(n_agents, replicate):
    actor_dir = DISTILL_ROOT / f"n{n_agents}_actor{replicate}"
    metadata_path = actor_dir / "metadata.json"
    eligible = False
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        eligible = bool(metadata["offline_gate"]["eligible"])
    command = [
        sys.executable,
        "experiments/run_cardinality_online_replicate.py",
        "--n-agents", str(n_agents),
        "--replicate", str(replicate),
        "--allow-ineligible",
    ]
    started = time.perf_counter()
    status, error = "pending", ""
    try:
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        with (OUTPUT_ROOT / f"n{n_agents}_actor{replicate}.log").open(
            "w", encoding="utf-8"
        ) as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT
            )
        status = "completed"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        (OUTPUT_ROOT / f"n{n_agents}_actor{replicate}.failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    return {
        "n_agents": n_agents,
        "replicate": replicate,
        "eligible": eligible,
        "status": status,
        "wall_time_sec": time.perf_counter() - started,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("workers must be in [1, 3]")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "launch_command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    selected_payloads = {
        str(n_agents): [1, 2, 3] for n_agents in (3, 4, 5, 6)
    }
    (OUTPUT_ROOT / "payload_selection.json").write_text(
        json.dumps(selected_payloads, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "scheduling_metadata.json").write_text(
        json.dumps({
            "workers": args.workers,
            "scheduling_note": (
                "Independent processes only; seeds and statistical protocol "
                "are unchanged."
            ),
        }, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    jobs = [
        (n_agents, replicate)
        for n_agents in (3, 4, 5, 6)
        for replicate in selected_payloads[str(n_agents)]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, n_agents, replicate): (
                n_agents, replicate
            )
            for n_agents, replicate in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_rows(rows)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
