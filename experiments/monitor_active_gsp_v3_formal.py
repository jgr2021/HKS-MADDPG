import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_EXPERIMENTS = [
    "raw_mlp",
    "action_raw_potential_residual005",
    "v3_active_gsp_residual005",
]
EPISODE_RE = re.compile(r"Episodes\s+(\d+)-(\d+)\s+of\s+(\d+)")
TRAINING_RE = re.compile(r"=== Training ([^:]+):")


def process_alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_launch_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_tail(path, max_lines=200):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def parse_log(path):
    current_experiment = None
    latest_progress = None
    for line in read_tail(path):
        match = TRAINING_RE.search(line)
        if match:
            current_experiment = match.group(1)
        match = EPISODE_RE.search(line)
        if match:
            latest_progress = {
                "start": int(match.group(1)),
                "end": int(match.group(2)),
                "total": int(match.group(3)),
            }
    return current_experiment, latest_progress


def find_result_dir(prefix):
    matches = [path for path in Path("experiments").glob(f"{prefix}_*") if path.is_dir()]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def read_results_rows(result_dir):
    if result_dir is None:
        return []
    csv_path = result_dir / "results.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarize(launch_rows):
    rows = []
    source_dirs = []
    for launch in launch_rows:
        seed = int(launch["seed"])
        pid = int(launch["pid"])
        stdout = Path(launch["stdout"])
        stderr = Path(launch["stderr"])
        prefix = launch["prefix"]
        current_experiment, progress = parse_log(stdout)
        result_dir = find_result_dir(prefix)
        result_rows = read_results_rows(result_dir)
        completed = sorted({row["experiment"] for row in result_rows})
        is_complete = sorted(completed) == sorted(EXPECTED_EXPERIMENTS)
        if result_dir is not None:
            source_dirs.append(result_dir)
        rows.append(
            {
                "seed": seed,
                "pid": pid,
                "alive": process_alive(pid),
                "current_experiment": current_experiment or "",
                "progress": progress,
                "result_dir": result_dir,
                "completed_experiments": completed,
                "complete": is_complete,
                "stdout": stdout,
                "stderr": stderr,
                "stderr_bytes": stderr.stat().st_size if stderr.exists() else None,
            }
        )
    return rows, source_dirs


def print_status(rows):
    print("# Active GSP v3 Formal Monitor")
    print()
    print(f"Checked: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    print()
    print("| seed | pid | alive | current | progress | completed rows | stderr bytes |")
    print("| ---: | ---: | --- | --- | --- | --- | ---: |")
    for row in rows:
        progress = ""
        if row["progress"]:
            progress = (
                f"{row['progress']['start']}-{row['progress']['end']}/"
                f"{row['progress']['total']}"
            )
        completed = ",".join(row["completed_experiments"])
        print(
            f"| {row['seed']} | {row['pid']} | {row['alive']} | "
            f"{row['current_experiment']} | {progress} | {completed} | "
            f"{row['stderr_bytes']} |"
        )
    print()
    for row in rows:
        if row["result_dir"]:
            print(f"- seed {row['seed']} result dir: `{row['result_dir']}`")


def all_complete(rows):
    return bool(rows) and all(row["complete"] for row in rows)


def run_command(command):
    print()
    print("Running:")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def maybe_review_and_figures(args, rows, source_dirs):
    if not all_complete(rows):
        return None
    if not args.review_if_complete:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.review_output_dir) if args.review_output_dir else Path("experiments") / f"active_gsp_v3_residual_lowlr_100k_formal_review_{stamp}"
    command = [
        sys.executable,
        "experiments/review_active_gsp_v3_results.py",
        *[str(path) for path in source_dirs],
        "--output-dir",
        str(output_dir),
    ]
    run_command(command)
    final_eval_output_dir = None
    if args.final_eval_if_complete:
        final_eval_output_dir = (
            Path(args.final_eval_output_dir)
            if args.final_eval_output_dir
            else Path("experiments") / f"active_gsp_v3_final_eval_{stamp}"
        )
        final_eval_command = [
            sys.executable,
            "experiments/final_eval_active_gsp_v3.py",
            str(output_dir),
            "--eval-episodes",
            str(args.final_eval_episodes),
            "--test-seed-base",
            str(args.test_seed_base),
            "--output-dir",
            str(final_eval_output_dir),
        ]
        run_command(final_eval_command)
    if args.figures_if_complete:
        run_command(
            [
                sys.executable,
                "experiments/make_paper_figures.py",
                "--active-formal-review",
                str(output_dir),
            ]
        )
    if args.tables_if_complete:
        table_command = [
            sys.executable,
            "experiments/make_paper_tables.py",
            "--active-formal-review",
            str(output_dir),
        ]
        if final_eval_output_dir is not None:
            table_command.extend(["--final-eval-dir", str(final_eval_output_dir)])
        run_command(table_command)
    if args.snippets_if_complete:
        snippet_command = [
            sys.executable,
            "experiments/make_paper_snippets.py",
            "--formal-review",
            str(output_dir),
        ]
        if final_eval_output_dir is not None:
            snippet_command.extend(["--final-eval-dir", str(final_eval_output_dir)])
        run_command(snippet_command)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch-csv",
        default="experiments/active_gsp_v3_residual_lowlr_100k_formal_20260703_230507_launch.csv",
    )
    parser.add_argument("--review-if-complete", action="store_true")
    parser.add_argument("--figures-if-complete", action="store_true")
    parser.add_argument("--tables-if-complete", action="store_true")
    parser.add_argument("--snippets-if-complete", action="store_true")
    parser.add_argument("--final-eval-if-complete", action="store_true")
    parser.add_argument("--final-eval-episodes", type=int, default=500)
    parser.add_argument("--final-eval-output-dir", default=None)
    parser.add_argument("--test-seed-base", type=int, default=880000)
    parser.add_argument("--review-output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    launch_rows = read_launch_csv(args.launch_csv)
    rows, source_dirs = summarize(launch_rows)
    print_status(rows)
    output_dir = maybe_review_and_figures(args, rows, source_dirs)
    if output_dir is not None:
        print()
        print(f"Review: `{output_dir}`")


if __name__ == "__main__":
    main()
