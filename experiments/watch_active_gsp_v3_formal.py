import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def append_log(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--max-checks", type=int, default=240)
    parser.add_argument(
        "--log",
        default="experiments/training_logs/active_gsp_v3_formal_watch_20260703_230507.log",
    )
    parser.add_argument("--final-eval-episodes", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = REPO_ROOT / args.log
    append_log(log_path, f"\n# Watcher started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for check_i in range(1, args.max_checks + 1):
        command = [
            sys.executable,
            "experiments/monitor_active_gsp_v3_formal.py",
            "--review-if-complete",
            "--final-eval-if-complete",
            "--figures-if-complete",
            "--tables-if-complete",
            "--snippets-if-complete",
            "--final-eval-episodes",
            str(args.final_eval_episodes),
        ]
        append_log(log_path, f"\n## Check {check_i}/{args.max_checks}: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        append_log(log_path, completed.stdout)
        if "Review: `" in completed.stdout:
            append_log(log_path, "Watcher completed final review/eval/figures.")
            return 0
        if completed.returncode != 0:
            append_log(log_path, f"Monitor returned nonzero status: {completed.returncode}")
        time.sleep(args.interval_sec)
    append_log(log_path, "Watcher reached max checks without completion.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
