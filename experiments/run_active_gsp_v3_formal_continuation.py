import csv
import os
import subprocess
import sys
import time
from pathlib import Path


SHORT_PREFIX = "ag3f100kc"
MISSING_EXPERIMENTS = [
    "action_raw_potential_residual005",
    "v3_active_gsp_residual005",
]
SEEDS = [1, 2, 3]
RAW_SOURCE_DIRS = [
    Path("experiments/active_gsp_v3_residual_lowlr_100k_formal_20260703_230507_seed1_20260703_230510"),
    Path("experiments/active_gsp_v3_residual_lowlr_100k_formal_20260703_230507_seed2_20260703_230510"),
    Path("experiments/active_gsp_v3_residual_lowlr_100k_formal_20260703_230507_seed3_20260703_230510"),
]
STATUS_PATH = Path("experiments/active_gsp_v3_formal_continuation_status.md")


def shell_line(command):
    return subprocess.list2cmdline([str(part) for part in command])


def write_status(stage, state, details=None, artifacts=None):
    lines = [
        "# Active GSP v3 Formal Continuation Status",
        "",
        f"Updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"PID: `{os.getpid()}`",
        "",
        "## Purpose",
        "",
        "Continue the focused 100k formal run after the original long-prefix job completed only `raw_mlp`.",
        "",
        "## Cause Of Original Stop",
        "",
        "- `raw_mlp` completed for seeds `1,2,3`.",
        "- `action_raw_potential_residual005` stopped at `1-5/100000` for all seeds.",
        "- Root cause: TensorBoard nested event path length was 267 characters for the long-prefix action-prior model on Windows.",
        "- Continuation uses short prefix `ag3f100kc`; corresponding event paths are about 212 characters.",
        "",
        "## Queue",
        "",
        "- Missing variants: `action_raw_potential_residual005`, `v3_active_gsp_residual005`",
        "- Seeds: `1,2,3`",
        "- Launch mode: single process, sequential variants and seeds",
        "- Episodes: `100000`",
        "- Eval episodes after each training: `100`",
        "- Final fixed-test evaluation after review: `500` episodes/model",
        "- Learning rate: `0.001`",
        "- Rollout threads: `4`",
        "- Batch size: `64`",
        "",
        "## Current Stage",
        "",
        f"- Stage: `{stage}`",
        f"- State: `{state}`",
    ]
    if details:
        lines.extend(["", "## Details", "", *[f"- {detail}" for detail in details]])
    if artifacts:
        lines.extend(["", "## Artifacts", ""])
        for name, path in artifacts.items():
            lines.append(f"- {name}: `{path}`")
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_checked(stage, command, artifacts=None):
    write_status(stage, "running", details=[f"Command: `{shell_line(command)}`"], artifacts=artifacts)
    subprocess.run(command, check=True)
    write_status(stage, "completed", details=[f"Command: `{shell_line(command)}`"], artifacts=artifacts)


def read_results(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_raw_sources():
    missing = []
    for source in RAW_SOURCE_DIRS:
        csv_path = source / "results.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        rows = read_results(csv_path)
        raw_rows = [row for row in rows if row.get("experiment") == "raw_mlp"]
        if len(raw_rows) != 1:
            missing.append(f"{csv_path}: expected one raw_mlp row, found {len(raw_rows)}")
        model_path = Path(raw_rows[0]["run_dir"]) / "model.pt" if raw_rows else None
        if model_path is None or not model_path.exists():
            missing.append(f"{csv_path}: missing raw model {model_path}")
    if missing:
        raise RuntimeError("Raw source validation failed:\n" + "\n".join(missing))


def find_continuation_result_dir(start_time):
    candidates = [
        path
        for path in Path("experiments").glob(f"{SHORT_PREFIX}_*")
        if path.is_dir() and (path / "results.csv").exists() and path.stat().st_mtime >= start_time - 60
    ]
    if not candidates:
        raise FileNotFoundError(f"no continuation result directory found for prefix {SHORT_PREFIX}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def validate_continuation_results(result_dir):
    rows = read_results(result_dir / "results.csv")
    expected = {(experiment, str(seed)) for experiment in MISSING_EXPERIMENTS for seed in SEEDS}
    observed = {(row["experiment"], str(row["seed"])) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"missing continuation result rows: {missing}")
    missing_models = []
    for row in rows:
        if row["experiment"] in MISSING_EXPERIMENTS:
            model_path = Path(row["run_dir"]) / "model.pt"
            if not model_path.exists():
                missing_models.append(str(model_path))
    if missing_models:
        raise FileNotFoundError("missing continuation model files:\n" + "\n".join(missing_models))


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    review_dir = Path("experiments") / f"active_gsp_v3_residual_lowlr_100k_formal_review_cont_{stamp}"
    final_eval_dir = Path("experiments") / f"active_gsp_v3_final_eval_cont_{stamp}"
    figure_dir = Path("experiments") / f"paper_figures_final_cont_{stamp}"
    table_dir = Path("experiments") / f"paper_tables_final_cont_{stamp}"
    snippet_dir = Path("experiments") / f"paper_snippets_final_cont_{stamp}"

    artifacts = {
        "status": STATUS_PATH,
        "review": review_dir,
        "final_eval": final_eval_dir,
        "figures": figure_dir,
        "tables": table_dir,
        "snippets": snippet_dir,
    }
    write_status("startup", "running", artifacts=artifacts)
    validate_raw_sources()

    train_start = time.time()
    train_command = [
        sys.executable,
        "run_active_gsp_v3_experiments.py",
        "--episodes",
        "100000",
        "--eval_episodes",
        "100",
        "--episode_length",
        "25",
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--experiments",
        ",".join(MISSING_EXPERIMENTS),
        "--lr",
        "0.001",
        "--n_rollout_threads",
        "4",
        "--n_training_threads",
        "6",
        "--batch_size",
        "64",
        "--steps_per_update",
        "100",
        "--print_interval",
        "5000",
        "--prefix",
        SHORT_PREFIX,
    ]
    run_checked("train_missing_variants", train_command, artifacts=artifacts)
    continuation_dir = find_continuation_result_dir(train_start)
    validate_continuation_results(continuation_dir)
    artifacts["continuation_results"] = continuation_dir

    source_dirs = [*RAW_SOURCE_DIRS, continuation_dir]
    review_command = [
        sys.executable,
        "experiments/review_active_gsp_v3_results.py",
        *[str(path) for path in source_dirs],
        "--output-dir",
        str(review_dir),
    ]
    run_checked("formal_review", review_command, artifacts=artifacts)

    final_eval_command = [
        sys.executable,
        "experiments/final_eval_active_gsp_v3.py",
        str(review_dir),
        "--eval-episodes",
        "500",
        "--test-seed-base",
        "880000",
        "--output-dir",
        str(final_eval_dir),
    ]
    run_checked("final_eval", final_eval_command, artifacts=artifacts)

    figure_command = [
        sys.executable,
        "experiments/make_paper_figures.py",
        "--active-formal-review",
        str(review_dir),
        "--output-dir",
        str(figure_dir),
    ]
    run_checked("paper_figures", figure_command, artifacts=artifacts)

    table_command = [
        sys.executable,
        "experiments/make_paper_tables.py",
        "--active-formal-review",
        str(review_dir),
        "--final-eval-dir",
        str(final_eval_dir),
        "--output-dir",
        str(table_dir),
    ]
    run_checked("paper_tables", table_command, artifacts=artifacts)

    snippet_command = [
        sys.executable,
        "experiments/make_paper_snippets.py",
        "--formal-review",
        str(review_dir),
        "--final-eval-dir",
        str(final_eval_dir),
        "--output-dir",
        str(snippet_dir),
    ]
    run_checked("paper_snippets", snippet_command, artifacts=artifacts)
    write_status("complete", "completed", artifacts=artifacts)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("failed", "failed", details=[f"{type(exc).__name__}: {exc}"])
        raise
