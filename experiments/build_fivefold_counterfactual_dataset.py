"""Collect source folds 24/25 and combine them with immutable folds 21--23."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ORIGINAL = Path(
    "experiments/reward_probe_dataset_20260712/counterfactual_dataset.npz"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def only_run(parent):
    runs = sorted(parent.glob("run_*"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected one source-policy run, got {runs}")
    return runs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-parent",
        type=Path,
        default=Path("experiments/independent_source_policy_extension_20260811d"),
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=Path("experiments/counterfactual_dataset_extension_20260811"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/counterfactual_dataset_fivefold_20260811"),
    )
    args = parser.parse_args()
    source_run = only_run(args.source_parent)
    command = [
        sys.executable,
        "experiments/probe_action_value_representations.py",
        "--screen-root", str(source_run),
        "--output-dir", str(args.extension_dir),
        "--train-seeds", "24,25",
        "--episodes", "100",
        "--horizon", "25",
        "--eval-seed-base", "44200000",
        "--collect-only",
    ]
    subprocess.run(command, check=True)
    extension = args.extension_dir / "counterfactual_dataset.npz"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(ORIGINAL) as loaded:
        original_data = {key: loaded[key] for key in loaded.files}
    with np.load(extension) as loaded:
        extension_data = {key: loaded[key] for key in loaded.files}
    if set(original_data) != set(extension_data):
        raise RuntimeError("Dataset schemas differ between old and new folds")
    group_offset = int(original_data["group"].max()) + 1
    extension_data["group"] = extension_data["group"] + group_offset
    combined = {
        key: np.concatenate([original_data[key], extension_data[key]], axis=0)
        for key in original_data
    }
    unique_seeds = sorted(int(value) for value in np.unique(combined["train_seed"]))
    if unique_seeds != [21, 22, 23, 24, 25]:
        raise RuntimeError(f"Unexpected fivefold source seeds: {unique_seeds}")
    output_path = args.output_dir / "counterfactual_dataset.npz"
    np.savez_compressed(output_path, **combined)
    metadata = {
        "original_dataset": str(ORIGINAL),
        "original_sha256": sha256(ORIGINAL),
        "extension_dataset": str(extension),
        "extension_sha256": sha256(extension),
        "combined_dataset": str(output_path),
        "combined_sha256": sha256(output_path),
        "rows": int(len(combined["train_seed"])),
        "train_seeds": unique_seeds,
        "new_source_policy_run": str(source_run),
        "group_offset_for_new_folds": group_offset,
        "command": subprocess.list2cmdline(command),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
