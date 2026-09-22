"""Action- and seed-stratified shuffled controls for representation probes."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.probe_action_value_representations import train_probe


CONTROLS = (
    ("active", "raw_action_active"),
    ("multiscale", "raw_action_multiscale"),
    ("vector_gsp", "raw_action_vector_gsp"),
)


def shuffled_copy(data, field, seed):
    output = {key: np.array(value, copy=True) for key, value in data.items()}
    rng = np.random.default_rng(seed)
    action_index = data["action"].argmax(axis=1)
    for train_seed in np.unique(data["train_seed"]):
        for action in range(5):
            indices = np.flatnonzero((data["train_seed"] == train_seed) & (action_index == action))
            output[field][indices] = data[field][rng.permutation(indices)]
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="experiments/action_value_representation_probe_20260712")
    parser.add_argument("--output-dir", default="experiments/action_value_probe_shuffled_20260712")
    parser.add_argument("--model-seeds", default="1,2,3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(Path(args.probe_root) / "counterfactual_dataset.npz") as loaded:
        data = {key: loaded[key] for key in loaded.files}
    rows = []
    for field, representation in CONTROLS:
        shuffled = shuffled_copy(data, field, 20260712)
        for model_seed in [int(value) for value in args.model_seeds.split(",")]:
            row = train_probe(shuffled, representation, model_seed, 23, args.epochs, args.threads)
            row["representation"] = representation + "_shuffled"
            rows.append(row)
    write_csv(output_dir / "results.csv", rows)
    summary = []
    for representation in sorted({row["representation"] for row in rows}):
        selected = [row for row in rows if row["representation"] == representation]
        output = {"representation": representation, "model_runs": len(selected)}
        for key in selected[0]:
            if key not in {"representation", "model_seed", "heldout_train_seed"}:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                output[f"{key}_mean"] = values.mean(); output[f"{key}_std"] = values.std(ddof=1)
        summary.append(output)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "metadata.json").write_text(json.dumps({
        "shuffle": "independent descriptor permutation within each training-seed/action stratum",
        "purpose": "preserve dimension and marginal action distribution while destroying state alignment",
    }, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
