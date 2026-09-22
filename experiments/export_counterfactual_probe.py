"""Export a reproducible Active-GSP Hungarian probe initialization payload."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.probe_action_value_representations import train_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="experiments/action_value_representation_probe_20260712")
    parser.add_argument("--output", default="experiments/action_value_representation_probe_20260712/active_hungarian_probe_seed1.pt")
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    with np.load(Path(args.probe_root) / "counterfactual_dataset.npz") as loaded:
        data = {key: loaded[key] for key in loaded.files}
    metrics, bundle = train_probe(
        data, "raw_action_active", args.model_seed, 23, args.epochs, 6, True
    )
    payload = {
        "representation": "raw_action_active",
        "target": "one_step_hungarian_delta_relative_to_noop",
        "model_seed": args.model_seed,
        "probe_state_dict": bundle["model"].state_dict(),
        "x_mean": bundle["x_mean"], "x_std": bundle["x_std"],
        "y_mean": bundle["y_mean"], "y_std": bundle["y_std"],
        "heldout_metrics": metrics,
    }
    torch.save(payload, args.output)
    Path(args.output).with_suffix(".json").write_text(json.dumps({
        "representation": payload["representation"], "target": payload["target"],
        "model_seed": args.model_seed, "heldout_metrics": metrics,
    }, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
