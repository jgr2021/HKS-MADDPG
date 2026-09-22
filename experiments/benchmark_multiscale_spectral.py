"""Reproducible CPU throughput benchmark for locked Active-GSP descriptors."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.multiscale_spectral_features import multiscale_spectral_action_features
from utils.networks import (
    LearnedActiveGSPResidualPolicy,
    LearnedMultiscaleSpectralResidualPolicy,
    _active_gsp_v3_features_tensor,
)


def measure(function, warmup, iterations):
    with torch.no_grad():
        for _ in range(warmup):
            function()
        started = time.perf_counter()
        for _ in range(iterations):
            function()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260711)
    torch.set_num_threads(args.threads)
    rows = []
    for batch_size in (1, 4, 64, 256):
        obs = torch.randn(batch_size, 18) * 0.5
        baseline_actor = LearnedActiveGSPResidualPolicy(18, 5, agent_index=0).eval()
        multiscale_actor = LearnedMultiscaleSpectralResidualPolicy(18, 5, agent_index=0).eval()
        functions = (
            ("current_single_energy_features", lambda: _active_gsp_v3_features_tensor(obs)),
            ("exact_multiscale_features", lambda: multiscale_spectral_action_features(obs, 0)),
            ("current_single_energy_actor", lambda: baseline_actor(obs)),
            ("exact_multiscale_actor", lambda: multiscale_actor(obs)),
        )
        iterations = args.iterations if batch_size <= 4 else max(100, args.iterations // 5)
        for name, function in functions:
            milliseconds = measure(function, warmup=20, iterations=iterations)
            rows.append({
                "batch_size": batch_size,
                "benchmark": name,
                "milliseconds_per_call": milliseconds,
                "items_per_second": batch_size * 1000.0 / milliseconds,
                "iterations": iterations,
            })
    with (output_dir / "throughput.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metadata.json").write_text(json.dumps({
        "torch_version": torch.__version__, "device": "cpu", "threads": args.threads,
        "polynomial_approximation_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
