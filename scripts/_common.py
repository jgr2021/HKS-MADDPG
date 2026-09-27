"""Shared imports and protocol settings for the public command-line tools."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

METHODS = {"raw": ("mlp", 18), "hks": ("confirm_fixed_hks", 21)}
CURVE_SEED = 91000000
FINAL_SEED = 92000000


def positive_int(value):
    import argparse
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def load_runtime(device="cpu"):
    from utils.adaptive_hks_controls import register_policies
    from experiments import run_d4_actor_gpu_screen as evaluation
    from experiments import run_passive_gsp_topology_pilot as protocol
    import run_vector_signal_gsp_experiment as metrics
    register_policies()
    evaluation.EVAL_DEVICE = device
    protocol.METRICS = metrics.METRICS
    protocol.evaluate_model = evaluation.evaluate_model
    return protocol, evaluation


def summarize(rows):
    import numpy as np
    import run_vector_signal_gsp_experiment as metrics
    result = {name: float(np.mean([r[name] for r in rows])) for name in metrics.METRICS}
    # Historical CSVs store the count of covered landmarks (0..3).
    result["coverage_fraction"] = result["final_coverage"] / 3.0
    result["eval_episodes"] = len(rows)
    return result
