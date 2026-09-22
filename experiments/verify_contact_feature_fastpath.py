"""Verify contact features and the predeclared fidelity gate without training."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_d4_actor_gpu_screen import read_csv, write_csv, write_json
from experiments.run_d4_actor_gpu_screen import sha256
from utils.active_gsp_v3_features import ACTION_TO_CONTROL_FLOAT32
from utils.contact_active_gsp_features import compute_contact_active_features
from utils.networks import _active_gsp_v3_features_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    root = options.diagnostic_dir.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["status"] == "completed"
    dataset = root / "diagnostic_dataset.npz"
    assert sha256(dataset) == protocol["dataset_sha256"]
    rows = read_csv(root / "feature_errors.csv")
    gate_rows = []
    for source in ("all", *protocol["sources"]):
        for feature in ("coverage_energy", "crowding_energy"):
            selected = {row["mode"]: row for row in rows if row["source"] == source
                        and row["separation_bin"] == "contact_lt030" and row["feature"] == feature}
            ratio = float(selected["known_contact_graph"]["rmse"]) / float(selected["legacy"]["rmse"])
            sign_delta = float(selected["known_contact_graph"]["sign_agreement"]) - float(selected["legacy"]["sign_agreement"])
            gate_rows.append({"source": source, "feature": feature, "rmse_reduction_fraction": 1 - ratio,
                              "sign_agreement_delta": sign_delta, "passes_rmse_threshold": ratio <= .90})
    gate_passed = all(row["passes_rmse_threshold"] for row in gate_rows) and all(
        row["sign_agreement_delta"] >= 0 for row in gate_rows if row["source"] == "all")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This verification explicitly requires CUDA; no CPU-only fallback")
    with np.load(dataset) as data:
        raw = torch.tensor(data["raw"])
        expected = data["predicted_features"][:, 2].copy()
        observed_actual = data["actual"].copy()
        predicted = data["predicted"][:, 2].copy()
        hidden_drift = (.075 * data["validation_only_velocities"][:, 1:]
                        + .05 * (data["validation_only_base_actions"][:, 1:] @ ACTION_TO_CONTROL_FLOAT32))
    residual = observed_actual[:, :, 1:] - predicted[:, :, 1:]
    hidden_error = float(np.abs(residual - hidden_drift[:, None]).max())
    assert hidden_error < 2e-6
    actions = torch.tensor(ACTION_TO_CONTROL_FLOAT32)
    equivalence, benchmarks = [], []
    with torch.no_grad():
        for device in ("cpu", "cuda"):
            inputs, controls = raw.to(device), actions.to(device)
            outputs = compute_contact_active_features(inputs, controls)
            error = float(np.abs(outputs.cpu().numpy() - expected).max())
            np.testing.assert_allclose(outputs.cpu().numpy(), expected, rtol=2e-3, atol=3e-6)
            equivalence.append({"device": str(outputs.device), "batch": len(inputs), "max_reference_error": error})
            for batch in (1, 4, 64, 512):
                for name, function in (("legacy_features", lambda x: _active_gsp_v3_features_tensor(x)),
                                       ("known_contact_features", lambda x: compute_contact_active_features(x, controls))):
                    x = inputs[:batch]
                    for _ in range(10):
                        function(x)
                    samples = []
                    for _ in range(3):
                        if device == "cuda":
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(100):
                            function(x)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        samples.append((time.perf_counter() - start) * 10)
                    benchmarks.append({"device": device, "batch": batch, "function": name,
                                       "median_ms_per_call": float(np.median(samples)),
                                       "scope": "resident observations/outputs; new controls preallocated; legacy internal allocations/transfers retained; no environment"})
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "feature_fidelity_gate.csv", gate_rows)
    write_csv(output / "benchmark.csv", benchmarks)
    summary = {"status": "completed", "new_training_steps": 0, "fidelity_gate_passed": gate_passed,
               "equivalence": equivalence, "unknown_nonfocal_drift_decomposition_max_error": hidden_error,
               "dataset_sha256": sha256(dataset), "gpu": torch.cuda.get_device_name(0),
               "source_hashes": {name: sha256(REPO_ROOT / name) for name in (
                   "utils/contact_active_gsp_features.py", "tests/test_contact_active_features.py",
                   "experiments/verify_contact_feature_fastpath.py")},
               "interpretation": "feature correctness and standalone speed only; no actor or training integration"}
    assert summary["dataset_sha256"] == protocol["dataset_sha256"]
    write_json(output / "verification.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
