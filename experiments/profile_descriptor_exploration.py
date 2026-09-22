"""Pre-training numerical and CPU latency audit on an existing local-obs bank."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.exploration_descriptor_policies import POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(5681)
    bank = ROOT / "experiments/gsp_exploration_20260906_phase2a/analysis_20260906/local_observation_audit_bank.npz"
    with np.load(bank) as archive:
        raw = torch.from_numpy(archive["observations"])
        kinds = archive["kind"]
    results = {}
    for method, cls in POLICIES.items():
        model = cls(18, 5).eval()
        features = model._hks_features(raw)
        assert torch.isfinite(features).all()
        std = features.double().std(0, unbiased=False)
        assert (std > 1e-6).all(), "Descriptor is degenerate on this bank"
        stats = {"min": features.min(0).values.tolist(), "max": features.max(0).values.tolist(),
                 "std": std.tolist(), "by_bank": {}, "latency_ms_per_batch": {}}
        for kind in sorted(set(kinds)):
            subset = features[kinds == kind]
            stats["by_bank"][kind] = {"count": len(subset), "std": subset.double().std(0, unbiased=False).tolist()}
        for count in (1, 12, 64):
            inputs = raw[:count]
            for _ in range(20):
                model._hks_features(inputs)
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                for _ in range(100):
                    model._hks_features(inputs)
                samples.append((time.perf_counter() - start) * 10)
            stats["latency_ms_per_batch"][count] = {"median": float(np.median(samples)), "min": min(samples), "max": max(samples)}
        results[method] = stats
    source_files = ("utils/exploration_descriptor_policies.py", "utils/exploration_topology_policies.py", "utils/networks.py")
    report = {"passed": True, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "local_observations": len(raw), "observation_bank": str(bank),
              "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
              "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
              "training_interactions_added": 0, "torch_cpu_threads": 1, "dtype": "float32",
              "latency_scope": "Feature-only CPU median of 9 blocks of 100 calls, 20 warmups; no CUDA transfer. Not end-to-end training throughput.",
              "selection_scope": "Fixed parameters before profile; finite/nonconstant eligibility only, no reward or evaluation-based tuning.",
              "methods": results}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Descriptor Preflight", "", "Fixed local-only descriptors on 4500 previously saved observations. No new training.", "",
             "| Method | Standard deviations | ms/batch M=1 | M=12 | M=64 |", "| --- | --- | ---: | ---: | ---: |"]
    for name, stats in results.items():
        lines.append("| " + name + " | " + ", ".join("%.6f" % item for item in stats["std"]) + " | " +
                     " | ".join("%.5f" % stats["latency_ms_per_batch"][count]["median"] for count in (1, 12, 64)) + " |")
    lines += ["", report["latency_scope"], "", report["selection_scope"],
              "", "The bank contains reset observations and a frozen 4ego policy's rollouts, not every future policy distribution.",
              "Normalized regularized resistance is a scaled resolvent contrast, not physical effective resistance. Isolated-node floor behavior remains part of this definition."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
