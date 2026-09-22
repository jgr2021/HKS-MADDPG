"""Compute-only replay profiling; no environment interaction or model updates."""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from utils.buffer import ReplayBuffer
from utils.replay_compute_cache import CachedRewardStatsReplayBuffer


def memory_status():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {"load_percent": status.load, "total_gib": status.total / 2**30,
            "available_gib": status.available / 2**30}


def measure(function, repetitions=12):
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        values.append((time.perf_counter() - start) * 1000)
    return float(np.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    memory = memory_status()
    if memory["available_gib"] < 3:
        raise RuntimeError("Less than3GiB RAM available; do not contend with live training")
    rows = []
    for count in (100000, 1000000):
        buffer = ReplayBuffer(count, 3, [18]*3, [5]*3)
        buffer.filled_i = count
        for i in range(3):
            buffer.rew_buffs[i][:] = np.linspace(-3 - i, 1 + i, count)
        indices = np.arange(1024)
        def reward_stats():
            return [(array.mean(), array.std()) for array in buffer.rew_buffs]
        def gather_cast():
            return [[torch.Tensor(array[indices]) for array in family] for family in
                    (buffer.obs_buffs, buffer.ac_buffs, buffer.rew_buffs, buffer.next_obs_buffs, buffer.done_buffs)]
        components = {
            "index_array_and_choice": lambda: np.random.choice(np.arange(count), size=1024, replace=False),
            "integer_choice_same_rng_candidate": lambda: np.random.choice(count, size=1024, replace=False),
            "reward_mean_std_3agents": reward_stats,
            "gather_and_cpu_tensor_cast": gather_cast,
            "original_sample_cpu": lambda: buffer.sample(1024, to_gpu=False),
        }
        result = {name: measure(function) for name, function in components.items()}
        cached = CachedRewardStatsReplayBuffer(1, 3, [18]*3, [5]*3)
        for name in ("filled_i", "max_steps", "obs_buffs", "ac_buffs", "rew_buffs", "next_obs_buffs", "done_buffs"):
            setattr(cached, name, getattr(buffer, name))
        blocks = {"original": [], "cached": []}
        for block in range(6):
            for name in (("original", "cached") if block % 2 == 0 else ("cached", "original")):
                target = buffer if name == "original" else cached
                cached._reward_stats = None
                start = time.perf_counter()
                for _ in range(12):
                    target.sample(1024, to_gpu=False)
                blocks[name].append((time.perf_counter() - start) * 1000)
        result["original_12_samples_ms"] = float(np.median(blocks["original"]))
        result["cached_12_samples_ms"] = float(np.median(blocks["cached"]))
        result["replay_block_speedup"] = result["original_12_samples_ms"] / result["cached_12_samples_ms"]
        if torch.cuda.is_available():
            tensors = gather_cast()
            def transfer():
                moved = [[value.cuda() for value in family] for family in tensors]
                torch.cuda.synchronize()
                return moved
            transfer()
            result["cpu_gpu_transfer_15_tensors"] = measure(transfer)
        rows.append({"replay_entries": count, "batch_size": 1024, "median_ms": result})
        del buffer, cached
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv"],
                         capture_output=True, text=True)
    output = {"memory_before": memory, "memory_after": memory_status(), "rows": rows,
              "gpu_snapshot": gpu.stdout.strip(),
              "scope": "Synthetic replay microbenchmark concurrent with live training; no model fitting or env steps. Zero observation storage, nonconstant rewards. Components measured separately, not additive percentages or end-to-end speedup."}
    (args.output / "profile.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
