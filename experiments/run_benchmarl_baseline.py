"""Launch a reproducible BenchMARL baseline on MPE2 simple_spread_v3."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ON_POLICY = {"mappo", "ippo"}
OFF_POLICY = {"qmix", "vdn", "iql"}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=sorted(ON_POLICY | OFF_POLICY), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames-per-batch", type=int, default=2_500)
    parser.add_argument("--minibatch-iters", type=int, default=5)
    parser.add_argument("--optimizer-steps", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--init-random-frames", type=int, default=2_500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_per_batch = min(args.frames_per_batch, args.frames)

    if args.device == "auto":
        import torch

        train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        train_device = args.device

    command = [
        sys.executable,
        "-m",
        "benchmarl.run",
        f"algorithm={args.algorithm}",
        "task=pettingzoo/simple_spread",
        "task.max_cycles=25",
        "task.local_ratio=0.5",
        "task.N=3",
        f"seed={args.seed}",
        f"experiment.max_n_frames={args.frames}",
        "experiment.prefer_continuous_actions=false",
        "experiment.sampling_device=cpu",
        f"experiment.train_device={train_device}",
        "experiment.evaluation=false",
        "experiment.render=false",
        # BenchMARL 1.3 combined with TorchRL 0.6 constructs a duplicated CSV
        # logger path on Windows. Hydra config and checkpoints remain sufficient
        # for reproducibility and evaluation, so disable the optional logger.
        "experiment.loggers=[]",
        "experiment.checkpoint_at_end=true",
        f"experiment.save_folder={output_dir.as_posix()}",
        f"hydra.run.dir={output_dir.as_posix()}",
    ]

    if args.algorithm in ON_POLICY:
        minibatch_size = min(500, frames_per_batch)
        while frames_per_batch % minibatch_size:
            minibatch_size -= 1
        command.extend(
            [
                f"experiment.on_policy_collected_frames_per_batch={frames_per_batch}",
                "experiment.on_policy_n_envs_per_worker=1",
                f"experiment.on_policy_n_minibatch_iters={args.minibatch_iters}",
                f"experiment.on_policy_minibatch_size={minibatch_size}",
            ]
        )
    else:
        command.extend(
            [
                f"experiment.off_policy_collected_frames_per_batch={frames_per_batch}",
                "experiment.off_policy_n_envs_per_worker=1",
                f"experiment.off_policy_n_optimizer_steps={args.optimizer_steps}",
                f"experiment.off_policy_train_batch_size={args.train_batch_size}",
                f"experiment.off_policy_memory_size={args.replay_size}",
                f"experiment.off_policy_init_random_frames={min(args.init_random_frames, args.frames)}",
            ]
        )

    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": args.algorithm,
        "seed": args.seed,
        "frames": args.frames,
        "environment": {
            "name": "mpe2.simple_spread_v3",
            "n_agents": 3,
            "max_cycles": 25,
            "local_ratio": 0.5,
            "continuous_actions": False,
        },
        "devices": {"sampling": "cpu", "training": train_device},
        "versions": {
            name: package_version(name)
            for name in ("benchmarl", "torch", "torchrl", "tensordict", "pettingzoo", "mpe2")
        },
        "command": command,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)

    checkpoints = sorted(output_dir.rglob("checkpoint_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint was produced under {output_dir}")

    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoints[-1].resolve()),
        "all_checkpoints": [str(path.resolve()) for path in checkpoints],
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
