"""Train Raw or HKS MADDPG using the existing audited experiment loop."""
import argparse
import json
import os
from pathlib import Path

from _common import CURVE_SEED, FINAL_SEED, METHODS, load_runtime, positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--steps", type=positive_int, default=2000000)
    parser.add_argument("--checkpoint-every", type=positive_int, default=20000)
    parser.add_argument("--rollout-threads", type=positive_int, default=4)
    options = parser.parse_args()
    if options.steps % (25 * options.rollout_threads):
        parser.error("--steps must be divisible by 25 * --rollout-threads")
    if options.checkpoint_every % options.rollout_threads:
        parser.error("--checkpoint-every must be divisible by --rollout-threads")
    if options.steps < 2000:
        parser.error("use at least 2000 steps so the 1024-sample replay batch can warm up")

    output = options.output.resolve()
    run_dir = output / options.method / f"seed_{options.seed}"
    if run_dir.exists():
        parser.error(f"run already exists: {run_dir}")

    protocol, evaluation = load_runtime()
    import torch
    from utils.cuda_protocol_audit import CudaAuditedMADDPG, require_cuda
    device = require_cuda()
    args = evaluation.training_args("formal")
    args.total_env_steps = options.steps
    args.checkpoint_steps = sorted(set(range(options.checkpoint_every, options.steps + 1,
                                            options.checkpoint_every)) | {options.steps})
    args.n_rollout_threads = options.rollout_threads
    args.batch_size = 1024
    args.budget_episodes = options.steps // 25
    # The original loop adds training_seed * 10000. Cancel it to share banks.
    args.eval_seed_base = FINAL_SEED - options.seed * 10000
    args.curve_eval_seed_base = CURVE_SEED - options.seed * 10000
    actor, width = METHODS[options.method]
    protocol.METHODS = {options.method: dict(env_id="simple_spread", actor_model=actor,
                                           actor_input_dim=width, short=options.method)}
    protocol.MADDPG = CudaAuditedMADDPG
    protocol.USE_CUDA = True
    CudaAuditedMADDPG.device_audit_path = run_dir / "gpu_device_audit.json"
    torch.set_num_threads(args.n_training_threads)
    run_dir.mkdir(parents=True)
    (run_dir / "protocol.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run_dir / "device.json").write_text(json.dumps(device, indent=2), encoding="utf-8")
    print(f"Training {options.method}, seed={options.seed}, rollout={options.rollout_threads}")
    print(f"Output: {run_dir}; progress: training.log", flush=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(output)
        protocol.train_one(options.method, options.seed, args, output, "public")
    finally:
        os.chdir(previous_cwd)
    print(f"Completed: {run_dir}")


if __name__ == "__main__":
    main()
