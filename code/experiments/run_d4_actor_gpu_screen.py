"""One-seed, actor-only D4 screen using the unchanged corrected MADDPG loop."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments import run_passive_gsp_topology_pilot as protocol
import run_vector_signal_gsp_experiment as metrics_protocol
from utils.cuda_protocol_audit import CudaAuditedMADDPG, require_cuda
from utils.d4_graph_residual import D4DirichletResidualPolicy, D4PotentialResidualPolicy


METHODS = {
    "d4_geometric_control": {"env_id": "simple_spread", "actor_model": "d4_potential_residual",
                             "actor_input_dim": 18, "short": "d4geom"},
    "d4_dirichlet_gsp": {"env_id": "simple_spread", "actor_model": "d4_dirichlet_residual",
                         "actor_input_dim": 18, "short": "d4gsp"},
}
SOURCE_FILES = [
    "utils/d4_graph_residual.py", "utils/cuda_protocol_audit.py", "utils/networks.py",
    "utils/agents.py", "utils/buffer.py", "utils/make_env.py", "utils/env_wrappers.py",
    "algorithms/maddpg.py", "main.py", "experiments/run_passive_gsp_topology_pilot.py",
    "run_vector_signal_gsp_experiment.py", "experiments/run_d4_actor_gpu_screen.py",
    "tests/test_d4_graph_residual.py",
]
EVAL_DEVICE = "cpu"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [REPO_ROOT / name for name in SOURCE_FILES]
    import multiagent.core
    import multiagent.environment
    import multiagent.scenarios.simple_spread
    paths += [Path(module.__file__) for module in
              (multiagent.core, multiagent.environment, multiagent.scenarios.simple_spread)]
    return {str(path.resolve()): sha256(path) for path in paths}


class EvaluationDeviceMADDPG(MADDPG):
    def prep_rollouts(self, device="cpu"):
        return super().prep_rollouts(device="gpu" if EVAL_DEVICE == "cuda" else "cpu")

    def step(self, observations, explore=False):
        return super().step([item.to(EVAL_DEVICE) for item in observations], explore=explore)


def evaluate_model(*args, **kwargs):
    previous = metrics_protocol.MADDPG
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        metrics_protocol.MADDPG = EvaluationDeviceMADDPG
        return metrics_protocol.evaluate_model(*args, **kwargs)
    finally:
        metrics_protocol.MADDPG = previous
        torch.set_num_threads(threads)


def benchmark_inference(repetitions=100):
    """Time a focal actor, including CUDA host transfers, without fitting weights."""
    torch.set_num_threads(1)
    torch.manual_seed(1942)
    rows = []
    for constructor in (D4PotentialResidualPolicy, D4DirichletResidualPolicy):
        cpu = constructor(18, 5).eval()
        gpu = copy.deepcopy(cpu).cuda().eval()
        for batch in (1, 4, 64):
            inputs = torch.randn(batch, 18)
            with torch.no_grad():
                expected = cpu(inputs)
                actual = gpu(inputs.cuda()).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            difference = float((actual - expected).abs().max())
            mismatches = int((actual.argmax(1) != expected.argmax(1)).sum())
            for device, model in (("cpu", cpu), ("cuda", gpu)):
                def invoke():
                    return model(inputs.to(device)).cpu()
                with torch.no_grad():
                    for _ in range(10):
                        invoke()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        invoke()
                    torch.cuda.synchronize()
                rows.append({"actor": constructor.__name__, "batch": batch, "device": device,
                             "ms_per_call_including_transfer": (time.perf_counter() - started) * 1000 / repetitions,
                             "max_cpu_gpu_logit_error": difference, "argmax_mismatches": mismatches})
    totals = {device: sum(row["ms_per_call_including_transfer"] for row in rows
                         if row["batch"] == 1 and row["device"] == device)
              for device in ("cpu", "cuda")}
    return {"rows": rows, "chosen_evaluation_device": min(totals, key=totals.get),
            "rule": "lower summed batch-1 inference latency over the two actors, including host transfers",
            "training_update_device": "cuda", "rollout_device": "cpu (unchanged training loop)",
            "evaluation_cpu_threads": 1, "training_cpu_threads": 6}


def training_args(phase):
    formal = phase == "formal"
    return SimpleNamespace(
        total_env_steps=100000 if formal else 1000,
        checkpoint_steps=[20000, 40000, 60000, 80000, 100000] if formal else [1000],
        n_rollout_threads=4, n_training_threads=6, buffer_length=1000000,
        episode_length=25, steps_per_update=100, batch_size=64, hidden_dim=64,
        lr=0.01, policy_audit_batch_size=512, print_interval=5000,
        eval_episodes=500 if formal else 5, curve_eval_episodes=100 if formal else 2,
        eval_seed_base=990000, curve_eval_seed_base=770000,
    )


def validate_run(directory, args):
    counters = json.loads((directory / "training_counters.json").read_text())
    shapes = json.loads((directory / "tensor_shape_report.json").read_text())
    audit = json.loads((directory / "gpu_device_audit.json").read_text())
    assert counters["global_env_steps"] == args.total_env_steps
    assert counters["saved_checkpoints"] == args.checkpoint_steps
    assert shapes["replay_buffer_obs_dims"] == [18, 18, 18]
    assert shapes["critic_fc1_in_features"] == 69
    assert audit["cuda_updates_verified"] == args.total_env_steps // 100 * 4 * 3
    assert len(audit["agents"]) == 3 and not audit["cpu_fallback"]
    for record in audit["agents"].values():
        for name in ("actor", "critic"):
            assert record[f"{name}_gradient_devices"] == ["cuda:0"]
        for name, forwards in record["forwards"].items():
            expected = [64, 18 if "actor" in name else 69]
            sampled = [item for item in forwards if item["purpose"] == "sampled_transition"]
            assert sampled and all(item["shape"] == expected for item in sampled)
            assert all(item["shape"][1] == expected[1] and item["device"] == "cuda:0"
                       and item["raw_values_match"] for item in forwards)
    checkpoint = directory / "checkpoints" / f"model_final_{args.total_env_steps}.pt"
    restored = MADDPG.init_from_save(checkpoint)
    for agent in restored.agents:
        for name in ("policy", "critic", "target_policy", "target_critic"):
            assert all(torch.isfinite(value).all() for value in getattr(agent, name).state_dict().values())
    return {"passed": True, "global_env_steps": counters["global_env_steps"],
            "cuda_updates_verified": audit["cuda_updates_verified"], "checkpoint_sha256": sha256(checkpoint)}


def evaluate_archived_baselines(root, args):
    sources = json.loads((REPO_ROOT / "experiments/corrected_active_gsp_multiseed_20260711_124239/sources.json").read_text())["1"]
    summaries = []
    for name, source in sources.items():
        checkpoint = REPO_ROOT / source / "checkpoints/model_final_100000.pt"
        before = sha256(checkpoint)
        rows = evaluate_model("simple_spread", checkpoint, args.eval_episodes,
                              args.episode_length, args.eval_seed_base + 10000)
        assert sha256(checkpoint) == before
        protocol.write_csv(root / f"archive_{name}_evaluation.csv", rows)
        summary = protocol.summarize_episode_rows(rows)
        summary.update(method=f"archive_{name}", seed=1, checkpoint=str(checkpoint), checkpoint_sha256=before,
                       interpretation="context only: historical training device and audit sampling not matched")
        summaries.append(summary)
    return summaries


def main():
    global EVAL_DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    options = parser.parse_args()
    if os.environ.get("MADDPG_FORCE_CPU") == "1":
        raise RuntimeError("MADDPG_FORCE_CPU=1 conflicts with the required GPU protocol")
    runtime = require_cuda()
    hashes = source_hashes()
    if options.phase == "formal":
        if options.smoke_dir is None:
            parser.error("formal training requires a completed --smoke-dir")
        smoke = json.loads((options.smoke_dir / "status.json").read_text())
        if smoke["status"] != "completed" or smoke["phase"] != "smoke" or smoke["source_hashes"] != hashes:
            raise RuntimeError("A successful smoke test of these exact source files is required")
        if smoke["completed_methods"] != list(METHODS):
            raise RuntimeError("Both matched smoke runs must be completed")
        for method in METHODS:
            validate_run(options.smoke_dir / method / "seed_1", training_args("smoke"))
    root = options.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    args = training_args(options.phase)
    status = {"status": "running", "phase": options.phase, "pid": os.getpid(), "seed": 1,
              "runtime": runtime, "source_hashes": hashes, "training": vars(args),
              "command": subprocess.list2cmdline([sys.executable] + sys.argv), "completed_methods": [],
              "protocol": "only actor architecture changes; raw replay and critic; unchanged corrected training function",
              "promotion_rule": "GSP must improve return, Hungarian distance and radius AUC over matched D4 control without increasing collision rate; one seed is exploratory only"}
    protocol.write_json(root / "status.json", status)
    try:
        if options.phase == "smoke":
            tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "tests.test_d4_graph_residual", "-v"],
                                   cwd=REPO_ROOT, capture_output=True, text=True)
            (root / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
            tests.check_returncode()
            benchmark = benchmark_inference()
        else:
            benchmark = json.loads((options.smoke_dir / "inference_benchmark.json").read_text())
        protocol.write_json(root / "inference_benchmark.json", benchmark)
        EVAL_DEVICE = benchmark["chosen_evaluation_device"]
        status["evaluation_device"] = EVAL_DEVICE
        protocol.MADDPG = CudaAuditedMADDPG
        protocol.USE_CUDA = True
        protocol.METHODS = METHODS
        protocol.METRICS = metrics_protocol.METRICS
        protocol.evaluate_model = evaluate_model
        stamp = time.strftime("%Y%m%d_%H%M%S")
        summaries = []
        for method in METHODS:
            torch.set_num_threads(args.n_training_threads)
            directory = root / method / "seed_1"
            CudaAuditedMADDPG.device_audit_path = directory / "gpu_device_audit.json"
            status["current_method"] = method
            protocol.write_json(root / "status.json", status)
            print(f"Starting {method}, seed=1, budget={args.total_env_steps}, updates=cuda, eval={EVAL_DEVICE}", flush=True)
            summary, _ = protocol.train_one(method, 1, args, root, stamp)
            validation = validate_run(directory, args)
            protocol.write_json(directory / "validation.json", validation)
            assert source_hashes() == hashes, "Source changed during the experiment"
            summaries.append(summary)
            status["completed_methods"].append(method)
            protocol.write_json(root / "status.json", status)
            print(f"Completed {method}: {json.dumps(validation)}", flush=True)
        if options.phase == "formal":
            status["current_method"] = "archived_baseline_evaluation"
            protocol.write_json(root / "status.json", status)
            summaries += evaluate_archived_baselines(root, args)
            target, control = summaries[1], summaries[0]
            deltas = {key: target[key] - control[key] for key in metrics_protocol.METRICS}
            passed = (deltas["return"] > 0 and deltas["hungarian_assignment_distance"] < 0
                      and deltas["coverage_radius_auc"] > 0 and deltas["collision_step_rate"] <= 0)
            protocol.write_json(root / "exploratory_gate.json", {"passed": passed, "deltas": deltas,
                                "n_training_seeds": 1, "paper_level_claim": False})
        protocol.write_csv(root / "seed_summaries.csv", summaries)
        status.update(status="completed", current_method=None)
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        protocol.write_json(root / "status.json", status)


if __name__ == "__main__":
    main()
