"""Fresh 100,000-episode-equivalent reruns, with explicit transition budgets."""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_gsp_exploration_campaign import (
    METHODS, TESTS, code_hashes, gate, read_json, sha256, verify_snapshot, write_json,
)
from experiments.run_d4_actor_gpu_screen import training_args as short_args

REGISTRATIONS = ["utils.exploration_" + name + "_policies" for name in
                 ("contact", "topology", "weight", "operator", "descriptor")]
LONG_TESTS = TESTS + tuple("tests.test_exploration_" + name + "_policies" for name in
                          ("topology", "weight", "operator", "descriptor")) + (
                              "tests.test_gsp_long_campaign",)


def training_args(phase):
    if phase not in ("formal", "smoke"):
        raise ValueError("Unknown phase")
    args = short_args(phase)
    args.total_env_steps = 2500000 if phase == "formal" else 2000
    args.checkpoint_steps = list(range(20000, 2500001, 20000)) if phase == "formal" else [2000]
    args.batch_size = 1024
    args.budget_episodes = args.total_env_steps // args.episode_length
    return args


def expected_updates(args):
    if args.steps_per_update % args.n_rollout_threads:
        raise ValueError("Update interval must align with vector transitions")
    first = (args.batch_size + args.steps_per_update - 1) // args.steps_per_update
    events = max(0, args.total_env_steps // args.steps_per_update - first + 1)
    return events * args.n_rollout_threads * 3


def methods():
    from utils.exploration_topology_policies import experiment_methods as topology
    from utils.exploration_weight_policies import experiment_methods as weights
    from utils.exploration_operator_policies import experiment_methods as operators
    from utils.exploration_descriptor_policies import experiment_methods as descriptors
    result = {name: dict(spec, group="active") for name, spec in METHODS.items()}
    for group, candidates in (("topology", topology(Path("unused"))),
                              ("weights", weights(ROOT)), ("operators", operators(ROOT)),
                              ("descriptors", descriptors(ROOT))):
        for name, spec in candidates.items():
            if name not in result and not spec.get("reuse_from"):
                result[name] = dict(spec, group=group)
    # Complete the ordinary geometry control before both active and passive variants.
    order = ["raw_mlp", "explore_geometric_stats3"] + [name for name in result
             if name not in ("raw_mlp", "explore_geometric_stats3")]
    result = {name: result[name] for name in order}
    for index, (name, spec) in enumerate(result.items(), 1):
        spec["short"] = "m%02d" % index
        spec.setdefault("actor_input_dim", 18)
        spec["fresh"] = True
        assert "reuse_from" not in spec
    result["raw_mlp"]["factor"] = "fresh long-horizon raw baseline"
    assert len(result) == 23
    return result


def require_space(campaign, minimum_gib=2):
    free = shutil.disk_usage(campaign).free
    if free < minimum_gib * 1024 ** 3:
        raise RuntimeError("Insufficient free disk space; all evidence preserved")
    return free


def freeze(campaign):
    campaign.mkdir(parents=True, exist_ok=False)
    require_space(campaign, 8)
    code = campaign / "code"
    paths = list(ROOT.glob("*.py"))
    for folder in ("utils", "algorithms", "tests"):
        paths += list((ROOT / folder).rglob("*.py"))
    paths += [ROOT / "experiments" / name for name in (
        "run_passive_gsp_topology_pilot.py", "run_d4_actor_gpu_screen.py",
        "run_gsp_exploration_campaign.py", "run_gsp_long_campaign.py")]
    for source in paths:
        target = code / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    import multiagent
    for source in Path(multiagent.__file__).parent.rglob("*.py"):
        target = code / "multiagent" / source.relative_to(Path(multiagent.__file__).parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    graph_definitions = {}
    for phase in ("2a", "2b", "2c", "2d"):
        previous = ROOT / ("experiments/gsp_exploration_20260906_phase" + phase)
        original = verify_snapshot(previous)
        graph_definitions[phase] = original["graph_protocol"]
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "original_repo": str(ROOT),
        "study": "long_horizon_redo", "seed": 1, "methods": methods(),
        "formal": vars(training_args("formal")), "smoke": vars(training_args("smoke")),
        "training_episodes": 100000, "episode_length": 25, "total_env_steps": 2500000,
        "checkpoint_interval_env_steps": 20000, "checkpoint_count": 125,
        "source_hashes": code_hashes(code), "registration_modules": REGISTRATIONS, "tests": LONG_TESTS,
        "frozen_graph_definitions_only": graph_definitions,
        "definition_note": "Inherited graph formulas only; old budgets/results/reuse rules do not apply.",
        "protocol": "actor-local augmentation; raw18 environment/replay; raw joint obs54+actions15 critic69; original reward/actions/optimizer objectives",
        "changes_from_short_screen": "2500000 transitions instead of 100000; original long-run batch1024 instead of short-screen batch64; smoke2000 to exercise updates",
        "other_parameters": "4 rollout envs, hidden64, Adam lr=.01, tau=.01, original update and replay algorithms unchanged",
        "resume": False, "reuse_old_results": False, "training_device": "cuda", "rollout_eval_device": "cpu",
        "initial_smokes": ["raw_mlp", "explore_regularized_resistance3"],
        "screening_rule": "Fixed final endpoint. Return up, assignment distance down, coverage AUC up, collision not up vs required controls AND raw. One seed, not proof of convergence or robust superiority.",
        "evaluation_rule": "100 episodes per checkpoint, 500 final; same banks as earlier for traceability, still screening, not independent test",
        "disk_rule": "At least8GiB before preparation and2GiB before each worker; no deletion or automatic migration",
        "queue_rule": "Sequential fresh runs; technical failure pauses; inspect raw learning before interpreting variants; no short-run ranking exclusions",
        "budget_scope": "23 unique fresh formal runs, not 23 groups; no automatic extra seeds or extra configurations",
    }
    write_json(campaign / "manifest.json", manifest)
    write_json(campaign / "queue_status.json", {"status": "prepared", "completed": [], "pid": None})
    refresh(campaign)


def validate_run(directory, args, spec):
    import torch
    from algorithms.maddpg import MADDPG
    counters = read_json(directory / "training_counters.json")
    shapes = read_json(directory / "tensor_shape_report.json")
    audit = read_json(directory / "gpu_device_audit.json")
    assert counters["global_env_steps"] == counters["total_env_steps"] == args.total_env_steps
    assert counters["env_episodes_started"] == args.total_env_steps // args.episode_length
    assert counters["episode_batches"] * args.n_rollout_threads == counters["env_episodes_started"]
    assert counters["saved_checkpoints"] == args.checkpoint_steps
    assert audit["cuda_updates_verified"] == expected_updates(args)
    assert not audit["cpu_fallback"] and len(audit["agents"]) == 3
    for key in ("actor_augmented_tensor_shape", "target_actor_augmented_tensor_shape"):
        assert shapes[key] == [args.batch_size, spec["actor_input_dim"]]
    for key in ("raw_actor_tensor_shape", "target_actor_raw_tensor_shape"):
        assert shapes[key] == [args.batch_size, 18]
    for key in ("critic_input_tensor_shape", "target_critic_input_tensor_shape"):
        assert shapes[key] == [args.batch_size, 69]
    for key in ("replay_obs_tensor_shapes", "replay_next_obs_tensor_shapes"):
        assert shapes[key] == [[args.batch_size, 18]] * 3
    assert shapes["replay_buffer_obs_dims"] == [18, 18, 18]
    for record in audit["agents"].values():
        for name in ("actor", "critic"):
            assert record[name + "_gradient_devices"] == ["cuda:0"]
        for name, forwards in record["forwards"].items():
            width = 18 if "actor" in name else 69
            sampled = [item for item in forwards if item["purpose"] == "sampled_transition"]
            assert sampled and all(item["shape"] == [args.batch_size, width] for item in sampled)
            assert all(item["raw_values_match"] and item["device"] == "cuda:0" for item in forwards)
    checkpoint_hashes = {}
    for label in ["model_step" + str(s) for s in args.checkpoint_steps] + ["model_final_" + str(args.total_env_steps)]:
        checkpoint = directory / "checkpoints" / (label + ".pt")
        restored = MADDPG.init_from_save(checkpoint)
        for agent in restored.agents:
            for name in ("policy", "critic", "target_policy", "target_critic"):
                assert all(torch.isfinite(value).all() for value in getattr(agent, name).state_dict().values())
        checkpoint_hashes[label] = sha256(checkpoint)
    with (directory / "learning_curve_eval.csv").open(encoding="utf-8-sig", newline="") as handle:
        curve = list(csv.DictReader(handle))
    assert [row["checkpoint"] for row in curve] == ["model_step" + str(s) for s in args.checkpoint_steps]
    assert all(int(row["eval_episodes"]) == args.curve_eval_episodes for row in curve)
    summary = read_json(directory / "summary.json")
    assert summary["eval_episodes"] == args.eval_episodes
    with (directory / "per_evaluation_episode_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    first_seed = args.eval_seed_base + 10000
    assert [int(row["test_seed"]) for row in rows] == list(range(first_seed, first_seed + args.eval_episodes))
    return {"passed": True, "global_env_steps": args.total_env_steps,
            "env_episodes": counters["env_episodes_started"], "batch_size": args.batch_size,
            "cuda_updates_verified": audit["cuda_updates_verified"], "checkpoint_hashes": checkpoint_hashes}


def worker(campaign, phase, method):
    manifest = verify_snapshot(campaign)
    if ROOT != campaign / "code":
        raise RuntimeError("Worker must use the immutable code snapshot")
    for module in REGISTRATIONS:
        importlib.import_module(module).register_policies()
    from experiments import run_passive_gsp_topology_pilot as protocol
    from experiments.run_d4_actor_gpu_screen import evaluate_model
    from utils.cuda_protocol_audit import CudaAuditedMADDPG, require_cuda
    import run_vector_signal_gsp_experiment as metrics
    import torch
    if os.environ.get("MADDPG_FORCE_CPU") == "1":
        raise RuntimeError("CPU override conflicts with this CUDA protocol")
    args, spec = training_args(phase), manifest["methods"][method]
    assert vars(args) == manifest[phase]
    if phase == "formal":
        smoke = read_json(campaign / ("smoke_" + method + ".json"))
        assert smoke["status"] == "completed" and smoke["validation"]["passed"]
        assert smoke["validation"] == validate_run(campaign / "smoke" / method / "seed_1", training_args("smoke"), spec)
    directory = campaign / phase / method / "seed_1"
    if directory.exists():
        raise FileExistsError("No overwrite or resume: " + str(directory))
    require_space(campaign)
    protocol.METHODS = {method: dict(spec, env_id="simple_spread")}
    protocol.MADDPG, protocol.USE_CUDA = CudaAuditedMADDPG, True
    protocol.METRICS, protocol.evaluate_model = metrics.METRICS, evaluate_model
    CudaAuditedMADDPG.device_audit_path = directory / "gpu_device_audit.json"
    stamp = time.strftime("%m%d_%H%M%S") + ("f" if phase == "formal" else "s")
    state = {"status": "running", "pid": os.getpid(), "phase": phase, "method": method,
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime": require_cuda(),
             "total_env_steps": args.total_env_steps, "total_env_episodes": args.total_env_steps // 25,
             "model_run_directory": str(campaign / "models/simple_spread" / ("pgt_" + spec["short"] + "_s1_" + stamp) / "run1"),
             "command": subprocess.list2cmdline([sys.executable] + sys.argv)}
    state_path = campaign / (phase + "_" + method + ".json")
    write_json(state_path, state)
    try:
        os.chdir(campaign)
        torch.set_num_threads(args.n_training_threads)
        protocol.train_one(method, 1, args, campaign / phase, stamp)
        validation = validate_run(directory, args, spec)
        verify_snapshot(campaign)
        write_json(directory / "validation.json", validation)
        state.update(status="completed", validation=validation, summary_path=str(directory / "summary.json"))
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(state_path, state)


def refresh(campaign):
    manifest = read_json(campaign / "manifest.json")
    rows, results = [], {}
    for name, spec in manifest["methods"].items():
        marker = campaign / ("formal_" + name + ".json")
        state = read_json(marker) if marker.exists() else {"status": "queued"}
        directory = campaign / "formal" / name / "seed_1"
        row = {"method": name, "group": spec["group"], "seed": 1, "status": state["status"],
               "budget_env_steps": 2500000, "budget_episodes": 100000, "episode_length": 25,
               "batch_size": 1024, "checkpoint_interval_env_steps": 20000,
               "control": spec["control"], "artifact_path": str(directory), "observed_failure": "",
               "possible_cause_not_proven": "", "converged": "not established"}
        if state["status"] == "completed":
            summary = read_json(directory / "summary.json")
            results[name] = summary
            for key in ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
                        "final_coverage", "final3", "nearest_landmark_distance", "global_env_steps", "env_steps_per_sec"):
                row[key] = summary[key]
            if spec["control"]:
                required = list(dict.fromkeys(["raw_mlp", spec["control"]] + spec.get("required_additional_controls", [])))
                if any(control not in results for control in required):
                    raise RuntimeError("Missing required long-horizon control")
                comparisons = {control: gate(summary, results[control]) for control in
                               dict.fromkeys(required + spec.get("additional_controls", [])) if control in results}
                row["screen_pass"] = all(comparisons[control]["passed"] for control in required)
                row["observed_failure"] = "; ".join(control + ": " + ", ".join(result["failed_criteria"])
                                                    for control, result in comparisons.items() if not result["passed"])
                row["possible_cause_not_proven"] = spec.get("hypothesis", "Not established; inspect learning curves and policies before attributing a mechanism.")
                write_json(directory / "screening_gate.json", {"comparisons": comparisons, "required_controls": required,
                           "screen_pass": row["screen_pass"], "paper_claim_allowed": False})
        if state["status"] == "running":
            model = Path(state["model_run_directory"])
            steps = [int(path.stem.replace("model_step", "")) for path in (model / "incremental").glob("model_step*.pt")]
            row["latest_checkpoint_env_steps"] = max(steps, default=0)
        if state["status"] == "failed":
            row["observed_failure"] = state["error"]
        rows.append(row)
    write_json(campaign / "long_master.json", rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    target = campaign / "long_master.csv"
    with target.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(target.with_suffix(".csv.tmp"), target)


def supervise(campaign):
    manifest = verify_snapshot(campaign)
    with (campaign / "controller_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, handle)
    status = {"status": "preflight", "pid": os.getpid(), "completed": [], "child_pid": None}
    state_path = campaign / "queue_status.json"
    write_json(state_path, status)
    try:
        with (campaign / "preflight_tests.log").open("x", encoding="utf-8") as handle:
            subprocess.run([sys.executable, "-B", "-m", "unittest", *LONG_TESTS, "-v"],
                           cwd=campaign / "code", stdout=handle, stderr=subprocess.STDOUT, check=True)
        sequence = [("smoke", name) for name in manifest["initial_smokes"]]
        sequence += [(phase, name) for name in manifest["methods"] for phase in ("smoke", "formal")
                     if not (phase == "smoke" and name in manifest["initial_smokes"])]
        for phase, name in sequence:
            if (campaign / "PAUSE_AFTER_CURRENT").exists():
                status.update(status="paused_by_file", child_pid=None)
                return
            require_space(campaign)
            verify_snapshot(campaign)
            command = [sys.executable, "-u", "-B", str(campaign / "code/experiments/run_gsp_long_campaign.py"),
                       "worker", "--campaign", str(campaign), "--phase", phase, "--method", name]
            with (campaign / (phase + "_" + name + ".process.log")).open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=campaign, stdout=handle, stderr=subprocess.STDOUT)
                status.update(status="running", current_method=name, phase=phase, child_pid=process.pid,
                              command=subprocess.list2cmdline(command))
                write_json(state_path, status)
                while process.poll() is None:
                    time.sleep(10)
                    status["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    write_json(state_path, status)
                    refresh(campaign)
            if process.returncode:
                raise RuntimeError(phase + "/" + name + " exited " + str(process.returncode))
            finished = read_json(campaign / (phase + "_" + name + ".json"))
            assert finished["status"] == "completed" and finished["validation"]["passed"]
            status["completed"].append(phase + "/" + name)
            status["child_pid"] = None
            refresh(campaign)
        status.update(status="completed", current_method=None, phase=None, child_pid=None)
    except BaseException as error:
        status.update(status="paused_technical_failure", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(state_path, status)
        refresh(campaign)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "supervise", "worker", "refresh"))
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--phase", choices=("smoke", "formal"))
    parser.add_argument("--method")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.mode == "prepare":
        freeze(campaign)
    elif args.mode == "supervise":
        supervise(campaign)
    elif args.mode == "worker":
        if not args.phase or not args.method:
            parser.error("worker requires phase and method")
        worker(campaign, args.phase, args.method)
    else:
        refresh(campaign)


if __name__ == "__main__":
    main()
