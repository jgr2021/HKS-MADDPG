"""Immutable sequential screening. Does not modify the corrected training loop."""

import argparse
import csv
import hashlib
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

METHODS = {
    "raw_mlp": {"actor_model": "mlp", "short": "raw", "control": None,
                "factor": "fresh GPU baseline"},
    "contact_d4_geometry": {"actor_model": "explore_contact_d4_geometry", "short": "cgeom",
                            "control": "raw_mlp", "factor": "contact-corrected action geometry"},
    "contact_d4_energy": {"actor_model": "explore_contact_d4_energy", "short": "cgsp",
                          "control": "contact_d4_geometry", "factor": "add Dirichlet energies"},
}
KEYS = ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate")
TESTS = ("tests.test_exploration_contact_policies", "tests.test_d4_graph_residual",
         "tests.test_gsp_exploration_campaign")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(Path(root).rglob("*.py"))}


def verify_snapshot(campaign):
    manifest = read_json(campaign / "manifest.json")
    if code_hashes(campaign / "code") != manifest["source_hashes"]:
        raise RuntimeError("Frozen Python source changed; queue paused without deleting evidence")
    return manifest


def gate(candidate, control):
    delta = {key: candidate[key] - control[key] for key in KEYS}
    failed = [name for name, passed in (
        ("return", delta["return"] > 0),
        ("assignment distance", delta["hungarian_assignment_distance"] < 0),
        ("coverage AUC", delta["coverage_radius_auc"] > 0),
        ("collision rate", delta["collision_step_rate"] <= 0),
    ) if not passed]
    return {"passed": not failed, "deltas": delta, "failed_criteria": failed,
            "interpretation": "exploratory seed-1 screen, not a training-seed confidence interval"}


def run_directory(campaign, phase, method, spec):
    if phase == "formal" and spec.get("reuse_from"):
        return Path(spec["reuse_from"])
    return campaign / phase / method / "seed_1"


def freeze(campaign, study="contact"):
    methods = METHODS
    registrations = ["utils.exploration_contact_policies"]
    tests = TESTS
    if study == "topology":
        from utils.exploration_topology_policies import experiment_methods
        source = ROOT / "experiments/gsp_exploration_20260906_phase1"
        if read_json(source / "queue_status.json")["status"] != "phase1_completed":
            raise RuntimeError("Complete and validate phase1 before launching topology study")
        verify_snapshot(source)
        methods = experiment_methods(source / "formal/raw_mlp/seed_1")
        registrations += ["utils.exploration_topology_policies"]
        tests += ("tests.test_exploration_topology_policies",)
    elif study == "weights":
        from utils.exploration_weight_policies import experiment_methods
        previous = ROOT / "experiments/gsp_exploration_20260906_phase2a"
        if read_json(previous / "queue_status.json")["status"] != "study_completed":
            raise RuntimeError("Finish topology study before the edge-weight study")
        verify_snapshot(previous)
        audit = read_json(previous / "analysis_20260906/audit.json")
        if not audit["passed"]:
            raise RuntimeError("Topology result audit is required")
        methods = experiment_methods(ROOT)
        if sum(not spec.get("reuse_from") for spec in methods.values()) > 6:
            raise RuntimeError("This batch reserves only six new formal runs")
        registrations += ["utils.exploration_topology_policies", "utils.exploration_weight_policies"]
        tests += ("tests.test_exploration_topology_policies", "tests.test_exploration_weight_policies")
    elif study == "operators":
        from utils.exploration_operator_policies import experiment_methods
        previous = ROOT / "experiments/gsp_exploration_20260906_phase2b"
        if read_json(previous / "queue_status.json")["status"] != "study_completed":
            raise RuntimeError("Finish the weight study before the operator study")
        verify_snapshot(previous)
        audit = read_json(previous / "analysis_20260906_completed/audit.json")
        if not (audit["passed"] and audit["study_complete"] and audit["new_completed"] == 6):
            raise RuntimeError("A complete weight-study audit is required")
        methods = experiment_methods(ROOT)
        if sum(not spec.get("reuse_from") for spec in methods.values()) != 4:
            raise RuntimeError("The operator group reserves exactly four new formal runs")
        registrations += ["utils.exploration_topology_policies", "utils.exploration_weight_policies",
                          "utils.exploration_operator_policies"]
        tests += ("tests.test_exploration_topology_policies", "tests.test_exploration_weight_policies",
                  "tests.test_exploration_operator_policies")
    elif study == "descriptors":
        from utils.exploration_descriptor_policies import experiment_methods
        previous = ROOT / "experiments/gsp_exploration_20260906_phase2c"
        if read_json(previous / "queue_status.json")["status"] != "study_completed":
            raise RuntimeError("Finish the operator study before the descriptor study")
        verify_snapshot(previous)
        audit = read_json(previous / "analysis_20260906_completed/audit.json")
        if not (audit["passed"] and audit["study_complete"] and audit["new_completed"] == 4):
            raise RuntimeError("A complete operator-study audit is required")
        profile_path = ROOT / "experiments/gsp_descriptor_preflight_20260906/profile.json"
        profile = read_json(profile_path)
        if not profile["passed"] or any(sha256(ROOT / name) != digest for name, digest in profile["source_hashes"].items()):
            raise RuntimeError("Descriptor profile is missing or its source changed")
        methods = experiment_methods(ROOT)
        if sum(not spec.get("reuse_from") for spec in methods.values()) != 4:
            raise RuntimeError("The final descriptor group reserves exactly four new formal runs")
        registrations += ["utils.exploration_topology_policies", "utils.exploration_weight_policies",
                          "utils.exploration_operator_policies", "utils.exploration_descriptor_policies"]
        tests += ("tests.test_exploration_topology_policies", "tests.test_exploration_weight_policies",
                  "tests.test_exploration_operator_policies", "tests.test_exploration_descriptor_policies")
    campaign.mkdir(parents=True, exist_ok=False)
    code = campaign / "code"
    code.mkdir()
    paths = list(ROOT.glob("*.py"))
    for name in ("utils", "algorithms", "tests"):
        paths += list((ROOT / name).rglob("*.py"))
    paths += [ROOT / "experiments" / name for name in (
        "run_passive_gsp_topology_pilot.py", "run_d4_actor_gpu_screen.py",
        "run_gsp_exploration_campaign.py",
    )]
    for source in paths:
        target = code / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    import multiagent
    mpe = Path(multiagent.__file__).parent
    for source in mpe.rglob("*.py"):
        target = code / "multiagent" / source.relative_to(mpe)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    from experiments.run_d4_actor_gpu_screen import training_args
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "original_repo": str(ROOT),
                "source_hashes": code_hashes(code), "methods": methods, "seed": 1,
                "study": study, "registration_modules": registrations, "tests": tests,
                "formal": vars(training_args("formal")), "smoke": vars(training_args("smoke")),
                "protocol": "actor-only; raw env/replay 18D; unchanged joint raw critic 69D; original reward, actions and updates",
                "training_device": "cuda", "rollout_and_eval_device": "cpu",
                "screening_rule": "return up, Hungarian distance down, coverage-radius AUC up, collisions not up against matched control AND raw; seed1 only",
                "holdout_rule": "Final seeds are a fixed screening bank. Many-method selection is exploratory. Confirmation requires unseen training/evaluation seeds.",
                "pause_rule": "Technical failure stops queue. Behavioral gate failure is retained and next predeclared method continues.",
                "source_mpe": str(mpe), "resume": False}
    if study == "topology":
        manifest["graph_protocol"] = {
            "nodes": "focal self at [0,0], other agents in observed order, then landmarks",
            "weights": "exp(-d^2/(2*sigma^2)), sigma_AA=0.8, sigma_AL=0.6, sigma_LL=0.6; all relation strengths=1",
            "operator": "I-D^(-1/2) W D^(-1/2); degree clamped at 1e-12; isolated diagonal=1",
            "hks_times": [.5, 1., 2.], "focal_index": 0, "output_features": 3,
            "geometry_control": "sum landmark nearest-agent distances; sum 3 unordered AA Gaussian affinities; mean focal-landmark distance",
            "actor": "raw18 plus 3 features into unchanged 21-64-64-5 MLP with original BatchNorm",
            "graph_gradients": "fixed deterministic preprocessing under no_grad; no learned edges",
            "precision": "float32 batched eigendecomposition; no CPU-GPU feature transfers",
        }
    if study == "weights":
        manifest["graph_protocol"] = {
            "graph": "6AL; identical normalized Laplacian and focal HKS readout to topology 6AL",
            "hks_times": [.5, 1., 2.], "degree_floor": 1e-12,
            "default_sigma_AL": .6, "sigma_multipliers": [.75, 1.25],
            "bounded_inverse": "sigma/(sigma+d), sigma=0.6",
            "soft_radius": "Gaussian(sigma=.6) * sigmoid((1.0-d)/0.15)",
            "knn": "k=2 among AL-permitted neighbors per node, union symmetrization; Gaussian sigma=.6",
            "constant_control": "raw18 plus exact (1+exp(-2*t))/2 for three t; same 21D MLP; no graph information",
            "binary_graph": "fixed K3,3 binary adjacency has a constant HKS; diagnostic only, not a training run",
            "queue_amendment": "constant-star control replaces planned binary training after star-HKS degeneracy audit; no added run budget",
            "additional_comparisons": "each weight method also compared with default 6AL and constant feature control; original primary gate unchanged",
        }
    if study == "operators":
        manifest["graph_protocol"] = {
            "graph": "Fixed 6AL; self origin, observed other agents then landmarks; Gaussian sigma_AL=.6",
            "hks_times": [.5, 1., 2.], "rwse_steps": [2, 4, 8], "degree_floor": 1e-12,
            "combinatorial": "L=D-W; focal diag exp(-t L); eigenvalues lower-clamped at zero only",
            "self_loop": "W'=W+I with loop weight=1, D'=rowsum W'; L'=I-D'^(-1/2) W' D'^(-1/2)",
            "rwse": "diag(S^k)_0, S=D^(-1/2) W D^(-1/2), k=2,4,8",
            "lazy_rwse": "diag(((I+S)/2)^k)_0, k=2,4,8; distinct from adding unit loops to W",
            "equivalence": "diag(P^k)=diag(S^k), P=D^-1 W; train only one implementation, not both",
            "bipartite_parity": "6AL without loops has zero odd-step RWSE; choose fixed nonconstant even powers",
            "isolation": "L_combinatorial=0 and self-loop L=0 => HKS=1; S=0 => RWSE=0; lazy=(.5I) => .5^k. Degree floor makes P substochastic below threshold.",
            "actor": "raw18 plus 3 features; same 21-64-64-5 MLP initialization and original BatchNorm",
            "gradient_device": "float32, actor-local no_grad preprocessing on input device; no added transfers",
            "comparisons": "primary vs raw AND geometry unchanged; auxiliary vs default normalized HKS and constant control",
            "scale_limit": "Identical HKS times are numerical controls, not matched physical diffusion times across L definitions; no retuning on final eval",
            "phase2_budget": {"cap": 20, "prior_new_formal_runs": 12, "this_group": 4, "remaining_after_reservation": 4},
        }
    if study == "descriptors":
        manifest["graph_protocol"] = {
            "graph": "Fixed 6AL Gaussian sigma_AL=.6, focal origin and relative observed positions only",
            "operator": "Ln=I-D^(-1/2) W D^(-1/2), degree floor=1e-12; no self loops",
            "geometry": "Three direct W[0,landmark] affinities in observed landmark order",
            "wks": "sum_k u_k(0)^2 g_kj / sum_k g_kj; g=exp(-.5*((log(lambda)-log(center))/.5)^2); positive lambda>1e-6",
            "wks_centers": [.25, .75, 1.5], "wks_log_bandwidth": .5,
            "wks_reference": "https://imagine.enpc.fr/~aubrym/projects/wks/texts/2011-wave-kernel-signature.pdf equation10; graph adaptation with fixed 3 bands, not original shape-analysis settings",
            "landmark_heat": "exp(-Ln)[0,3:6], fixed t=1; returns 3 landmark-aligned values",
            "regularized_resistance": ".05*(K00+Kll-K0l-Kl0), K=(Ln+.1I)^-1; normalized resolvent contrast, NOT ordinary electrical effective resistance",
            "resistance_regularization": .1, "resistance_output_scale": .05,
            "isolation": "W=0 => Ln=I; geometry=0, heat offdiag=0, WKS=1/6, scaled normalized contrast=1/11",
            "actor": "same initialization and parameter count as 21D HKS actor; raw18 env/replay and raw69 joint critic unchanged",
            "channel_semantics": "WKS channels are fixed bands; geometry/heat/resistance channels follow landmark order. Features are landmark-equivariant, not claiming whole-MLP permutation equivariance.",
            "screening": "Original primary vs raw AND geometric_stats3 retained; heat/resistance additionally REQUIRED to pass landmark_affinity3. No relaxed gate.",
            "profile_path": str(profile_path), "profile_sha256": sha256(profile_path),
            "phase2_budget": {"cap": 20, "prior_new_formal_runs": 16, "this_group": 4, "remaining_after_reservation": 0},
            "stop_after": "End this bounded screen after four formal runs; analyze and ask user before more training or seeds",
        }
    write_json(campaign / "manifest.json", manifest)
    write_json(campaign / "queue_status.json", {"status": "prepared", "pid": None, "completed": []})
    refresh_table(campaign)


def refresh_table(campaign):
    manifest = read_json(campaign / "manifest.json")
    results = {}
    rows = []
    for method, spec in manifest["methods"].items():
        run = run_directory(campaign, "formal", method, spec)
        state = campaign / f"formal_{method}.json"
        status = read_json(state) if state.exists() else {"status": "queued"}
        row = {"method": method, "factor": spec["factor"], "seed": 1,
               "budget_env_steps": 100000, "checkpoint_interval": 20000, "control": spec["control"],
               "status": status["status"], "cohort": campaign.name, "n_training_seeds": 1,
               "observed_failure": "", "possible_cause_not_proven": "", "artifact_path": str(run)}
        row["reused"] = bool(spec.get("reuse_from"))
        if status["status"] == "completed":
            result = read_json(run / "summary.json")
            results[method] = result
            row.update({key: result[key] for key in KEYS + ("final_coverage", "env_steps_per_sec", "global_env_steps")})
            if spec["control"] in results:
                decision = gate(result, results[spec["control"]])
                raw_decision = gate(result, results["raw_mlp"])
                row.update(screen_pass=decision["passed"] and raw_decision["passed"],
                           observed_failure=", ".join(decision["failed_criteria"]),
                           raw_failed_criteria=", ".join(raw_decision["failed_criteria"]))
                row.update({"delta_" + key: value for key, value in decision["deltas"].items()})
                if not decision["passed"]:
                    row["possible_cause_not_proven"] = spec.get("hypothesis", (
                        "Hypotheses only: feature scaling, residual decision changes, missing neighbor velocities/actions; requires checkpoint/masking diagnostics."
                    ))
                additional = {name: gate(result, results[name]) for name in spec.get("additional_controls", []) if name in results}
                required = spec.get("required_additional_controls", [])
                if any(name not in additional for name in required):
                    raise RuntimeError("Required matched control must complete before candidate comparison")
                row["screen_pass"] = row["screen_pass"] and all(additional[name]["passed"] for name in required)
                row["additional_control_failures"] = "; ".join(f"{name}: {', '.join(item['failed_criteria'])}" for name, item in additional.items() if not item["passed"])
                gate_dir = campaign / "comparisons" / method if spec.get("reuse_from") else run
                write_json(gate_dir / "screening_gate.json", {"matched_control": decision, "raw": raw_decision,
                           "additional_controls": additional, "required_additional_controls": required, "paper_claim_allowed": False})
        elif status["status"] == "failed":
            row["observed_failure"] = status.get("error", "technical failure")
            row["possible_cause_not_proven"] = "Technical failure, not evidence against the method. Queue paused."
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    target = campaign / "exploration_master.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    write_json(campaign / "exploration_master.json", rows)
    display = ("method", "status", "return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate", "screen_pass")
    lines = ["# Sequential GSP Exploration", "", "Seed 1 is screening, not a paper-level result. Missing results are not zeros.", "",
             "| " + " | ".join(display) + " |", "| " + " | ".join("---" for _ in display) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(round(row[key], 6)) if isinstance(row.get(key), float)
                                        else str(row.get(key, "")) for key in display) + " |")
    (campaign / "EXPLORATION_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def worker(campaign, phase, method):
    manifest = verify_snapshot(campaign)
    if Path(__file__).resolve().parents[1] != campaign / "code":
        raise RuntimeError("Workers must execute the frozen copy")
    import importlib
    for module in manifest.get("registration_modules", ["utils.exploration_contact_policies"]):
        importlib.import_module(module).register_policies()
    from experiments import run_d4_actor_gpu_screen as screen
    from experiments import run_passive_gsp_topology_pilot as protocol
    import run_vector_signal_gsp_experiment as metrics
    from utils.cuda_protocol_audit import CudaAuditedMADDPG, require_cuda
    import torch
    if os.environ.get("MADDPG_FORCE_CPU") == "1":
        raise RuntimeError("CPU training override conflicts with locked CUDA protocol")
    runtime = require_cuda()
    args = screen.training_args(phase)
    if vars(args) != manifest[phase]:
        raise RuntimeError("Training parameters differ from manifest")
    if phase == "formal":
        smoke = read_json(campaign / f"smoke_{method}.json")
        if smoke["status"] != "completed" or not smoke["validation"]["passed"]:
            raise RuntimeError("Successful fresh smoke required")
        screen.validate_run(campaign / "smoke" / method / "seed_1", screen.training_args("smoke"))
    directory = campaign / phase / method / "seed_1"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite or resume {directory}")
    spec = manifest["methods"][method]
    if spec.get("reuse_from"):
        raise RuntimeError("A reused result must never launch training")
    protocol.METHODS = {method: {**spec, "env_id": "simple_spread", "actor_input_dim": spec.get("actor_input_dim", 18)}}
    protocol.MADDPG = CudaAuditedMADDPG
    protocol.USE_CUDA = True
    protocol.METRICS = metrics.METRICS
    protocol.evaluate_model = screen.evaluate_model
    CudaAuditedMADDPG.device_audit_path = directory / "gpu_device_audit.json"
    state_path = campaign / f"{phase}_{method}.json"
    state = {"status": "running", "pid": os.getpid(), "phase": phase, "method": method,
             "runtime": runtime, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "command": subprocess.list2cmdline([sys.executable] + sys.argv)}
    write_json(state_path, state)
    try:
        os.chdir(campaign)
        torch.set_num_threads(args.n_training_threads)
        summary, curve = protocol.train_one(method, 1, args, campaign / phase,
                                            time.strftime("%Y%m%d_%H%M%S") + "_" + phase)
        validation = screen.validate_run(directory, args)
        shapes = read_json(directory / "tensor_shape_report.json")
        for name in ("actor_augmented_tensor_shape", "target_actor_augmented_tensor_shape"):
            if shapes[name] != [64, spec.get("actor_input_dim", 18)]:
                raise RuntimeError(f"Wrong {name}: {shapes[name]}")
        if len(curve) != len(args.checkpoint_steps):
            raise RuntimeError("Missing checkpoint evaluations")
        if summary["eval_episodes"] != args.eval_episodes:
            raise RuntimeError("Wrong final evaluation episode count")
        for step in args.checkpoint_steps:
            checkpoint = directory / "checkpoints" / f"model_step{step}.pt"
            loaded = screen.MADDPG.init_from_save(checkpoint)
            for agent in loaded.agents:
                for name in ("policy", "critic", "target_policy", "target_critic"):
                    if not all(torch.isfinite(value).all() for value in getattr(agent, name).state_dict().values()):
                        raise RuntimeError(f"Nonfinite weights at {step}")
        verify_snapshot(campaign)
        state.update(status="completed", validation=validation, summary_path=str(directory / "summary.json"))
        write_json(directory / "validation.json", validation)
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(state_path, state)


def validate_reuse(campaign, method, spec):
    source = Path(spec["reuse_from"])
    old_campaign = source.parents[2]
    old_manifest = verify_snapshot(old_campaign)
    new_manifest = verify_snapshot(campaign)
    if old_manifest["formal"] != new_manifest["formal"]:
        raise RuntimeError("Reused baseline has different training/evaluation parameters")
    allowed = {"raw_mlp": "mlp", "explore_geometric_stats3": "explore_geometric_stats3",
               "explore_hks_6al": "explore_hks_6al", "explore_constant_star3": "explore_constant_star3"}
    if method not in allowed or spec["actor_model"] != allowed[method]:
        raise RuntimeError("Reuse is restricted to audited, unchanged controls")
    old_spec = old_manifest["methods"][method]
    if any(spec.get(key, 18) != old_spec.get(key, 18) for key in ("actor_model", "actor_input_dim")):
        raise RuntimeError("Reused actor identity or dimensions differ")
    files = [name for name in old_manifest["source_hashes"] if name.startswith(("utils/", "algorithms/", "multiagent/"))]
    files += ["main.py", "experiments/run_passive_gsp_topology_pilot.py", "run_vector_signal_gsp_experiment.py",
              "experiments/run_d4_actor_gpu_screen.py"]
    for file in files:
        if old_manifest["source_hashes"][file] != new_manifest["source_hashes"][file]:
            raise RuntimeError(f"Reused model source mismatch: {file}")
    import importlib
    for module in new_manifest["registration_modules"]:
        importlib.import_module(module).register_policies()
    from experiments.run_d4_actor_gpu_screen import validate_run, training_args
    validated = validate_run(source, training_args("formal"))
    previous = read_json(old_campaign / f"formal_{method}.json")
    if previous["status"] != "completed" or previous["validation"] != validated:
        raise RuntimeError("Reused model validation or checkpoint hash differs")
    record = {"status": "completed", "reused": True, "validation": validated,
              "summary_path": str(source / "summary.json"),
              "evidence_hashes": {name: sha256(source / name) for name in (
                  "resolved_config.json", "training_counters.json", "summary.json",
                  "gpu_device_audit.json", "per_evaluation_episode_metrics.csv", "learning_curve_eval.csv")}}
    write_json(campaign / f"formal_{method}.json", record)


def supervise(campaign):
    manifest = verify_snapshot(campaign)
    with (campaign / "controller_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, handle)
    status = {"status": "preflight", "pid": os.getpid(), "completed": [], "child_pid": None}
    state_path = campaign / "queue_status.json"
    write_json(state_path, status)
    try:
        with (campaign / "preflight_tests.log").open("w", encoding="utf-8") as handle:
            tests = subprocess.run([sys.executable, "-B", "-m", "unittest", *manifest.get("tests", TESTS), "-v"],
                                   cwd=campaign / "code", stdout=handle, stderr=subprocess.STDOUT)
        tests.check_returncode()
        verify_snapshot(campaign)
        script = campaign / "code" / "experiments" / "run_gsp_exploration_campaign.py"
        for method, spec in manifest["methods"].items():
            if spec.get("reuse_from"):
                validate_reuse(campaign, method, spec)
                status["completed"].append(f"reused/{method}")
                refresh_table(campaign)
                continue
            for phase in ("smoke", "formal"):
                if (campaign / "PAUSE_AFTER_CURRENT").exists():
                    status.update(status="paused_by_file", child_pid=None)
                    return
                command = [sys.executable, "-B", str(script), "worker", "--campaign", str(campaign),
                           "--phase", phase, "--method", method]
                with (campaign / f"{phase}_{method}.process.log").open("x", encoding="utf-8") as handle:
                    process = subprocess.Popen(command, cwd=campaign, stdout=handle, stderr=subprocess.STDOUT)
                    status.update(status="running", current_method=method, phase=phase, child_pid=process.pid,
                                  command=subprocess.list2cmdline(command))
                    write_json(state_path, status)
                    while process.poll() is None:
                        time.sleep(10)
                        status["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                        write_json(state_path, status)
                        refresh_table(campaign)
                if process.returncode:
                    raise RuntimeError(f"{phase}/{method} exited {process.returncode}; inspect process log")
                finished = read_json(campaign / f"{phase}_{method}.json")
                if finished["status"] != "completed":
                    raise RuntimeError("Worker exited without validated completion")
                status["completed"].append(f"{phase}/{method}")
                status["child_pid"] = None
                refresh_table(campaign)
        status.update(status="phase1_completed" if manifest.get("study", "contact") == "contact" else "study_completed",
                      current_method=None, phase=None)
    except BaseException as error:
        status.update(status="paused_technical_failure", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(state_path, status)
        refresh_table(campaign)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "supervise", "worker", "refresh"))
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--phase", choices=("smoke", "formal"))
    parser.add_argument("--method")
    parser.add_argument("--study", choices=("contact", "topology", "weights", "operators", "descriptors"), default="contact")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.mode == "prepare":
        freeze(campaign, args.study)
    elif args.mode == "supervise":
        supervise(campaign)
    elif args.mode == "worker":
        if not args.phase or not args.method:
            parser.error("worker requires --phase and --method")
        worker(campaign, args.phase, args.method)
    else:
        refresh_table(campaign)


if __name__ == "__main__":
    main()
