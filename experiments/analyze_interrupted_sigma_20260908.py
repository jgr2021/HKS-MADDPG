"""Evaluate externally interrupted HKS runs without resuming training."""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/gsp_long_20260906b"
OUTPUT = CAMPAIGN / "analysis_interrupted_sigma_20260908"
sys.path.insert(0, str(CAMPAIGN / "code"))

EVAL_EPISODES = 500
EVAL_SEED_BASE = 1_000_000
PLANNED_ENV_STEPS = 2_500_000
SPECS = (
    {
        "method": "explore_hks_sigma075",
        "reference_method": "explore_hks_6al",
        "expected_checkpoint_step": 1_840_000,
        "model_run_directory": CAMPAIGN / "models/simple_spread/pgt_m11_s1_0907_173750f/run1",
    },
    {
        "method": "explore_hks_sigma125",
        "reference_method": "explore_hks_6al",
        "expected_checkpoint_step": 1_400_000,
        "model_run_directory": CAMPAIGN / "models/simple_spread/pgt_m12_s1_0907_203806f/run1",
    },
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_logged_steps(method):
    log = (CAMPAIGN / "formal" / method / "seed_1" / "training.log").read_text(encoding="utf-8")
    return int(re.findall(r"global_env_steps=(\d+)/2500000", log)[-1])


def latest_checkpoint(directory):
    checkpoints = list((directory / "incremental").glob("model_step*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No incremental checkpoint in " + str(directory))
    return max(checkpoints, key=lambda path: int(path.stem.replace("model_step", "")))


def summarise(rows, metrics, method, step, checkpoint, before_hash, elapsed):
    summary = {key: sum(float(row[key]) for row in rows) / len(rows) for key in metrics}
    summary.update(
        method=method,
        checkpoint_env_steps=step,
        eval_episodes=EVAL_EPISODES,
        checkpoint=str(checkpoint),
        checkpoint_sha256=before_hash,
        evaluation_seconds=elapsed,
        success_count=sum(row["final3"] for row in rows),
    )
    return summary


def evaluate_one(evaluator, metrics, method, step, checkpoint):
    before_hash = digest(checkpoint)
    started = time.perf_counter()
    rows = evaluator.evaluate_model("simple_spread", checkpoint, EVAL_EPISODES, 25, EVAL_SEED_BASE)
    if [row["test_seed"] for row in rows] != list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_EPISODES)):
        raise RuntimeError("Unexpected evaluation seed bank")
    if digest(checkpoint) != before_hash:
        raise RuntimeError("Evaluation modified checkpoint: " + str(checkpoint))
    file_name = method + "_step" + str(step) + "_episodes.csv"
    with (OUTPUT / file_name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summarise(rows, metrics, method, step, checkpoint, before_hash, time.perf_counter() - started)


def comparison(candidate, baseline):
    return {
        "return_delta": candidate["return"] - baseline["return"],
        "hungarian_distance_delta": candidate["hungarian_assignment_distance"] - baseline["hungarian_assignment_distance"],
        "coverage_auc_delta": candidate["coverage_radius_auc"] - baseline["coverage_radius_auc"],
        "collision_rate_delta": candidate["collision_step_rate"] - baseline["collision_step_rate"],
        "final_coverage_delta": candidate["final_coverage"] - baseline["final_coverage"],
        "full_coverage_rate_delta": candidate["final3"] - baseline["final3"],
    }


def main():
    from experiments import run_d4_actor_gpu_screen as evaluator
    from experiments import run_gsp_long_campaign as campaign
    import run_vector_signal_gsp_experiment as metrics
    import torch

    torch.set_num_threads(1)
    if evaluator.EVAL_DEVICE != "cpu":
        raise RuntimeError("Interrupted checkpoint analysis must use the frozen CPU evaluator")
    campaign.verify_snapshot(CAMPAIGN)
    for module in campaign.REGISTRATIONS:
        importlib.import_module(module).register_policies()

    OUTPUT.mkdir(exist_ok=False)
    reports = []
    source_hashes_before = {}
    for spec in SPECS:
        method = spec["method"]
        marker = CAMPAIGN / ("formal_" + method + ".json")
        state = read(marker)
        if state.get("status") != spec.get("expected_marker_status", "running"):
            raise RuntimeError("Expected preserved interrupted marker for " + method)
        last_logged = last_logged_steps(method)
        checkpoint = latest_checkpoint(spec["model_run_directory"])
        step = int(checkpoint.stem.replace("model_step", ""))
        if step != spec["expected_checkpoint_step"] or not step <= last_logged < PLANNED_ENV_STEPS:
            raise RuntimeError("Unexpected preserved checkpoint for " + method)

        sources = {
            method: checkpoint,
            "raw_mlp": CAMPAIGN / "formal/raw_mlp/seed_1/checkpoints" / checkpoint.name,
            spec["reference_method"]: CAMPAIGN / "formal" / spec["reference_method"] / "seed_1/checkpoints" / checkpoint.name,
        }
        for source_method, source in sources.items():
            if not source.is_file():
                raise FileNotFoundError("Missing matched checkpoint for " + source_method + ": " + str(source))
            source_hashes_before[str(source)] = digest(source)

        results = {}
        for source_method, source in sources.items():
            results[source_method] = evaluate_one(evaluator, metrics.METRICS, source_method, step, source)
            print(json.dumps(results[source_method]), flush=True)

        if any(digest(Path(path)) != before_hash for path, before_hash in source_hashes_before.items()):
            raise RuntimeError("A checkpoint hash changed during evaluation")
        reports.append(
            {
                "method": method,
                "reference_method": spec["reference_method"],
                "pre_reconciliation_marker": state,
                "last_logged_env_steps": last_logged,
                "checkpoint_env_steps": step,
                "planned_env_steps": PLANNED_ENV_STEPS,
                "checkpoint_selection": "Last saved checkpoint, not selected by performance",
                "results": results,
                "vs_raw_mlp": comparison(results[method], results["raw_mlp"]),
                "vs_reference_hks": comparison(results[method], results[spec["reference_method"]]),
            }
        )

    campaign.verify_snapshot(CAMPAIGN)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "interrupted_checkpoint_evaluated_not_completed_training",
        "training_steps_added": 0,
        "evaluation_device": "cpu",
        "evaluation_threads": 1,
        "evaluation_seed_first": EVAL_SEED_BASE,
        "evaluation_seed_last": EVAL_SEED_BASE + EVAL_EPISODES - 1,
        "checkpoint_hashes": source_hashes_before,
        "caveat": "One training seed; post-hoc interrupted-budget analysis, not a 2.5M endpoint or proof of robust superiority",
        "runs": reports,
    }
    with (OUTPUT / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("Analysis saved: " + str(OUTPUT / "analysis.json"), flush=True)


if __name__ == "__main__":
    main()
