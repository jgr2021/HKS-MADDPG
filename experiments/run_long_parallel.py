"""Two-slot fresh-run scheduler around the unchanged long-horizon worker."""
import argparse
import csv
import ctypes
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "experiments/gsp_long_20260906a"
NEW = PROJECT / "experiments/gsp_long_20260906b"
INHERITED = ("raw_mlp", "explore_geometric_stats3")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data, exclusive=False):
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def memory():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_page",
                                             "avail_page", "total_virtual", "avail_virtual", "extra")]
    s = Status()
    s.length = ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise OSError("Cannot read memory headroom")
    return {k: getattr(s, k) for k in ("load", "avail_phys", "avail_page", "total_page")}


def load(root):
    sys.path.insert(0, str(root / "code"))
    from experiments import run_gsp_long_campaign as c
    assert c.ROOT == root / "code"
    for module in c.REGISTRATIONS:
        importlib.import_module(module).register_policies()
    return c


def select_next(names, done, active):
    return next((name for name in names if name not in done and name not in active), None)


def required_controls(spec):
    return list(dict.fromkeys(["raw_mlp", spec["control"]] + spec.get("required_additional_controls", []))) if spec["control"] else []


def continuation_inventory(root, c, manifest, skipped):
    if not skipped or set(skipped) - set(manifest["methods"]):
        raise ValueError("Explicit known failed methods must be supplied")
    done, smoke_done = set(), set()
    for name, spec in manifest["methods"].items():
        for phase in ("smoke", "formal"):
            marker = root / (phase + "_" + name + ".json")
            if not marker.exists():
                if (root / phase / name / "seed_1").exists():
                    raise RuntimeError("Untracked existing artifacts: " + name)
                continue
            state = read(marker)
            if name in skipped:
                if phase == "formal" and state.get("status") != "failed":
                    raise ValueError("Only failed formal attempts may be skipped")
                continue
            if state.get("status") != "completed" or not state.get("validation", {}).get("passed"):
                raise RuntimeError("Unresolved attempt must not be retried: " + name)
            actual = c.validate_run(root / phase / name / "seed_1", c.training_args(phase), spec)
            if actual != state["validation"]:
                raise RuntimeError("Completed evidence changed: " + name)
            (done if phase == "formal" else smoke_done).add(name)
    for name in skipped:
        if read(root / ("formal_" + name + ".json")).get("status") != "failed":
            raise ValueError("Skipped method is not a failed formal attempt")
    if not done <= smoke_done:
        raise RuntimeError("Completed formal run lacks validated smoke")
    return done, smoke_done


def prepare():
    assert not NEW.exists()
    assert shutil.disk_usage(NEW.parent).free > 4 * 1024**3
    NEW.mkdir()
    shutil.copytree(OLD / "code", NEW / "code")
    manifest = read(OLD / "manifest.json")
    manifest.update(queue_rule="Maximum two independent workers; no training resume; technical errors stop new admissions",
                    inherited_completed_methods=list(INHERITED), inherited_campaign=str(OLD),
                    cancelled_attempt="contact_d4_geometry at902400 steps; retained only in prior campaign",
                    reuse_old_results="Only validated completed raw/ordinary-geometry long results, never incomplete or short runs")
    write(NEW / "manifest.json", manifest, True)
    for name in INHERITED:
        for phase in ("smoke", "formal"):
            marker = read(OLD / (phase + "_" + name + ".json"))
            assert marker["status"] == "completed" and marker["validation"]["passed"]
            shutil.copytree(OLD / phase / name, NEW / phase / name)
            shutil.copy2(OLD / (phase + "_" + name + ".json"), NEW / (phase + "_" + name + ".json"))
    c = load(NEW)
    c.verify_snapshot(NEW)
    for name in INHERITED:
        for phase in ("smoke", "formal"):
            actual = c.validate_run(NEW / phase / name / "seed_1", c.training_args(phase), manifest["methods"][name])
            assert actual == read(NEW / (phase + "_" + name + ".json"))["validation"]
    write(NEW / "parallel_preflight.json", {"passed": True, "inherited_validated": list(INHERITED),
          "max_workers": 2, "memory": memory(), "frozen_source_files": len(manifest["source_hashes"]),
          "fresh_first_methods": ["contact_d4_geometry", "contact_d4_energy"],
          "training_changes": "None; original frozen worker and ReplayBuffer"}, True)
    refresh(NEW, c)
    print("Prepared and validated " + str(NEW), flush=True)


def refresh(root, c):
    manifest = read(root / "manifest.json")
    decisions = read(root / "queue_skip_decisions.json") if (root / "queue_skip_decisions.json").exists() else {}
    states = {name: read(root / ("formal_" + name + ".json")) if (root / ("formal_" + name + ".json")).exists()
              else {"status": "queued"} for name in manifest["methods"]}
    results = {n: read(root / "formal" / n / "seed_1/summary.json") for n, s in states.items() if s["status"] == "completed"}
    rows = []
    for name, spec in manifest["methods"].items():
        s = states[name]
        row = dict(method=name, group=spec["group"], status=s["status"], seed=1, budget_env_steps=2500000,
                   budget_episodes=100000, batch_size=1024, episode_length=25, checkpoint_interval_env_steps=20000,
                   control=spec["control"], artifact_path=str(root / "formal" / name / "seed_1"),
                   observed_failure="", possible_cause_not_proven="", converged="not established",
                   provenance="validated completed result from campaign a" if name in INHERITED else "fresh campaign b")
        if name in results:
            row.update({k: results[name][k] for k in ("return", "hungarian_assignment_distance", "coverage_radius_auc",
                "collision_step_rate", "final_coverage", "final3", "global_env_steps", "env_steps_per_sec")})
            required = required_controls(spec)
            if required:
                missing = [control for control in required if control not in results]
                if missing:
                    row["comparison_status"] = "waiting for controls: " + ", ".join(missing)
                else:
                    comparisons = {control: c.gate(results[name], results[control]) for control in
                                   dict.fromkeys(required + spec.get("additional_controls", [])) if control in results}
                    row["screen_pass"] = all(comparisons[x]["passed"] for x in required)
                    row["observed_failure"] = "; ".join(x + ": " + ", ".join(v["failed_criteria"]) for x, v in comparisons.items() if not v["passed"])
                    row["possible_cause_not_proven"] = spec.get("hypothesis", "Not established")
                    write(root / "formal" / name / "seed_1/screening_gate.json", dict(comparisons=comparisons,
                          required_controls=required, screen_pass=row["screen_pass"], paper_claim_allowed=False))
        elif s["status"] == "running":
            files = (Path(s["model_run_directory"]) / "incremental").glob("model_step*.pt")
            row["latest_checkpoint_env_steps"] = max((int(p.stem.replace("model_step", "")) for p in files), default=0)
        elif s["status"] == "failed":
            row["observed_failure"] = s.get("error", "Worker failed")
        if name in decisions:
            row["queue_disposition"] = "skipped_by_user_no_retry"
            row["skip_reason"] = decisions[name]["reason"]
        rows.append(row)
    write(root / "long_master.json", rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temp = root / "long_master.csv.tmp"
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, root / "long_master.csv")


def supervise(root, continue_after_failure=False, skip_failed=()):
    c = load(root)
    manifest = c.verify_snapshot(root)
    assert read(root / "parallel_preflight.json")["passed"]
    skipped = set(skip_failed)
    if continue_after_failure:
        previous = read(root / "parallel_status.json")
        if previous["status"] != "paused_technical_failure" or previous["active"]:
            raise RuntimeError("Continuation requires a drained technical-failure pause")
        if {item["method"] for item in previous["failures"]} != skipped:
            raise ValueError("All previous failures must be explicitly acknowledged")
        # Exclusive receipt preserves the old start guard and prevents duplicate continuations.
        receipt = dict(pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       previous_status=previous, skipped=sorted(skipped), training_resume=False)
        write(root / "parallel_continuation_started_20260907.json", receipt, True)
        done, smoke_done = continuation_inventory(root, c, manifest, skipped)
        write(root / "queue_skip_decisions.json", {name: dict(
            reason="User requested no rerun; continue all other queued methods",
            failure_marker="formal_" + name + ".json", time=receipt["time"])
            for name in skipped}, True)
    else:
        if skipped:
            raise ValueError("Skip requires explicit failure continuation")
        write(root / "parallel_started.json", dict(pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z")), True)
        done, smoke_done = set(INHERITED), set(INHERITED)
    active = {}
    status = dict(status="starting", pid=os.getpid(), max_workers=2, failures=[], active=[],
                  completed=[n for n in manifest["methods"] if n in done], skipped=sorted(skipped),
                  acknowledged_failures=previous["failures"] if continue_after_failure else [])
    def launch(name, phase):
        c.require_space(root)
        c.verify_snapshot(root)
        assert not (root / phase / name / "seed_1").exists()
        command = [sys.executable, "-u", "-B", str(root / "code/experiments/run_gsp_long_campaign.py"),
                   "worker", "--campaign", str(root), "--phase", phase, "--method", name]
        with (root / (phase + "_" + name + ".process.log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        active[name] = dict(process=process, phase=phase, command=subprocess.list2cmdline(command))
    try:
        while len(done | skipped) < len(manifest["methods"]):
            for name, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                del active[name]
                marker = root / (item["phase"] + "_" + name + ".json")
                result = read(marker) if marker.exists() else {}
                if code != 0 or result.get("status") != "completed" or not result.get("validation", {}).get("passed"):
                    status["failures"].append(dict(method=name, phase=item["phase"], exit_code=code, error=result.get("error")))
                elif item["phase"] == "smoke":
                    smoke_done.add(name)
                else:
                    done.add(name)
            pause = bool(status["failures"]) or (root / "PAUSE_AFTER_CURRENT").exists()
            while not pause and len(active) < 2:
                name = select_next(manifest["methods"], done | skipped, active)
                if name is None:
                    break
                mem = memory()
                if min(mem["avail_phys"], mem["avail_page"]) < 3 * 1024**3:
                    status["admission_note"] = "Waiting for at least3GiB physical and commit allowance"
                    break
                status.pop("admission_note", None)
                launch(name, "formal" if name in smoke_done else "smoke")
            status.update(status="draining_after_failure" if status["failures"] and active else
                          "paused_technical_failure" if status["failures"] else
                          "paused_by_file" if pause and not active else "running",
                          active=[dict(method=n, phase=v["phase"], pid=v["process"].pid, command=v["command"]) for n, v in active.items()],
                          completed=[n for n in manifest["methods"] if n in done], memory=memory(),
                          checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            write(root / "parallel_status.json", status)
            refresh(root, c)
            if pause and not active:
                return
            time.sleep(10)
        status["status"] = "completed_with_skips" if skipped else "completed"
    except BaseException as error:
        status.update(status="controller_failure", error=repr(error), traceback=traceback.format_exc(),
                      active=[dict(method=n, phase=v["phase"], pid=v["process"].pid) for n, v in active.items()])
        raise
    finally:
        write(root / "parallel_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "supervise"))
    parser.add_argument("--campaign", type=Path, default=NEW)
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--skip-failed", nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        supervise(args.campaign.resolve(), args.continue_after_failure, args.skip_failed)
