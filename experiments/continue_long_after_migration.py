"""Audited relocation and fresh-phase continuation; never resumes model training."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import traceback

SOURCE = Path("C:/Users/10431/gsp_runs/long_20260906a")
DEST = Path("D:/Documents/Python Scripts/maddpg-pytorch-GSP-v1/experiments/gsp_long_20260906a")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def inventory(root):
    entries = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            info = path.lstat()
            if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise RuntimeError("Refusing reparse point: " + str(path))
        for name in files:
            path = Path(current) / name
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            entries[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return entries


def load_frozen(root):
    sys.path.insert(0, str(root / "code"))
    from experiments import run_gsp_long_campaign as campaign
    assert campaign.ROOT == root / "code"
    for name in campaign.REGISTRATIONS:
        importlib.import_module(name).register_policies()
    return campaign


def validated_sequence(root, campaign):
    manifest = campaign.verify_snapshot(root)
    sequence = [("smoke", name) for name in manifest["initial_smokes"]]
    sequence += [(phase, name) for name in manifest["methods"] for phase in ("smoke", "formal")
                 if not (phase == "smoke" and name in manifest["initial_smokes"])]
    completed, pending = [], []
    for phase, name in sequence:
        marker = root / (phase + "_" + name + ".json")
        directory = root / phase / name / "seed_1"
        if marker.exists():
            state = read(marker)
            if state["status"] != "completed":
                raise RuntimeError("Refusing incomplete phase: " + str(marker))
            actual = campaign.validate_run(directory, campaign.training_args(phase), manifest["methods"][name])
            assert actual == state["validation"] == read(directory / "validation.json")
            completed.append(phase + "/" + name)
        else:
            if directory.exists() or (root / (phase + "_" + name + ".process.log")).exists():
                raise RuntimeError("Refusing existing unvalidated artifacts")
            pending.append((phase, name))
    return completed, pending


def copy_and_validate():
    assert SOURCE.resolve() == SOURCE and DEST.resolve() == DEST
    assert not SOURCE.is_symlink() and not DEST.exists()
    assert read(SOURCE / "queue_status.json")["status"] == "paused_by_file"
    assert read(SOURCE / "formal_raw_mlp.json")["status"] == "completed"
    before = inventory(SOURCE)
    total = sum(item["bytes"] for item in before.values())
    assert shutil.disk_usage(DEST.parent).free > total + 2 * 1024**3
    shutil.copytree(SOURCE, DEST, symlinks=True)
    assert inventory(SOURCE) == before == inventory(DEST)
    ops = DEST / "operations"
    ops.mkdir()
    write_new(ops / "migration_inventory.json", {
        "source": str(SOURCE), "destination": str(DEST), "file_count": len(before),
        "total_bytes": total, "files": before, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "historical_absolute_paths": "Preserved as provenance. Resolve old source prefix against destination for relocated artifacts."})
    campaign = load_frozen(DEST)
    completed, pending = validated_sequence(DEST, campaign)
    assert completed == ["smoke/raw_mlp", "smoke/explore_regularized_resistance3", "formal/raw_mlp"]
    shutil.copy2(__file__, ops / "continue_long_after_migration.py")
    write_new(ops / "migration_validation.json", {
        "passed": True, "completed": completed, "pending": pending,
        "frozen_source_files": len(campaign.verify_snapshot(DEST)["source_hashes"]),
        "continuation": "Independent controller; original frozen worker; no replay cache applied; no model resume.",
        "source_deleted": False})
    print(json.dumps({"verified_files": len(before), "bytes": total, "completed": completed,
                      "pending_phases": len(pending)}, indent=2), flush=True)


def continue_queue():
    root = DEST
    ops = root / "operations"
    assert read(ops / "migration_validation.json")["passed"]
    assert read(ops / "cleanup_receipt.json")["source_deleted"] and not SOURCE.exists()
    campaign = load_frozen(root)
    completed, pending = validated_sequence(root, campaign)
    write_new(ops / "continuation_started.json", {"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    assert (root / "PAUSE_AFTER_CURRENT").exists()
    (root / "PAUSE_AFTER_CURRENT").rename(ops / "PAUSE_AFTER_CURRENT.migration_evidence")
    status = {"status": "starting", "pid": os.getpid(), "completed": completed, "child_pid": None}
    state_path = ops / "continuation_status.json"
    try:
        for phase, name in pending:
            if (root / "PAUSE_AFTER_CURRENT").exists():
                status["status"] = "paused_by_file"
                return
            campaign.require_space(root)
            campaign.verify_snapshot(root)
            command = [sys.executable, "-u", "-B", str(root / "code/experiments/run_gsp_long_campaign.py"),
                       "worker", "--campaign", str(root), "--phase", phase, "--method", name]
            with (root / (phase + "_" + name + ".process.log")).open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
                status.update(status="running", current_method=name, phase=phase, child_pid=process.pid,
                              command=subprocess.list2cmdline(command))
                campaign.write_json(state_path, status)
                while process.poll() is None:
                    time.sleep(10)
                    status["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    campaign.write_json(state_path, status)
                    campaign.refresh(root)
            if process.returncode:
                raise RuntimeError(phase + "/" + name + " exited " + str(process.returncode))
            marker = read(root / (phase + "_" + name + ".json"))
            assert marker["status"] == "completed" and marker["validation"]["passed"]
            status["completed"].append(phase + "/" + name)
            status["child_pid"] = None
            campaign.refresh(root)
        status.update(status="completed", current_method=None, phase=None, child_pid=None)
    except BaseException as error:
        status.update(status="paused_technical_failure", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        campaign.write_json(state_path, status)
        campaign.refresh(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("copy-validate", "continue"))
    args = parser.parse_args()
    if args.mode == "copy-validate":
        copy_and_validate()
    else:
        continue_queue()
