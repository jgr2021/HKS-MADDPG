"""Refresh the catalogue analysis workbook when experiment artifacts change."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments" / "catalog76_2m_20260911"
UPDATER = (
    ROOT
    / "outputs"
    / "01a07266-cabc-77f1-8789-cd8f70798e62"
    / "long_exploration"
    / "update_catalog.mjs"
)
STATUS = CAMPAIGN / "analysis_table_watcher_status.json"
LOG = CAMPAIGN / "analysis_table_watcher.log"


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_signature() -> tuple[object, ...]:
    state = read_json(CAMPAIGN / "catalog_status.json")
    relevant_state = {
        "status": state.get("status"),
        "completed": state.get("completed", []),
        "skipped": state.get("skipped", []),
        "inherited_completed": state.get("inherited_completed", []),
        "active": [
            {"method": item.get("method"), "phase": item.get("phase")}
            for item in state.get("active", [])
        ],
        "failures": state.get("failures", []),
    }
    signature: list[object] = [
        ("catalog_status", json.dumps(relevant_state, sort_keys=True))
    ]
    for path in sorted(CAMPAIGN.glob("analysis_catalog_*.json")):
        stat = path.stat()
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def refresh(node: Path) -> None:
    result = subprocess.run(
        [str(node), str(UPDATER)],
        cwd=UPDATER.parent,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] exit={result.returncode}\n")
        if result.stdout:
            stream.write(result.stdout.rstrip() + "\n")
        if result.stderr:
            stream.write(result.stderr.rstrip() + "\n")
    if result.returncode:
        raise RuntimeError(f"Workbook updater exited with {result.returncode}")


def terminal_campaign() -> bool:
    state = read_json(CAMPAIGN / "catalog_status.json")
    handoff = CAMPAIGN / 'priority_handoff_20260919.json'
    if handoff.exists() and read_json(handoff).get('status') == 'priority_queue_running':
        # The replacement controller publishes its first state just after the watcher starts.
        if state.get('status') == 'paused_by_user':
            return False
    return state.get("status") != "running" and not state.get("active")


def run(node: Path, interval: float, once: bool) -> None:
    if not node.is_file():
        raise FileNotFoundError(node)
    if not UPDATER.is_file():
        raise FileNotFoundError(UPDATER)

    last_signature = None
    while True:
        signature = artifact_signature()
        if signature != last_signature:
            try:
                refresh(node)
                last_signature = signature
                write_json(
                    STATUS,
                    {
                        "status": "watching" if not once else "completed_once",
                        "pid": os.getpid(),
                        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "artifact_count": len(signature),
                    },
                )
            except Exception as error:
                write_json(
                    STATUS,
                    {
                        "status": "refresh_failed",
                        "pid": os.getpid(),
                        "error": repr(error),
                        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                raise

        if once:
            return
        if terminal_campaign():
            write_json(
                STATUS,
                {
                    "status": "campaign_finished",
                    "pid": os.getpid(),
                    "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            return
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    run(args.node.resolve(), args.interval, args.once)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(repr(exc), file=sys.stderr)
        raise
