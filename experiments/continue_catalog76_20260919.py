"""Continue unstarted catalogue methods after the September 16 interruption."""
import json
import os
import subprocess
import sys
import time

import continue_catalog76_20260915 as previous


DEST = previous.DEST
RECEIPT = DEST / 'continuation_20260919_started.json'
previous.SKIPPED = previous.SKIPPED | {
    'catalog_affinity_filter', 'catalog_advantage_filter'
}
previous.RECEIPT = RECEIPT


def process_inventory():
    command = ['powershell.exe', '-NoProfile', '-Command',
               'Get-Process | Select-Object -ExpandProperty Id']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return {int(line) for line in result.stdout.splitlines() if line.strip().isdigit()}


def main():
    mode = sys.argv[1:]
    if mode not in (['preflight'], ['run']):
        raise SystemExit('Use preflight or run')
    if RECEIPT.exists():
        raise RuntimeError('Continuation already started; do not duplicate')
    live_pids = process_inventory()
    previous.pid_alive = lambda pid: int(pid) in live_pids
    report = previous.preflight()
    if mode == ['preflight']:
        print(json.dumps({key: report[key] for key in
            ('completed', 'skipped', 'skip_details', 'pending', 'frozen_files')}, indent=2))
        return
    with RECEIPT.open('x', encoding='utf-8') as stream:
        json.dump(dict(report, pid=os.getpid(), started=time.time()), stream, indent=2)
    previous.bridge.write(DEST / 'queue_skip_decisions.json', report['skip_details'])
    previous.bridge.write(DEST / 'bridge_status.json', dict(
        status='catalogue_running', pid=os.getpid(), continuation='20260919'))
    try:
        previous.bridge.catalog.run(DEST, True,
            inherited_done={'raw_mlp', 'explore_geometric_stats3'},
            on_complete=previous.bridge.analyze, continuation=report)
    except BaseException as error:
        previous.bridge.write(DEST / 'continuation_20260919_error.json',
            dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()
