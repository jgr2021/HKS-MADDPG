"""Drain current workers, then prioritize two user-selected method/control pairs."""
import json
import os
import subprocess
import sys
import time

import continue_catalog76_20260919 as recovery

previous = recovery.previous
DEST = recovery.DEST
PRIORITY = ['catalog_action_current_delta_control', 'local_geometry_residual',
            'catalog_action_current_delta', 'local_spectral_al_residual']
PAUSE = DEST / 'PAUSE_AFTER_CURRENT'
TOKEN = 'User-requested priority handoff 20260919; owned by prioritize_paper_candidates_20260919.py\n'
STATUS = DEST / 'priority_handoff_20260919.json'
RECEIPT = DEST / 'priority_handoff_20260919_started.json'


def write_state(status, **extra):
    previous.bridge.write(STATUS, dict(status=status, pid=os.getpid(),
        priority_methods=PRIORITY, checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra))


def run():
    state = previous.bridge.catalog.read(DEST / 'catalog_status.json')
    old = previous.bridge.catalog.read(DEST / 'bridge_status.json')
    live = recovery.process_inventory()
    if old['pid'] not in live or any(x['pid'] not in live for x in state['active']):
        raise RuntimeError('Current controller or worker absent; inspect before handoff')
    if state['failures'] or PAUSE.exists():
        raise RuntimeError('An existing failure or pause requires review')
    manifest = previous.bridge.catalog.read(DEST / 'manifest.json')
    assert set(PRIORITY) <= set(manifest['methods'])
    for name in PRIORITY:
        if name in state.get('skipped', []):
            raise RuntimeError('Priority method previously interrupted: '+name)
    with RECEIPT.open('x', encoding='utf-8') as stream:
        json.dump(dict(controller_pid=old['pid'], active=state['active'],
            priority_methods=PRIORITY, pid=os.getpid()), stream, indent=2)
    with PAUSE.open('x', encoding='utf-8') as stream:
        stream.write(TOKEN)
    write_state('waiting_for_current_workers', controller_pid=old['pid'])
    while True:
        state = previous.bridge.catalog.read(DEST / 'catalog_status.json')
        if state['failures']:
            raise RuntimeError('Training failure during handoff; pause retained')
        if any(json.loads(p.read_text(encoding='utf-8')).get('prominent_candidate_review')
               for p in DEST.glob('analysis_catalog_*.json')):
            raise RuntimeError('Prominent candidate requires review; pause retained')
        live = recovery.process_inventory()
        if not state['active'] and old['pid'] not in live:
            break
        if old['pid'] not in live or any(x['pid'] not in live for x in state['active']):
            # Allow one controller cycle to reconcile a just-finished worker.
            time.sleep(20)
            state = previous.bridge.catalog.read(DEST / 'catalog_status.json')
            live = recovery.process_inventory()
            if state['active'] and (old['pid'] not in live or any(x['pid'] not in live for x in state['active'])):
                raise RuntimeError('Unexpected process interruption; pause retained')
        time.sleep(10)
    if state['status'] != 'paused_by_user' or PAUSE.read_text(encoding='utf-8') != TOKEN:
        raise RuntimeError('Unexpected pause ownership or queue status')
    watcher = previous.bridge.catalog.read(DEST / 'analysis_table_watcher_status.json')
    deadline = time.monotonic() + 180
    while watcher['pid'] in recovery.process_inventory():
        if time.monotonic() > deadline:
            raise RuntimeError('Previous workbook watcher has not drained')
        time.sleep(5)
    PAUSE.unlink()
    previous.SKIPPED = set(state.get('skipped', []))
    previous.pid_alive = lambda pid: int(pid) in recovery.process_inventory()
    report = previous.preflight()
    previous.bridge.write(DEST / 'bridge_status.json', dict(status='catalogue_running',
        pid=os.getpid(), continuation='priority_20260919'))
    write_state('priority_queue_running', completed_at_handoff=report['completed'])
    # The previous watcher exits when the drained queue becomes terminal.
    node = r'C:\Users\10431\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
    watcher_script = DEST.parents[1] / 'experiments/watch_catalog_analysis_table.py'
    subprocess.Popen([sys.executable, str(watcher_script), '--node', node],
        cwd=DEST.parents[1], creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    previous.bridge.catalog.run(DEST, True,
        inherited_done={'raw_mlp', 'explore_geometric_stats3'},
        on_complete=previous.bridge.analyze, continuation=report,
        priority_methods=PRIORITY)
    write_state('controller_returned')


if __name__ == '__main__':
    if sys.argv[1:] != ['run']:
        raise SystemExit('Use run')
    try:
        run()
    except BaseException as error:
        write_state('handoff_failed', error=repr(error))
        raise
