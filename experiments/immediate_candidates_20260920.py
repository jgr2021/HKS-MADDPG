"""Adopt live workers without restarting them; dispatch the requested candidate now."""
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import time

import continue_catalog76_20260919 as recovery

bridge = recovery.previous.bridge
DEST = recovery.DEST
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel.GetExitCodeProcess.restype = wintypes.BOOL


class ExistingWorker:
    def __init__(self, pid):
        self.pid = pid
        self.handle = kernel.OpenProcess(0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def poll(self):
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if code.value == 259 else code.value


def run():
    from experiments.run_gsp_exploration_campaign import verify_snapshot
    verify_snapshot(DEST)
    state = bridge.catalog.read(DEST / 'catalog_status.json')
    if state['failures'] or (DEST / 'PAUSE_AFTER_CURRENT').exists():
        raise RuntimeError('Unresolved failure or pause')
    expected = {'catalog_action_current_delta_control', 'local_spectral_al_residual'}
    if {x['method'] for x in state['active']} != expected:
        raise RuntimeError('Queue changed; inspect before takeover')
    adopted = {x['method']: dict(process=ExistingWorker(x['pid']), phase=x['phase'])
               for x in state['active']}
    if any(x['process'].poll() is not None for x in adopted.values()):
        raise RuntimeError('Worker finished during takeover; inspect')
    smoke = []
    for path in DEST.glob('smoke_*.json'):
        value = bridge.catalog.read(path)
        if value.get('status') == 'completed' and value.get('validation', {}).get('passed'):
            smoke.append(path.stem[len('smoke_'):])
    receipt = DEST / 'immediate_candidates_20260920_started.json'
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(dict(previous=state, pid=os.getpid(), time=time.time(),
                       purpose='Immediate candidates; retain live workers; temporary third slot'), stream, indent=2)
    # Terminate only the verified scheduler, never its worker processes.
    command = "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=11200'; "
    command += "if ($p.CommandLine -notlike '*prioritize_paper_candidates_20260919.py*') { throw 'Controller identity mismatch' }; Stop-Process -Id 11200 -ErrorAction Stop"
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], check=True)
    bridge.write(DEST / 'bridge_status.json', dict(status='catalogue_running', pid=os.getpid(), continuation='immediate_candidates_20260920'))
    bridge.write(DEST / 'priority_handoff_20260919.json', dict(status='superseded_by_immediate_candidates', pid=os.getpid(), checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    report = dict(completed=state['completed'], skipped=state['skipped'], smoke_completed=smoke)
    bridge.catalog.run(DEST, True, inherited_done={'raw_mlp', 'explore_geometric_stats3'},
        on_complete=bridge.analyze, continuation=report, adopted_active=adopted,
        priority_methods=['catalog_action_current_delta', 'local_spectral_al_residual'],
        urgent_methods=['catalog_action_current_delta'])


if __name__ == '__main__':
    if sys.argv[1:] != ['run']:
        raise SystemExit('Use run')
    try:
        run()
    except BaseException as error:
        bridge.write(DEST / 'immediate_candidates_20260920_error.json', dict(error=repr(error)))
        raise
