"""Explicit continuation after user-approved skips; frozen training is unchanged."""
import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'experiments/gsp_long_20260906b'
SKIPPED = {'contact_d4_energy', 'explore_hks_sigma075', 'explore_hks_sigma125'}
ANALYSIS = CAMPAIGN / 'analysis_interrupted_sigma_20260908/analysis.json'
RUN_TAG = '20260908'
EXPECTED_PENDING = 11
EXPECTED_DONE = 9
EXTRA_INTERRUPTED = []
spec = importlib.util.spec_from_file_location('previous_queue', CAMPAIGN / 'operations/run_long_parallel_continue_20260907.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
read, write = previous.read, previous.write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pid_alive(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if handle:
        kernel.CloseHandle(handle)
        return True
    error = ctypes.get_last_error()
    if error == 87:
        return False
    raise OSError(error, 'Unable to verify recorded PID')


def verify_analysis():
    data = read(ANALYSIS)
    assert data['training_steps_added'] == 0
    for path, sha in data['checkpoint_hashes'].items():
        assert digest(Path(path)) == sha
    for run in data['runs']:
        assert read(CAMPAIGN / ('formal_' + run['method'] + '.json')) == run['pre_reconciliation_marker']
        for name, summary in run['results'].items():
            path = ANALYSIS.parent / (name + '_step' + str(run['checkpoint_env_steps']) + '_episodes.csv')
            with path.open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 500
            assert [int(r['test_seed']) for r in rows] == list(range(1000000, 1000500))
            for key in ('return', 'hungarian_assignment_distance', 'coverage_radius_auc', 'collision_step_rate', 'final_coverage', 'final3'):
                assert abs(sum(float(r[key]) for r in rows) / 500 - summary[key]) < 1e-10
    data['runs'].extend(EXTRA_INTERRUPTED)
    return data


def inventory(c, manifest):
    done, smoke = set(), set()
    for name, config in manifest['methods'].items():
        if name in SKIPPED:
            continue
        for phase in ('smoke', 'formal'):
            marker = CAMPAIGN / (phase + '_' + name + '.json')
            directory = CAMPAIGN / phase / name / 'seed_1'
            if not marker.exists():
                assert not directory.exists(), 'Untracked attempt: ' + name
                continue
            state = read(marker)
            assert state['status'] == 'completed' and state['validation']['passed'], 'Unresolved attempt: ' + name
            assert c.validate_run(directory, c.training_args(phase), config) == state['validation']
            (done if phase == 'formal' else smoke).add(name)
    assert done <= smoke
    return done, smoke


def preflight():
    old_status = read(CAMPAIGN / 'parallel_status.json')
    for pid in [old_status['pid']] + [a['pid'] for a in old_status.get('active', [])]:
        assert not pid_alive(pid), 'Recorded process still exists: ' + str(pid)
    c = previous.load(CAMPAIGN)
    import torch
    torch.set_num_threads(1)
    manifest = c.verify_snapshot(CAMPAIGN)
    analysis = verify_analysis()
    assert read(CAMPAIGN / 'formal_contact_d4_energy.json')['status'] == 'failed'
    done, smoke = inventory(c, manifest)
    pending = [n for n in manifest['methods'] if n not in done | SKIPPED]
    assert len(done) == EXPECTED_DONE and len(pending) == EXPECTED_PENDING
    report = dict(passed=True, checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  completed=sorted(done), skipped=sorted(SKIPPED), pending=pending,
                  previous_status=old_status, analysis_sha256=digest(ANALYSIS),
                  preserved_markers={n: digest(CAMPAIGN / ('formal_' + n + '.json')) for n in SKIPPED},
                  frozen_files=len(manifest['source_hashes']), training_resume=False)
    return c, manifest, analysis, done, smoke, report


def refresh(c, analysis):
    previous.refresh(CAMPAIGN, c)
    rows = read(CAMPAIGN / 'long_master.json')
    by_name = {r['method']: r for r in analysis['runs']}
    for row in rows:
        if row['method'] in by_name:
            run = by_name[row['method']]
            row.update(status='interrupted', queue_disposition='skipped_by_user_no_retry',
                       observed_failure=run.get('pre_reconciliation_marker', {}).get('error') or 'Controller and workers absent; cause unproven',
                       last_logged_env_steps=run['last_logged_env_steps'],
                       latest_checkpoint_env_steps=run['checkpoint_env_steps'],
                       interruption_analysis=run.get('analysis_path', str(ANALYSIS)) if run.get('results') else None)
    write(CAMPAIGN / 'long_master.json', rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = CAMPAIGN / 'long_master.csv.tmp'
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(CAMPAIGN / 'long_master.csv')


def supervise():
    receipt = CAMPAIGN / ('parallel_continuation_started_' + RUN_TAG + '.json')
    assert not receipt.exists(), 'Continuation already launched'
    c, manifest, analysis, done, smoke, report = preflight()
    write(receipt, dict(report, pid=os.getpid()), True)
    decisions = read(CAMPAIGN / 'queue_skip_decisions.json')
    for run in analysis['runs']:
        decisions[run['method']] = dict(reason='User explicitly requests skip without training retry; preserve authorized evaluations',
                                      analysis=run.get('analysis_path', str(ANALYSIS)) if run.get('results') else None, last_logged_env_steps=run['last_logged_env_steps'],
                                      checkpoint_env_steps=run['checkpoint_env_steps'])
    write(CAMPAIGN / 'queue_skip_decisions.json', decisions)
    active, failures = {}, []
    status = dict(pid=os.getpid(), max_workers=2, skipped=sorted(SKIPPED), failures=failures,
                  acknowledged_failures=report['previous_status'].get('acknowledged_failures', []),
                  reconciled_interrupted_methods=[r['method'] for r in analysis['runs']])
    while True:
        for name, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            del active[name]
            state = read(CAMPAIGN / (item['phase'] + '_' + name + '.json'))
            if code or state.get('status') != 'completed' or not state.get('validation', {}).get('passed'):
                failures.append(dict(method=name, phase=item['phase'], exit_code=code, error=state.get('error')))
            else:
                (done if item['phase'] == 'formal' else smoke).add(name)
        paused = bool(failures) or (CAMPAIGN / 'PAUSE_AFTER_CURRENT').exists()
        for name, config in manifest['methods'].items():
            if paused or len(active) >= 2:
                break
            if name in done | SKIPPED or name in active:
                continue
            if any(n not in done for n in previous.required_controls(config)):
                continue
            mem = previous.memory()
            if min(mem['avail_phys'], mem['avail_page']) < 3 * 1024**3:
                break
            c.require_space(CAMPAIGN)
            c.verify_snapshot(CAMPAIGN)
            phase = 'formal' if name in smoke else 'smoke'
            assert not (CAMPAIGN / phase / name / 'seed_1').exists()
            command = [sys.executable, '-u', '-B', str(CAMPAIGN / 'code/experiments/run_gsp_long_campaign.py'),
                       'worker', '--campaign', str(CAMPAIGN), '--phase', phase, '--method', name]
            with (CAMPAIGN / (phase + '_' + name + '.process.log')).open('x') as stream:
                process = subprocess.Popen(command, cwd=CAMPAIGN, stdout=stream, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
            active[name] = dict(process=process, phase=phase, command=subprocess.list2cmdline(command))
        complete = len(done | SKIPPED) == len(manifest['methods'])
        status.update(status='completed_with_skips' if complete else 'draining_after_failure' if failures and active else
                      'paused_technical_failure' if failures else 'paused_by_file' if paused else 'running',
                      active=[dict(method=n, phase=i['phase'], pid=i['process'].pid, command=i['command']) for n, i in active.items()],
                      completed=[n for n in manifest['methods'] if n in done], memory=previous.memory(),
                      checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        write(CAMPAIGN / 'parallel_status.json', status)
        refresh(c, analysis)
        if complete or paused and not active:
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'supervise', 'launch'])
    args = parser.parse_args()
    if args.mode == 'preflight':
        c, _, analysis, _, _, report = preflight()
        write(CAMPAIGN / ('continuation_preflight_' + RUN_TAG + '.json'), report, True)
        refresh(c, analysis)
        print(json.dumps(report), flush=True)
    elif args.mode == 'supervise':
        try:
            supervise()
        except BaseException:
            receipt = CAMPAIGN / ('parallel_continuation_started_' + RUN_TAG + '.json')
            if receipt.exists() and read(receipt).get('pid') == os.getpid():
                status = read(CAMPAIGN / 'parallel_status.json')
                status.update(status='controller_failure', error=traceback.format_exc(),
                              checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                write(CAMPAIGN / 'parallel_status.json', status)
            raise
    else:
        with (CAMPAIGN / ('operations/continuation_' + RUN_TAG + '.controller.log')).open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-B', str(Path(sys.argv[0]).resolve()), 'supervise'],
                                     cwd=CAMPAIGN, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_BREAKAWAY_FROM_JOB)
        print(json.dumps({'controller_pid': child.pid}), flush=True)


if __name__ == '__main__':
    main()
