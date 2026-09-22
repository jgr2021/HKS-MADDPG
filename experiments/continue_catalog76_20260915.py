"""Continue unstarted catalogue methods; preserve and skip interrupted runs."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import launch_catalog76_2m as launch
from resume_long_queue_20260908 import pid_alive

bridge = launch.bridge
DEST = launch.DEST
SKIPPED = {'catalog_compact_hks', 'catalog_relation_scales_hks'}
RECEIPT = DEST / 'continuation_20260915_started.json'


def preflight():
    state = bridge.catalog.read(DEST / 'catalog_status.json')
    controller = bridge.catalog.read(DEST / 'bridge_status.json')
    for pid in [controller['pid']] + [item['pid'] for item in state['active']]:
        if pid_alive(pid):
            raise RuntimeError('Recorded process still exists: ' + str(pid))
    if state['failures'] or (DEST / 'PAUSE_AFTER_CURRENT').exists():
        raise RuntimeError('Unresolved failure or candidate pause requires review')
    if {item['method'] for item in state['active']} != SKIPPED - set(state.get('skipped', [])):
        raise RuntimeError('Unexpected interrupted methods')
    sys.path.insert(0, str(DEST / 'code'))
    from experiments.run_gsp_exploration_campaign import verify_snapshot
    manifest = verify_snapshot(DEST)
    if manifest['formal']['total_env_steps'] != 2000000:
        raise ValueError('Wrong training budget')
    completed, smoke, skips = set(), set(), {}
    for name in manifest['methods']:
        for phase in ('smoke', 'formal'):
            marker = DEST / (phase + '_' + name + '.json')
            directory = DEST / phase / name / 'seed_1'
            if not marker.exists():
                if directory.exists() or (DEST / (phase+'_'+name+'.process.log')).exists():
                    raise RuntimeError('Untracked attempt: ' + name)
                continue
            item = bridge.catalog.read(marker)
            if name in SKIPPED and phase == 'formal':
                if item['status'] != 'running' or pid_alive(item['pid']):
                    raise RuntimeError('Unexpected interruption state: ' + name)
                log = (directory / 'training.log').read_text(encoding='utf-8')
                steps = int(re.findall(r'global_env_steps=(\d+)', log)[-1])
                skips[name] = dict(last_logged_env_steps=steps,
                    reason='Controller and workers absent; interruption cause unconfirmed',
                    disposition='preserve_without_evaluation_or_retry',
                    marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest())
                continue
            if item['status'] != 'completed' or not item.get('validation', {}).get('passed'):
                raise RuntimeError('Unresolved attempt: ' + name)
            (completed if phase == 'formal' else smoke).add(name)
            if phase == 'formal':
                summary = bridge.summarize(directory)['summary']
                analysis = bridge.catalog.read(DEST / ('analysis_'+name+'.json'))
                if summary['method'] != name or analysis['summary'] != summary:
                    raise RuntimeError('Analysis identity mismatch: ' + name)
                if analysis['prominent_candidate_review']:
                    raise RuntimeError('Candidate requires review: ' + name)
    if completed != set(state['completed']) or set(skips) != SKIPPED:
        raise RuntimeError('Inventory does not match recorded state')
    inherited = {'raw_mlp', 'explore_geometric_stats3'}
    for name in inherited:
        directory = DEST / 'matched_controls' / name / 'seed_1'
        summary = bridge.catalog.read(directory / 'summary.json')
        checkpoint = Path(summary['checkpoint'])
        launch.validate_cached_control(directory, name, checkpoint,
            hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    pending = [name for name in manifest['methods'] if name not in completed | SKIPPED]
    for name in pending:
        spec = manifest['methods'][name]
        controls = [spec.get('control')] + spec.get('required_additional_controls', [])
        if any(control in SKIPPED for control in controls):
            raise RuntimeError('Pending method depends on skipped control: ' + name)
    return dict(completed=sorted(completed), smoke_completed=sorted(smoke),
        skipped=sorted(SKIPPED), skip_details=skips, pending=pending,
        previous_status=state, previous_bridge=controller,
        frozen_files=len(manifest['source_hashes']), training_resume=False)


def main():
    mode = sys.argv[1:]
    if mode not in (['preflight'], ['run']):
        raise SystemExit('Use preflight or run')
    if RECEIPT.exists():
        raise RuntimeError('Continuation already started; do not duplicate')
    report = preflight()
    if mode == ['preflight']:
        print(json.dumps(report, indent=2))
        return
    with RECEIPT.open('x', encoding='utf-8') as stream:
        json.dump(dict(report, pid=os.getpid(), started=time.time()), stream, indent=2)
    bridge.write(DEST / 'queue_skip_decisions.json', report['skip_details'])
    bridge.write(DEST / 'bridge_status.json', dict(status='catalogue_running',
        pid=os.getpid(), continuation='20260915'))
    try:
        bridge.catalog.run(DEST, True, inherited_done={'raw_mlp','explore_geometric_stats3'},
            on_complete=bridge.analyze, continuation=report)
    except BaseException as error:
        bridge.write(DEST / 'continuation_20260915_error.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()
