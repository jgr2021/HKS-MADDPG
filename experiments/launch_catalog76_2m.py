"""User-authorized 2M-step catalogue; reuse evaluated 2M baseline checkpoints."""
import csv
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import time

import launch_catalog76_20260911 as bridge

DEST = bridge.ROOT / 'experiments/catalog76_2m_20260911'
bridge.DEST = DEST
bridge.BUDGET = 2000000
bridge.REFERENCE_ROOT = DEST / 'matched_controls'


def prepare():
    from experiments.run_gsp_exploration_campaign import verify_snapshot, code_hashes
    old = bridge.ROOT / 'experiments/catalog76_run_20260911'
    manifest = verify_snapshot(old)
    assert not DEST.exists()
    DEST.mkdir()
    shutil.copytree(old / 'code', DEST / 'code')
    # Mechanical budget substitution in a new snapshot; historical snapshots stay intact.
    path = DEST / 'code/experiments/run_gsp_long_campaign.py'
    source = path.read_text(encoding='utf-8')
    source = source.replace('2500001', '2000001').replace('2500000', '2000000').replace('100000-episode', '80000-episode')
    source = source.replace('100,000-episode', '80,000-episode').replace('"training_episodes": 100000', '"training_episodes": 80000').replace('"budget_episodes": 100000', '"budget_episodes": 80000')
    path.write_text(source, encoding='utf-8')
    formal = manifest['formal']
    formal.update(total_env_steps=2000000, budget_episodes=80000,
                  checkpoint_steps=list(range(20000, 2000001, 20000)))
    manifest.update(study='catalog76_2m_authorized', source_hashes=code_hashes(DEST / 'code'),
                    comparison_budget=2000000, stopped_previous_at_user_request=True)
    bridge.write(DEST / 'manifest.json', manifest)
    verify_snapshot(DEST)
    status_path = bridge.SOURCE / 'parallel_status.json'
    status = bridge.catalog.read(status_path)
    bridge.write(DEST / 'previous_source_status.json', status)
    log = bridge.SOURCE / 'formal/explore_regularized_resistance3/seed_1/training.log'
    actual = int(re.findall(r'global_env_steps=(\d+)', log.read_text())[-1])
    bridge.write(DEST / 'budget_change.json', dict(requested_stop=1500000, actual_last_logged=actual,
                 new_budget=2000000, old_budget=2500000, originals_preserved=True))
    status.update(status='stopped_by_user_budget_change', active=[], checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    bridge.write(status_path, status)
    print(dict(prepared=str(DEST), methods=len(manifest['methods']), actual_stop=actual))


def validate_cached_control(directory, name, checkpoint, sha):
    summary = bridge.catalog.read(directory / 'summary.json')
    assert summary['method'] == name and summary['global_env_steps'] == 2000000
    assert summary['eval_episodes'] == 500 and summary['training_steps_added'] == 0
    assert summary['checkpoint_sha256'] == sha and Path(summary['checkpoint']) == checkpoint
    with (directory / 'per_evaluation_episode_metrics.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 500 and [int(r['test_seed']) for r in rows] == list(range(1000000, 1000500))
    for key in bridge.METRICS + ('final_coverage', 'final3'):
        assert abs(sum(float(r[key]) for r in rows) / 500 - summary[key]) < 1e-10


def run():
    from resume_long_queue_20260908 import pid_alive
    assert not any(pid_alive(p) for p in (14680, 28716, 21148)), 'Old process remains'
    sys.path.insert(0, str(DEST / 'code'))
    bridge.catalog.register()
    from experiments.run_gsp_exploration_campaign import verify_snapshot
    verify_snapshot(DEST)
    from experiments.run_d4_actor_gpu_screen import evaluate_model
    import torch
    torch.set_num_threads(1)
    bridge.write(DEST / 'bridge_status.json', dict(status='evaluating_2m_controls', pid=os.getpid()))
    for name in ('raw_mlp', 'explore_geometric_stats3'):
        checkpoint = bridge.SOURCE / 'formal' / name / 'seed_1/checkpoints/model_step2000000.pt'
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        directory = bridge.REFERENCE_ROOT / name / 'seed_1'
        if directory.exists():
            validate_cached_control(directory, name, checkpoint, sha)
            print('Verified existing control without reevaluation: ' + name, flush=True)
            continue
        rows = evaluate_model('simple_spread', checkpoint, 500, 25, 1000000)
        assert len(rows) == 500 and [r['test_seed'] for r in rows] == list(range(1000000,1000500))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
        directory.mkdir(parents=True)
        with (directory / 'per_evaluation_episode_metrics.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {k: sum(float(r[k]) for r in rows) / 500 for k in bridge.METRICS + ('final_coverage','final3')}
        summary.update(global_env_steps=2000000, eval_episodes=500, method=name,
                       checkpoint=str(checkpoint), checkpoint_sha256=sha, training_steps_added=0)
        bridge.write(directory / 'summary.json', summary)
    bridge.write(DEST / 'bridge_status.json', dict(status='catalogue_running', pid=os.getpid()))
    bridge.catalog.run(DEST, True, inherited_done={'raw_mlp','explore_geometric_stats3'}, on_complete=bridge.analyze)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] in (['scheduled'], ['recover-registration']):
        recovery = sys.argv[1] == 'recover-registration'
        suffix = '_registration_recovery' if recovery else ''
        if recovery:
            assert not (DEST / 'catalog_started.json').exists(), 'Training already started'
            assert 'Unknown actor_model' in bridge.catalog.read(DEST / 'bridge_error.json')['error']
        with (DEST / ('bridge' + suffix + '.log')).open('x', encoding='utf-8', buffering=1) as stream:
            sys.stdout = sys.stderr = stream
            try:
                run()
            except BaseException as error:
                bridge.write(DEST / ('bridge' + suffix + '_error.json'), dict(error=repr(error)))
                raise
    else:
        raise SystemExit('Use prepare or scheduled')
