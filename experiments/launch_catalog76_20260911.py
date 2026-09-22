"""Authorized catalogue continuation after the active original queue finishes."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import catalog76 as catalog

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/gsp_long_20260906b'
TEMPLATE = ROOT / 'experiments/c76ready_20260908'
DEST = ROOT / 'experiments/catalog76_run_20260911'
METRICS = ('return', 'hungarian_assignment_distance', 'coverage_radius_auc', 'collision_step_rate')
BUDGET = 2500000
REFERENCE_ROOT = SOURCE / 'formal'


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def prepare():
    from experiments.run_gsp_exploration_campaign import verify_snapshot
    manifest = verify_snapshot(TEMPLATE)
    old = catalog.read(SOURCE / 'manifest.json')
    names = set(old['methods'])
    methods = {n: s for n, s in manifest['methods'].items() if n not in names}
    assert methods and not set(methods) & names
    assert not DEST.exists()
    DEST.mkdir()
    shutil.copytree(TEMPLATE / 'code', DEST / 'code')
    manifest.update(methods=methods, prepared_from=str(SOURCE),
                    study='catalog76_authorized_remaining_20260911',
                    reuse_old_results=True, inherited_completed_methods=[],
                    excluded_existing_methods=sorted(set(manifest['methods']) & names))
    write(DEST / 'manifest.json', manifest)
    write(DEST / 'authorization.json', dict(new_methods=list(methods),
          blocked=catalog.BLOCKED, excluded_existing=manifest['excluded_existing_methods'],
          protocol=catalog.DEFAULTS, interrupted_policy='preserve_without_evaluation_or_retry',
          catalog_entries=catalog.entries()))
    verify_snapshot(DEST)
    print(json.dumps(dict(prepared=str(DEST), new_configurations=len(methods))))


def summarize(directory):
    summary = catalog.read(directory / 'summary.json')
    validation = catalog.read(directory / 'validation.json')
    assert validation['passed'] and summary['global_env_steps'] == BUDGET
    assert summary['eval_episodes'] == 500
    with (directory / 'per_evaluation_episode_metrics.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 500
    assert [int(r['test_seed']) for r in rows] == list(range(1000000, 1000500))
    for k in METRICS + ('final_coverage', 'final3'):
        values = [float(r[k]) for r in rows]
        assert all(math.isfinite(v) for v in values)
        assert abs(sum(values) / len(values) - summary[k]) < 1e-10
    with (directory / 'learning_curve_eval.csv').open(encoding='utf-8-sig') as stream:
        curve = list(csv.DictReader(stream))
    assert len(curve) == BUDGET // 20000
    return dict(summary=summary, last_six_checkpoint_ranges={
        k: [min(float(r[k]) for r in curve[-6:]), max(float(r[k]) for r in curve[-6:])]
        for k in METRICS}, artifact_path=str(directory),
        limitation='Single training seed; intermediate checkpoints use100 episodes, final uses500. Causes and convergence unproven.')


def analyze(name, directory=None):
    directory = directory or DEST / 'formal' / name / 'seed_1'
    result = summarize(directory)
    manifest = catalog.read((SOURCE if directory.is_relative_to(SOURCE) else DEST) / 'manifest.json')
    spec = manifest['methods'][name]
    comparisons = {}
    for control in dict.fromkeys(['raw_mlp', spec.get('control')] + spec.get('required_additional_controls', [])):
        if not control or control == name:
            continue
        path = DEST / 'formal' / control / 'seed_1' / 'summary.json'
        if not path.exists():
            path = REFERENCE_ROOT / control / 'seed_1' / 'summary.json'
        reference = catalog.read(path)
        assert reference['global_env_steps'] == BUDGET and reference['eval_episodes'] == 500
        delta = {k: result['summary'][k] - reference[k] for k in METRICS}
        comparisons[control] = dict(delta=delta, all_four_improved=
            delta['return'] > 0 and delta['coverage_radius_auc'] > 0 and
            delta['hungarian_assignment_distance'] < 0 and delta['collision_step_rate'] < 0)
    result['comparisons'] = comparisons
    result['prominent_candidate_review'] = bool(comparisons) and all(x['all_four_improved'] for x in comparisons.values())
    write(DEST / ('analysis_' + name + '.json'), result)
    if result['prominent_candidate_review']:
        (DEST / 'PAUSE_AFTER_CURRENT').touch(exist_ok=True)
    return result


def supervise():
    with (DEST / 'bridge_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), started=time.time()), stream)
    while True:
        state = catalog.read(SOURCE / 'parallel_status.json')
        write(DEST / 'bridge_status.json', dict(status='waiting_for_original_queue',
              pid=os.getpid(), source_status=state['status'], checked_at=time.time()))
        if state['status'] == 'completed_with_skips' and not state['active']:
            break
        if state['status'] != 'running':
            raise RuntimeError('Original queue requires attention: ' + state['status'])
        from resume_long_queue_20260908 import pid_alive
        if not pid_alive(state['pid']):
            raise RuntimeError('Original controller absent; no automatic restart')
        time.sleep(30)
    inherited = set(state['completed'])
    analyze('explore_regularized_resistance3', SOURCE / 'formal/explore_regularized_resistance3/seed_1')
    manifest = catalog.read(DEST / 'manifest.json')
    for spec in manifest['methods'].values():
        for control in [spec.get('control')] + spec.get('required_additional_controls', []):
            assert not control or control in inherited or control in manifest['methods']
    write(DEST / 'inherited_results.json', {n: str(SOURCE / 'formal' / n / 'seed_1') for n in inherited})
    write(DEST / 'bridge_status.json', dict(status='catalogue_running', pid=os.getpid()))
    catalog.run(DEST, True, inherited_done=inherited, on_complete=analyze)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['scheduled']:
        with (DEST / 'bridge.log').open('x', encoding='utf-8', buffering=1) as log:
            sys.stdout = sys.stderr = log
            try:
                supervise()
            except BaseException as error:
                write(DEST / 'bridge_error.json', dict(error=repr(error), pid=os.getpid()))
                raise
    else:
        raise SystemExit('Use prepare or scheduled')
