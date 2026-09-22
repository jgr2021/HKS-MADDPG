"""Evaluate preserved final checkpoint without resuming training."""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'experiments/gsp_long_20260906b'
OUTPUT = CAMPAIGN / 'analysis_interrupted_energy_20260907'
sys.path.insert(0, str(CAMPAIGN / 'code'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from experiments import run_gsp_long_campaign as campaign
    from experiments import run_d4_actor_gpu_screen as evaluator
    import run_vector_signal_gsp_experiment as metrics
    import torch
    torch.set_num_threads(1)
    assert evaluator.EVAL_DEVICE == 'cpu'
    campaign.verify_snapshot(CAMPAIGN)
    for module in campaign.REGISTRATIONS:
        importlib.import_module(module).register_policies()
    marker = CAMPAIGN / 'formal_contact_d4_energy.json'
    marker_hash = digest(marker)
    failure = read(marker)
    assert failure['status'] == 'failed'
    log = (CAMPAIGN / 'formal/contact_d4_energy/seed_1/training.log').read_text(encoding='utf-8')
    last_logged = int(re.findall(r'global_env_steps=(\d+)/2500000', log)[-1])
    candidates = list((Path(failure['model_run_directory']) / 'incremental').glob('model_step*.pt'))
    checkpoint = max(candidates, key=lambda p: int(p.stem.replace('model_step', '')))
    step = int(checkpoint.stem.replace('model_step', ''))
    assert step <= last_logged < 2500000
    OUTPUT.mkdir(exist_ok=False)
    results = []
    for method in ('contact_d4_energy', 'raw_mlp', 'contact_d4_geometry'):
        source = checkpoint if method == 'contact_d4_energy' else CAMPAIGN / 'formal' / method / 'seed_1/checkpoints' / checkpoint.name
        before = digest(source)
        start = time.perf_counter()
        rows = evaluator.evaluate_model('simple_spread', source, 500, 25, 1000000)
        assert [r['test_seed'] for r in rows] == list(range(1000000, 1000500))
        assert digest(source) == before and digest(marker) == marker_hash
        with (OUTPUT / (method + '_episodes.csv')).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {key: sum(float(r[key]) for r in rows) / len(rows) for key in metrics.METRICS}
        summary.update(method=method, checkpoint_env_steps=step, eval_episodes=500,
                       checkpoint=str(source), checkpoint_sha256=before,
                       evaluation_seconds=time.perf_counter() - start,
                       success_count=sum(r['final3'] for r in rows))
        results.append(summary)
        print(json.dumps(summary), flush=True)
    campaign.verify_snapshot(CAMPAIGN)
    report = dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  status='interrupted_checkpoint_evaluated_not_completed_training',
                  last_logged_env_steps=last_logged, checkpoint_env_steps=step,
                  planned_env_steps=2500000, training_steps_added=0,
                  evaluation_device='cpu', evaluation_threads=1,
                  evaluation_seed_first=1000000, evaluation_seed_last=1000499,
                  original_failure_marker_sha256=marker_hash,
                  checkpoint_selection='Last saved checkpoint, not selected by performance',
                  caveat='One training seed; post-hoc interrupted-budget analysis, not a 2.5M endpoint or proof of robust superiority',
                  results=results)
    with (OUTPUT / 'analysis.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print('Analysis saved: ' + str(OUTPUT / 'analysis.json'), flush=True)


if __name__ == '__main__':
    main()
