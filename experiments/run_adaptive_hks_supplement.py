"""Prepare an immutable 6-method x 5-seed campaign; training is explicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = ('raw', 'geometry', 'fixed_adjacency', 'adaptive_adjacency', 'fixed_hks', 'adaptive_hks')


def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(destination):
    import multiagent
    destination.mkdir(parents=True, exist_ok=False)
    code = destination / 'code'
    sources = list(ROOT.glob('*.py'))
    for folder in ('utils', 'algorithms', 'tests'):
        sources += list((ROOT / folder).rglob('*.py'))
    sources += list((ROOT / 'experiments').glob('*.py'))
    for source in sources:
        target = code / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    environment = Path(multiagent.__file__).parent
    for source in environment.rglob('*.py'):
        target = code / 'multiagent' / source.relative_to(environment)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        'methods': list(NAMES), 'seeds': [41, 42, 43, 44, 45],
        'formal_steps': 2000000, 'smoke_steps': 2000,
        'final_seed_start': 92000000, 'curve_seed_start': 91000000,
        'note': 'Proposed fresh evaluation banks; do not tune on final results. No old runs reused.',
        'hashes': {str(p.relative_to(code)): digest(p) for p in sorted(code.rglob('*.py'))},
    }
    write(destination / 'manifest.json', manifest)
    commands = []
    script = code / 'experiments' / Path(__file__).name
    for phase in ('smoke', 'formal'):
        seeds = [41] if phase == 'smoke' else manifest['seeds']
        for seed in seeds:
            for method in NAMES:
                command = [sys.executable, str(script), 'run', '--campaign', str(destination),
                           '--phase', phase, '--method', method, '--seed', str(seed)]
                commands.append({'phase': phase, 'method': method, 'seed': seed, 'argv': command})
    write(destination / 'commands.json', commands)
    print(f'Prepared {len(commands)} commands; no training started: {destination}')


def verify(campaign, manifest):
    for name, expected in manifest['hashes'].items():
        if digest(campaign / 'code' / name) != expected:
            raise RuntimeError('Frozen source changed: ' + name)


def run(campaign, phase, method, seed):
    manifest = json.loads((campaign / 'manifest.json').read_text())
    if ROOT != campaign / 'code':
        raise RuntimeError('Run the script inside campaign/code, as recorded in commands.json')
    verify(campaign, manifest)
    if seed not in manifest['seeds'] or (phase == 'smoke' and seed != 41):
        raise ValueError('Seed is outside the frozen protocol')
    if phase == 'formal':
        smoke = campaign / 'smoke' / method / 'seed_41' / 'validation.json'
        if not smoke.exists() or not json.loads(smoke.read_text())['passed']:
            raise RuntimeError('Complete the matching smoke run first')
    directory = campaign / phase / method / f'seed_{seed}'
    if directory.exists():
        raise FileExistsError('Refusing overwrite: ' + str(directory))
    from utils.adaptive_hks_controls import register_policies, METHODS
    register_policies()
    from experiments import run_passive_gsp_topology_pilot as protocol
    from experiments.run_d4_actor_gpu_screen import training_args, evaluate_model
    from utils.cuda_protocol_audit import CudaAuditedMADDPG, require_cuda
    import run_vector_signal_gsp_experiment as metrics
    import torch
    require_cuda()
    args = training_args(phase)
    args.total_env_steps = manifest[phase + '_steps']
    args.checkpoint_steps = list(range(20000, args.total_env_steps + 1, 20000)) if phase == 'formal' else [2000]
    args.batch_size = 1024
    args.budget_episodes = args.total_env_steps // 25
    # Existing trainer adds seed*10000; cancel it to share episode banks across all runs.
    args.eval_seed_base = manifest['final_seed_start'] - seed * 10000
    args.curve_eval_seed_base = manifest['curve_seed_start'] - seed * 10000
    if phase == 'smoke':
        args.eval_seed_base = 93000000 - seed * 10000
        args.curve_eval_seed_base = 94000000 - seed * 10000
    actor, width = METHODS[method]
    spec = dict(env_id='simple_spread', actor_model=actor, actor_input_dim=width, short=method)
    protocol.METHODS = {method: spec}
    protocol.MADDPG, protocol.USE_CUDA = CudaAuditedMADDPG, True
    protocol.METRICS, protocol.evaluate_model = metrics.METRICS, evaluate_model
    CudaAuditedMADDPG.device_audit_path = directory / 'gpu_device_audit.json'
    torch.set_num_threads(args.n_training_threads)
    os.chdir(campaign)
    directory.mkdir(parents=True)
    write(directory / 'protocol.json', vars(args))
    try:
        protocol.train_one(method, seed, args, campaign / phase, 'ahks_' + phase)
        from experiments.run_gsp_long_campaign import validate_run
        validation_args = SimpleNamespace(**vars(args))
        validation_args.eval_seed_base += (seed - 1) * 10000
        validation = validate_run(directory, validation_args, spec)
        verify(campaign, manifest)
        write(directory / 'validation.json', validation)
    except BaseException as error:
        write(directory / 'failure.json', {'error': repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'queue'])
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--phase', choices=['smoke', 'formal'])
    parser.add_argument('--method', choices=NAMES)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.mode == 'prepare':
        prepare(campaign)
    elif args.mode == 'run':
        if args.phase is None or args.method is None or args.seed is None:
            parser.error('run requires --phase, --method, --seed')
        run(campaign, args.phase, args.method, args.seed)
    else:
        if args.phase is None:
            parser.error('queue requires --phase')
        for item in json.loads((campaign / 'commands.json').read_text()):
            if item['phase'] == args.phase:
                subprocess.run(item['argv'], check=True, cwd=campaign)


if __name__ == '__main__':
    main()
