"""Descriptor scale errors on real observations and synchronized actor timings."""
import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from utils.adaptive_hks_controls import register_policies, METHODS
from utils.agents import POLICY_TYPES
from utils.ahks_environments import SpreadEnv
from utils.catalog76_policies import features
from utils.exploration_topology_policies import local_positions, topology_constants


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--observations', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=95000000)
    args = parser.parse_args()
    if min(args.observations, args.repeats) <= 0:
        parser.error('Counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    env = SpreadEnv(args.seed)
    raw = []
    try:
        obs = env.reset()
        while len(raw) < args.observations:
            raw.extend(obs.copy())
            obs, _, done, _ = env.step(rng.randint(0, 5, size=3))
            if done:
                obs = env.reset()
    finally:
        env.close()
    raw = torch.tensor(np.array(raw[:args.observations]), dtype=torch.float32)
    np.save(args.output/'raw_observations.npy', raw.numpy())
    mask, _ = topology_constants('6al')
    errors = []
    for factor in (.5, .75, 1., 1.5, 2.):
        scaled = raw.clone()
        scaled[:, 2:14] *= factor  # self absolute position and all relative positions
        points = local_positions(scaled, '6al')
        distances = torch.linalg.vector_norm(points[:, :, None]-points[:, None, :], dim=-1)
        floor_rate = float((distances[:, mask.bool()].median(-1).values <= .05).float().mean())
        for name, config in [('fixed_hks', {}), ('adaptive_hks', {'kernel': 'adaptive'})]:
            error = torch.linalg.vector_norm(features(raw, config)-features(scaled, config), dim=-1)
            errors.append(dict(method=name, scale=factor, mean_l2_error=float(error.mean()),
                               max_l2_error=float(error.max()), bandwidth_floor_rate=floor_rate,
                               observations=len(raw)))
    save_csv(args.output/'descriptor_scale.csv', errors)
    register_policies()
    timings = []
    for name in ('raw', 'fixed_hks', 'adaptive_hks'):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            model = POLICY_TYPES[METHODS[name][0]](18, 5, hidden_dim=64).eval().to(device)
            for size in (1, 4, 1024):
                host = raw.repeat((size+len(raw)-1)//len(raw), 1)[:size]
                resident = host.to(device)
                for transfer in (False, True):
                    def invoke():
                        return model(host.to(device)).cpu() if transfer else model(resident)
                    def synchronize():
                        if device == 'cuda':
                            torch.cuda.synchronize()
                    samples = []
                    with torch.no_grad():
                        for _ in range(10):
                            invoke()
                        synchronize()
                        for _ in range(args.repeats):
                            synchronize()
                            start = time.perf_counter()
                            invoke()
                            synchronize()
                            samples.append((time.perf_counter()-start)*1000)
                    timings.append(dict(method=name, device=device, batch=size, include_host_transfers=transfer,
                        mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples)),
                        std_ms=float(np.std(samples)), repeats=args.repeats,
                        actor_parameters=sum(p.numel() for p in model.parameters())))
    save_csv(args.output/'actor_runtime.csv', timings)
    config = dict(seed=args.seed, observations=len(raw), repeats=args.repeats, threads=1, warmup=10,
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  note='Untrained actors; descriptor property/inference overhead only, not policy-scale invariance or training throughput.')
    (args.output/'protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), scale_rows=len(errors), runtime_rows=len(timings))))


if __name__ == '__main__':
    main()
