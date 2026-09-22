"""Aggregate independent training seeds, never treating episodes as replicates."""
import argparse
import json
from pathlib import Path
import numpy as np

# Two-sided Student-t 95% critical values for this frozen five-seed protocol.
T975 = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052}

METRICS = ('return', 'hungarian_assignment_distance', 'coverage_radius_auc', 'collision_step_rate')


def estimate(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else None
    if not 1 <= n <= 5:
        raise ValueError('This protocol requires one to five independent training seeds')
    half = float(T975[n-1] * std / np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, std=std,
                ci95=None if half is None else [mean-half, mean+half])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.campaign / 'manifest.json').read_text())
    rows, missing = {}, []
    for method in manifest['methods']:
        rows[method] = {}
        for seed in manifest['seeds']:
            directory = args.campaign / 'formal' / method / f'seed_{seed}'
            validation = directory / 'validation.json'
            if not validation.exists() or not json.loads(validation.read_text())['passed']:
                missing.append([method, seed])
                continue
            result = json.loads((directory / 'summary.json').read_text())
            if result['global_env_steps'] != manifest['formal_steps'] or result['eval_episodes'] != 500:
                raise ValueError('Budget/evaluation mismatch: ' + str(directory))
            rows[method][seed] = result
    output = {'complete': not missing, 'missing': missing, 'methods': {}, 'paired_differences': {}}
    for method, seeds in rows.items():
        if seeds:
            output['methods'][method] = {key: estimate([r[key] for r in seeds.values()]) for key in METRICS}
    for treatment, control in [('adaptive_hks', 'raw'), ('adaptive_hks', 'fixed_hks'),
                               ('adaptive_hks', 'adaptive_adjacency'),
                               ('adaptive_adjacency', 'fixed_adjacency'), ('adaptive_hks', 'geometry')]:
        common = sorted(rows[treatment].keys() & rows[control].keys())
        if common:
            output['paired_differences'][treatment + '_minus_' + control] = {
                'seeds': common, 'metrics': {
                    key: estimate([rows[treatment][s][key]-rows[control][s][key] for s in common])
                    for key in METRICS}}
    output['note'] = 'Student-t intervals over training seeds; differences are treatment minus control. Partial results are not confirmatory.'
    path = args.campaign / 'aggregate.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(f'{path}: complete={not missing}; missing={len(missing)}')


if __name__ == '__main__':
    main()
