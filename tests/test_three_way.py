"""Matched-control checks: no training or performance claims."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from _common import METHODS, load_runtime
import train
import evaluate
from summarize import aggregate
import torch
from utils import catalog76_policies as catalog
from utils import adaptive_hks_controls as controls
from algorithms.maddpg import MADDPG
from main import make_parallel_env


class ThreeWayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol, _ = load_runtime()

    def test_graph_and_statistics(self):
        torch.manual_seed(41)
        raw = torch.cat((torch.zeros(1, 18), torch.randn(12, 18) * .4))
        adjacency = controls.AdjacencyStatisticsPolicy(18, 5)
        hks = controls.FixedHKSPolicy(18, 5)
        with patch.object(controls, 'graph', wraps=catalog.graph) as ag, patch.object(
                catalog, 'graph', wraps=catalog.graph) as hg:
            a = adjacency.augment_observation(raw)
            h = hks.augment_observation(raw)
        aw = catalog.graph(*ag.call_args.args)
        hw = catalog.graph(*hg.call_args.args)
        torch.testing.assert_close(aw, hw, rtol=0, atol=0)
        focal = aw[:, 0, 3:6]
        expected = torch.stack((focal.mean(-1), focal.max(-1).values,
                                focal.std(-1, unbiased=False)), -1)
        torch.testing.assert_close(a[:, 18:], expected)
        for augmented in (a, h):
            self.assertEqual(tuple(augmented.shape), (13, 21))
            torch.testing.assert_close(augmented[:, :18], raw)
            self.assertTrue(torch.isfinite(augmented).all())
        permuted = raw.clone()
        permuted[:, 4:10] = raw[:, 4:10].reshape(-1, 3, 2)[:, [2, 0, 1]].reshape(-1, 6)
        torch.testing.assert_close(adjacency.augment_observation(permuted)[:, 18:], a[:, 18:])

    def test_capacity_and_matched_initialization(self):
        states, counts = {}, {}
        for method, (actor, width) in METHODS.items():
            env = make_parallel_env('simple_spread', 1, 41, True)
            try:
                torch.manual_seed(41)
                model = MADDPG.init_from_env(env, actor_model=actor)
                agent = model.agents[0]
                counts[method] = tuple(sum(p.numel() for p in module.parameters())
                                       for module in (agent.policy, agent.critic))
                states[method] = agent.policy.state_dict()
                self.assertEqual(agent.critic.fc1.in_features, 69)
                self.assertEqual(getattr(agent.policy, 'mlp', agent.policy).fc1.in_features, width)
            finally:
                env.close()
        self.assertEqual(counts['raw'], (5737, 8843))
        self.assertEqual(counts['hks'], (5935, 8843))
        self.assertEqual(counts['fixed_adjacency'], counts['hks'])
        self.assertEqual(counts['hks'][0] - counts['raw'][0], 198)
        self.assertEqual(len({count[1] for count in counts.values()}), 1)
        for key in states['hks']:
            torch.testing.assert_close(states['hks'][key], states['fixed_adjacency'][key], rtol=0, atol=0)

    def test_dispatch_all_methods_and_seeds(self):
        previous = self.protocol.MADDPG, self.protocol.USE_CUDA, self.protocol.METHODS
        try:
            with tempfile.TemporaryDirectory() as tmp:
                protocols = []
                for method, (actor, width) in METHODS.items():
                    for seed in range(41, 46):
                        with patch.object(sys, 'argv', ['train.py', '--method', method, '--seed', str(seed), '--output', tmp]), patch(
                                'utils.cuda_protocol_audit.require_cuda', return_value={}), patch.object(self.protocol, 'train_one') as dispatch:
                            train.main()
                        args = dispatch.call_args.args[2]
                        self.assertEqual(self.protocol.METHODS[method]['actor_model'], actor)
                        self.assertEqual(self.protocol.METHODS[method]['actor_input_dim'], width)
                        self.assertEqual(args.total_env_steps, 2000000)
                        self.assertEqual(args.checkpoint_steps, list(range(20000, 2000001, 20000)))
                        self.assertEqual((args.curve_eval_episodes, args.eval_episodes), (100, 500))
                        normalized = vars(args).copy()
                        normalized['eval_seed_base'] += seed * 10000
                        normalized['curve_eval_seed_base'] += seed * 10000
                        self.assertEqual(normalized['eval_seed_base'], 92000000)
                        self.assertEqual(normalized['curve_eval_seed_base'], 91000000)
                        protocols.append(normalized)
                self.assertTrue(all(p == protocols[0] for p in protocols))
        finally:
            self.protocol.MADDPG, self.protocol.USE_CUDA, self.protocol.METHODS = previous

    def test_adjacency_evaluation_cli_fresh_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = make_parallel_env('simple_spread', 1, 41, True)
            try:
                model = MADDPG.init_from_env(env, actor_model='confirm_fixed_adjacency')
                model.save(root / 'model.pt')
            finally:
                env.close()
            command = [sys.executable, str(Path(evaluate.__file__)), '--checkpoint',
                       str(root / 'model.pt'), '--episodes', '2', '--output', str(root / 'eval')]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / 'eval/summary.json').read_text())
            self.assertEqual(summary['eval_episodes'], 2)
            self.assertEqual(summary['seed_start'], 92000000)
            self.assertEqual(summary['selection'], 'provided_checkpoint')

    def test_three_method_summary_and_invalid_values(self):
        rows = [dict(method=method, seed=seed, selection='highest_mean_curve_return',
                     seed_start=92000000, eval_episodes=500, episode_length=25,
                     device='cpu', curve_seed_start=91000000, curve_episodes=100,
                     coverage_fraction=.5, full_coverage_rate=.2, collision_step_rate=.1,
                     **{'return': float(seed - 40)})
                for method in METHODS for seed in range(41, 46)]
        self.assertEqual(len(aggregate(rows)), 12)
        with self.assertRaises(ValueError):
            aggregate(rows + [rows[0]])
        rows[0]['return'] = float('nan')
        with self.assertRaises(ValueError):
            aggregate(rows)

    def test_evaluation_rejects_overlapping_bank_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(evaluate, 'select_checkpoint', return_value=(
                Path('unused.pt'), {'method': 'fixed_adjacency', 'seed': '41', 'return': '-1'}, (91000000, 100))), patch.object(
                evaluate, 'load_runtime') as runtime, patch.object(sys, 'argv', [
                'evaluate.py', '--run-dir', tmp, '--output', tmp + '/eval', '--seed-start', '91000099']):
            with self.assertRaises(SystemExit):
                evaluate.main()
            runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
