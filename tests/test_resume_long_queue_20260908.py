import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments import resume_long_queue_20260908 as q


class RecoveryTests(unittest.TestCase):
    def test_inventory_rejects_changed_completed_and_untracked_attempts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'CAMPAIGN', Path(tmp)):
            root = Path(tmp)
            manifest = {'methods': {'raw': {}}}
            c = SimpleNamespace(training_args=lambda phase: phase, validate_run=Mock(return_value={'passed': False}))
            q.write(root / 'smoke_raw.json', {'status': 'completed', 'validation': {'passed': True}})
            with self.assertRaises(AssertionError):
                q.inventory(c, manifest)

    def test_smoke_before_formal_two_slots_and_failure_drains(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(q, 'CAMPAIGN', Path(tmp)):
            root = Path(tmp)
            q.write(root / 'queue_skip_decisions.json', {'energy': {'reason': 'keep'}})
            methods = {'raw_mlp': {'control': None}, **{n: {'control': 'raw_mlp'} for n in ('a', 'b', 'c')}}
            campaign = Mock()
            report = {'previous_status': {}}
            launched, live, peak = [], set(), [0]

            class Process:
                pid = 123
                def __init__(self, command, **kwargs):
                    self.name = command[command.index('--method') + 1]
                    self.phase = command[command.index('--phase') + 1]
                    launched.append((self.name, self.phase))
                    live.add(self.name)
                    peak[0] = max(peak[0], len(live))
                    self.failed = self.name == 'a' and self.phase == 'formal'
                    q.write(root / (self.phase + '_' + self.name + '.json'),
                            {'status': 'failed' if self.failed else 'completed', 'validation': {'passed': not self.failed}})
                def poll(self):
                    live.discard(self.name)
                    return 1 if self.failed else 0

            with patch.object(q, 'preflight', return_value=(campaign, {'methods': methods}, {'runs': []}, {'raw_mlp'}, {'raw_mlp'}, report)), \
                 patch.object(q, 'refresh'), patch.object(q, 'SKIPPED', set()), \
                 patch.object(q.previous, 'memory', return_value={'avail_phys': 10*1024**3, 'avail_page': 10*1024**3}), \
                 patch.object(q.subprocess, 'Popen', Process), patch.object(q.time, 'sleep'):
                q.supervise()
                self.assertLessEqual(peak[0], 2)
                self.assertEqual(launched, [('a', 'smoke'), ('b', 'smoke'), ('a', 'formal'), ('b', 'formal')])
                status = q.read(root / 'parallel_status.json')
                self.assertEqual(status['status'], 'paused_technical_failure')
                self.assertEqual(status['active'], [])
                self.assertIn('b', status['completed'])
                self.assertNotIn('a', status['completed'])
                with self.assertRaises(AssertionError):
                    q.supervise()


if __name__ == '__main__':
    unittest.main()
