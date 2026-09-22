import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import catalog76 as c


class ContinuationTests(unittest.TestCase):
    def test_inherited_control_not_retrained_and_analysis_before_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            manifest = dict(prepared_from=str(source), methods={
                'a': {'control': 'raw'}, 'b': {'control': 'a'}})
            events = []

            class Process:
                pid = 123

                def __init__(self, command, **kwargs):
                    name = command[command.index('--method') + 1]
                    phase = command[command.index('--phase') + 1]
                    events.append((name, phase))
                    (root / (phase + '_' + name + '.json')).write_text(json.dumps(
                        {'status': 'completed', 'validation': {'passed': True}}))

                def poll(self):
                    return 0

            def analyze(name):
                events.append((name, 'analysis'))

            with patch('experiments.run_gsp_exploration_campaign.verify_snapshot', return_value=manifest), \
                    patch.object(c, 'memory_info', return_value={'avail_phys': 10*1024**3, 'avail_page': 10*1024**3}), \
                    patch.object(c.subprocess, 'Popen', Process), patch.object(c.time, 'sleep'):
                c.run(root, True, inherited_done={'raw'}, on_complete=analyze)
            self.assertEqual(events, [('a', 'smoke'), ('a', 'formal'), ('a', 'analysis'),
                                      ('b', 'smoke'), ('b', 'formal'), ('b', 'analysis')])
            state = c.read(root / 'catalog_status.json')
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['completed'], ['a', 'b'])
            self.assertEqual(state['inherited_completed'], ['raw'])


if __name__ == '__main__':
    unittest.main()
