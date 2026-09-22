import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from experiments import catalog76


class ContinuationTests(unittest.TestCase):
    def exercise(self, smoke_failure=False, priority=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            campaign = root / 'campaign'
            campaign.mkdir()
            receipt = campaign / 'catalog_started.json'
            receipt.write_text('original receipt')
            old = campaign / 'formal_interrupted.json'
            old.write_text('original interrupted marker')
            manifest = dict(prepared_from=str(source), methods={
                'finished': {}, 'interrupted': {}, 'new': {'control': 'raw_mlp'}})
            if priority:
                manifest['methods'] = {'earlier': {'control': 'raw_mlp'}, **manifest['methods']}
            calls = []
            analyses = []

            def spawn(command, **kwargs):
                method = command[command.index('--method')+1]
                phase = command[command.index('--phase')+1]
                calls.append((method, phase))
                failed = smoke_failure and phase == 'smoke'
                (campaign / (phase+'_'+method+'.json')).write_text(json.dumps({
                    'status': 'failed' if failed else 'completed',
                    'validation': {'passed': not failed}}))
                return Mock(pid=12345, poll=Mock(return_value=1 if failed else 0))

            with patch('experiments.run_gsp_exploration_campaign.verify_snapshot', return_value=manifest), \
                 patch.object(catalog76, 'memory_info', return_value={'avail_phys': 20*1024**3, 'avail_page': 20*1024**3}), \
                 patch.object(catalog76.shutil, 'disk_usage', return_value=Mock(free=20*1024**3)), \
                 patch.object(catalog76.subprocess, 'Popen', side_effect=spawn), \
                 patch.object(catalog76.time, 'sleep'):
                catalog76.run(campaign, True, inherited_done={'raw_mlp'},
                    priority_methods=['new'] if priority else [],
                    on_complete=analyses.append, continuation={
                        'completed': ['finished'], 'skipped': ['interrupted'],
                        'smoke_completed': ['finished', 'interrupted']})
            self.assertEqual(receipt.read_text(), 'original receipt')
            self.assertEqual(old.read_text(), 'original interrupted marker')
            state = json.loads((campaign / 'catalog_status.json').read_text())
            self.assertEqual(state['skipped'], ['interrupted'])
            return calls, analyses, state

    def test_only_unstarted_method_runs_smoke_then_formal(self):
        calls, analyses, state = self.exercise()
        self.assertEqual(calls, [('new', 'smoke'), ('new', 'formal')])
        self.assertEqual(analyses, ['new'])
        self.assertEqual(state['status'], 'completed_with_skips')
        self.assertEqual(state['completed'], ['finished', 'new'])

    def test_failed_smoke_stops_admissions_without_retry(self):
        calls, analyses, state = self.exercise(smoke_failure=True)
        self.assertEqual(calls, [('new', 'smoke')])
        self.assertEqual(analyses, [])
        self.assertEqual(state['status'], 'paused_technical_failure')

    def test_priority_dispatches_before_manifest_order(self):
        calls, analyses, state = self.exercise(priority=True)
        self.assertEqual(calls[:2], [('new', 'smoke'), ('earlier', 'smoke')])
        self.assertEqual(set(analyses), {'new', 'earlier'})
        self.assertEqual(state['status'], 'completed_with_skips')


if __name__ == '__main__':
    unittest.main()
