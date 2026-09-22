"""Continue the remaining original method without retrying interrupted runs."""
import sys
from pathlib import Path

import resume_long_queue_20260910 as prior

queue = prior.queue
queue.RUN_TAG = '20260911'
queue.EXPECTED_DONE = 13
queue.EXPECTED_PENDING = 1
queue.SKIPPED.update({'explore_wks3', 'explore_landmark_heat3'})
verify_previous = queue.verify_analysis


def verify_with_interrupted():
    data = verify_previous()
    for name, step, logged in (
            ('explore_wks3', 1200000, 1219200),
            ('explore_landmark_heat3', 1100000, 1107800)):
        marker = queue.read(queue.CAMPAIGN / ('formal_' + name + '.json'))
        assert marker['status'] == 'running'
        assert not queue.pid_alive(marker['pid'])
        checkpoint = Path(marker['model_run_directory']) / 'incremental' / ('model_step%d.pt' % step)
        assert checkpoint.is_file()
        data['runs'].append(dict(method=name, checkpoint_env_steps=step,
                                 last_logged_env_steps=logged, results={},
                                 pre_reconciliation_marker=marker,
                                 checkpoint_sha256=queue.digest(checkpoint)))
    return data


queue.verify_analysis = verify_with_interrupted

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'scheduled':
        log = queue.CAMPAIGN / 'operations/continuation_20260911.scheduler.log'
        with log.open('x', encoding='utf-8', buffering=1) as stream:
            sys.stdout = sys.stderr = stream
            sys.argv[1] = 'supervise'
            queue.main()
    else:
        queue.main()
