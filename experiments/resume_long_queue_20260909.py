"""Continue remaining methods; user skips the interrupted weight runs without evaluation."""
import resume_long_queue_20260908 as queue

queue.RUN_TAG = '20260909'
queue.EXPECTED_PENDING = 9
queue.EXTRA_INTERRUPTED = [
    dict(method='explore_hks_inverse', last_logged_env_steps=1625200,
         checkpoint_env_steps=1620000),
    dict(method='explore_hks_soft_radius', last_logged_env_steps=1584200,
         checkpoint_env_steps=1580000),
]
queue.SKIPPED.update(r['method'] for r in queue.EXTRA_INTERRUPTED)

if __name__ == '__main__':
    queue.main()
