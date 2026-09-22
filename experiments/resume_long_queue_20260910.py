"""Continue five remaining methods after verified interrupted-run analysis."""
import resume_long_queue_20260908 as queue

queue.RUN_TAG = '20260910'
queue.EXPECTED_DONE = 11
queue.EXPECTED_PENDING = 5
queue.SKIPPED.update({'explore_hks_inverse', 'explore_hks_soft_radius',
                      'explore_hks_self_loop', 'explore_rwse_248'})
verify_original = queue.verify_analysis


def verify_all_analyses():
    original = queue.ANALYSIS
    merged = {'runs': []}
    try:
        for relative in ('analysis_interrupted_sigma_20260908',
                         'analysis_interrupted_weights_20260909',
                         'analysis_interrupted_operators_20260910'):
            queue.ANALYSIS = queue.CAMPAIGN / relative / 'analysis.json'
            data = verify_original()
            for run in data['runs']:
                run['analysis_path'] = str(queue.ANALYSIS)
                merged['runs'].append(run)
    finally:
        queue.ANALYSIS = original
    return merged


queue.verify_analysis = verify_all_analyses

if __name__ == '__main__':
    queue.main()
