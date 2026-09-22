"""Evaluate user-approved interrupted operator runs without training."""
import analyze_interrupted_sigma_20260908 as evaluation

evaluation.OUTPUT = evaluation.CAMPAIGN / 'analysis_interrupted_operators_20260910'
evaluation.SPECS = (
    dict(method='explore_hks_self_loop', reference_method='explore_hks_6al',
         expected_checkpoint_step=1780000,
         model_run_directory=evaluation.CAMPAIGN / 'models/simple_spread/pgt_m17_s1_0909_183053f/run1'),
    dict(method='explore_rwse_248', reference_method='explore_hks_6al',
         expected_marker_status='failed', expected_checkpoint_step=1820000,
         model_run_directory=evaluation.CAMPAIGN / 'models/simple_spread/pgt_m18_s1_0909_191936f/run1'),
)

if __name__ == '__main__':
    evaluation.main()
