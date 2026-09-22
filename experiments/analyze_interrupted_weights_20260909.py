"""User-authorized last-checkpoint evaluation; no training or checkpoint changes."""
import analyze_interrupted_sigma_20260908 as evaluation

evaluation.OUTPUT = evaluation.CAMPAIGN / 'analysis_interrupted_weights_20260909'
evaluation.SPECS = (
    dict(method='explore_hks_inverse', reference_method='explore_hks_6al',
         expected_checkpoint_step=1620000,
         model_run_directory=evaluation.CAMPAIGN / 'models/simple_spread/pgt_m13_s1_0908_142047f/run1'),
    dict(method='explore_hks_soft_radius', reference_method='explore_hks_6al',
         expected_checkpoint_step=1580000,
         model_run_directory=evaluation.CAMPAIGN / 'models/simple_spread/pgt_m14_s1_0908_142047f/run1'),
)

if __name__ == '__main__':
    evaluation.main()
