import unittest

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.make_env import make_env
from utils.networks import (
    LearnedActiveGSPResidualPolicy,
    LearnedActiveRWSEResidualPolicy,
    LearnedActiveSCFResidualPolicy,
    LearnedCrowdingGSPResidualPolicy,
    LearnedGatedDirichletControlPolicy,
    LearnedGatedDirichletGSPPolicy,
    LearnedGatedDirichletGSP025Policy,
    LearnedMatchingControlPolicy,
    LearnedRawPotentialResidualPolicy,
    LearnedRWSEPotentialControlPolicy,
    LearnedSCFPotentialControlPolicy,
    LearnedSpectralMatchingResidualPolicy,
    LinearMatchingControlPolicy,
    LinearSpectralMatchingPolicy,
    MatchingOnlyControlPolicy,
    SpectralMatchingOnlyPolicy,
    LocalGeometryResidualPolicy,
    LocalSpectralALResidualPolicy,
    _normalized_agent_landmark_adjacency,
    _action_conditioned_rwse_features_tensor,
    _action_conditioned_spectral_coordinate_features_tensor,
    _action_conditioned_spectral_matching_features_tensor,
    _candidate_soft_matching_tensor,
    _reconstruct_relative_task_nodes,
)


class LocalSpectralActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.obs = torch.randn(12, 18, dtype=torch.float32)

    def test_shapes_finite_and_raw_critic(self):
        for actor_model in ["local_geometry_residual", "local_spectral_al_residual"]:
            env = make_parallel_env("simple_spread", 1, 321, True)
            try:
                maddpg = MADDPG.init_from_env(env, actor_model=actor_model)
            finally:
                env.close()
            policy = maddpg.agents[0].policy
            output = policy(self.obs)
            self.assertEqual(tuple(output.shape), (12, 5))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(policy.mlp.fc1.in_features, 18)
            self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)
            self.assertEqual(maddpg.agents[0].target_critic.fc1.in_features, 69)
            self.assertEqual(policy.last_graph_node_shape, (12, 6, 7))

    def test_local_only_rows_do_not_interfere(self):
        policy = LocalSpectralALResidualPolicy(18, 5, agent_index=0)
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            expected = policy(self.obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_graph_branch_ignores_absolute_translation(self):
        policy = LocalSpectralALResidualPolicy(18, 5, agent_index=1)
        translated = self.obs.clone()
        translated[:, 2:4] += torch.tensor([17.0, -9.0])
        with torch.no_grad():
            expected = policy.graph_embedding(self.obs)
            actual = policy.graph_embedding(translated)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_graph_reconstruction_uses_focal_relative_coordinates(self):
        positions, features = _reconstruct_relative_task_nodes(self.obs, 2)
        torch.testing.assert_close(positions[:, 2, :], torch.zeros(12, 2))
        torch.testing.assert_close(positions[:, 3:, :], self.obs[:, 4:10].reshape(12, 3, 2))
        self.assertEqual(tuple(features.shape), (12, 6, 7))

    def test_normalized_adjacency_is_symmetric_and_finite(self):
        positions, _ = _reconstruct_relative_task_nodes(self.obs, 0)
        adjacency = _normalized_agent_landmark_adjacency(positions)
        torch.testing.assert_close(adjacency, adjacency.transpose(1, 2))
        self.assertTrue(torch.isfinite(adjacency).all())
        self.assertEqual(int(torch.count_nonzero(adjacency[:, :3, :3])), 0)
        self.assertEqual(int(torch.count_nonzero(adjacency[:, 3:, 3:])), 0)

    def test_spectral_diffusion_changes_geometry_control_embedding(self):
        geometry = LocalGeometryResidualPolicy(18, 5, agent_index=0)
        spectral = LocalSpectralALResidualPolicy(18, 5, agent_index=0)
        spectral.load_state_dict(geometry.state_dict())
        with torch.no_grad():
            geometry_embedding = geometry.graph_embedding(self.obs)
            spectral_embedding = spectral.graph_embedding(self.obs)
        self.assertFalse(torch.allclose(geometry_embedding, spectral_embedding))

    def test_backward_reaches_spectral_branch(self):
        policy = LocalSpectralALResidualPolicy(18, 5, agent_index=0)
        loss = policy(self.obs).square().mean()
        loss.backward()
        self.assertIsNotNone(policy.graph_out.weight.grad)
        self.assertGreater(float(policy.graph_out.weight.grad.abs().sum()), 0.0)

    def test_batch_sizes_and_extreme_values(self):
        policy = LocalSpectralALResidualPolicy(18, 5, agent_index=0)
        policy.eval()
        for batch_size in [1, 12, 64]:
            obs = torch.zeros(batch_size, 18)
            obs[:, 4:14] = 1e4
            with torch.no_grad():
                output = policy(obs)
            self.assertEqual(tuple(output.shape), (batch_size, 5))
            self.assertTrue(torch.isfinite(output).all())

    def test_learned_action_residual_is_local_finite_and_near_raw_at_init(self):
        policy = LearnedActiveGSPResidualPolicy(18, 5, agent_index=0)
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            raw_logits = policy.mlp(self.obs)
            logits = policy(self.obs)
            first_changed = policy(changed)[:1]
        self.assertEqual(policy.last_action_feature_shape, (12, 5, 4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertLess(float((logits - raw_logits).abs().max()), 0.02)
        torch.testing.assert_close(first_changed, logits[:1], rtol=1e-5, atol=1e-6)

    def test_learned_active_features_add_only_spectral_energy_terms(self):
        raw_control = LearnedRawPotentialResidualPolicy(18, 5, agent_index=0)
        active = LearnedActiveGSPResidualPolicy(18, 5, agent_index=0)
        with torch.no_grad():
            raw_features = raw_control.action_features(self.obs)
            active_features = active.action_features(self.obs)
        torch.testing.assert_close(raw_features[:, :, :2], active_features[:, :, :2])
        torch.testing.assert_close(raw_features[:, :, 2:], torch.zeros_like(raw_features[:, :, 2:]))
        self.assertGreater(float(active_features[:, :, 2:].abs().sum()), 0.0)
        torch.testing.assert_close(active_features[:, 0, :], torch.zeros_like(active_features[:, 0, :]))

    def test_learned_action_residual_backward_reaches_feature_path(self):
        policy = LearnedActiveGSPResidualPolicy(18, 5, agent_index=0)
        loss = policy(self.obs).square().mean()
        loss.backward()
        grad = policy.feature_encoder[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_crowding_variant_masks_only_coverage_energy(self):
        full = LearnedActiveGSPResidualPolicy(18, 5, agent_index=0)
        crowding = LearnedCrowdingGSPResidualPolicy(18, 5, agent_index=0)
        with torch.no_grad():
            full_features = full.action_features(self.obs)
            crowding_features = crowding.action_features(self.obs)
        torch.testing.assert_close(crowding_features[:, :, :2], full_features[:, :, :2])
        torch.testing.assert_close(
            crowding_features[:, :, 2], torch.zeros_like(crowding_features[:, :, 2])
        )
        torch.testing.assert_close(crowding_features[:, :, 3], full_features[:, :, 3])

    def test_action_conditioned_rwse_shape_noop_and_finite(self):
        for agent_index in range(3):
            features = _action_conditioned_rwse_features_tensor(self.obs, agent_index)
            self.assertEqual(tuple(features.shape), (12, 5, 5))
            self.assertTrue(torch.isfinite(features).all())
            torch.testing.assert_close(
                features[:, 0, :], torch.zeros_like(features[:, 0, :]), atol=1e-6, rtol=1e-6
            )

    def test_rwse_control_masks_only_spectral_channels(self):
        active = LearnedActiveRWSEResidualPolicy(18, 5, agent_index=1)
        control = LearnedRWSEPotentialControlPolicy(18, 5, agent_index=1)
        with torch.no_grad():
            active_features = active.action_features(self.obs)
            control_features = control.action_features(self.obs)
        torch.testing.assert_close(active_features[:, :, :2], control_features[:, :, :2])
        torch.testing.assert_close(
            control_features[:, :, 2:], torch.zeros_like(control_features[:, :, 2:])
        )
        self.assertGreater(float(active_features[:, :, 2:].abs().sum()), 0.0)

    def test_rwse_actor_is_local_only_and_preserves_raw_critic(self):
        env = make_parallel_env("simple_spread", 1, 444, True)
        try:
            maddpg = MADDPG.init_from_env(env, actor_model="learned_active_rwse_residual")
        finally:
            env.close()
        policy = maddpg.agents[0].policy
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            expected = policy(self.obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(policy.last_action_feature_shape, (12, 5, 5))
        self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)
        self.assertEqual(maddpg.agents[0].target_critic.fc1.in_features, 69)

    def test_rwse_extreme_overlap_and_far_geometry(self):
        for magnitude in [0.0, 1e4]:
            obs = torch.zeros(64, 18)
            obs[:, 4:14] = magnitude
            features = _action_conditioned_rwse_features_tensor(obs, 0)
            self.assertTrue(torch.isfinite(features).all())

    def test_scf_shape_noop_finite_and_directional_channels(self):
        features = _action_conditioned_spectral_coordinate_features_tensor(self.obs, 0)
        self.assertEqual(tuple(features.shape), (12, 5, 6))
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(
            features[:, 0, :], torch.zeros_like(features[:, 0, :]), atol=1e-6, rtol=1e-6
        )
        self.assertGreater(float(features[:, :, 2:].abs().sum()), 0.0)

    def test_scf_control_masks_only_directional_spectral_channels(self):
        active = LearnedActiveSCFResidualPolicy(18, 5, agent_index=2)
        control = LearnedSCFPotentialControlPolicy(18, 5, agent_index=2)
        with torch.no_grad():
            active_features = active.action_features(self.obs)
            control_features = control.action_features(self.obs)
        torch.testing.assert_close(active_features[:, :, :2], control_features[:, :, :2])
        torch.testing.assert_close(
            control_features[:, :, 2:], torch.zeros_like(control_features[:, :, 2:])
        )

    def test_scf_local_only_and_extreme_geometry(self):
        policy = LearnedActiveSCFResidualPolicy(18, 5, agent_index=0)
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            expected = policy(self.obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for magnitude in [0.0, 1e4]:
            obs = torch.zeros(64, 18)
            obs[:, 4:14] = magnitude
            features = _action_conditioned_spectral_coordinate_features_tensor(obs, 0)
            self.assertTrue(torch.isfinite(features).all())

    def test_gated_dirichlet_initial_function_matches_control(self):
        active = LearnedGatedDirichletGSPPolicy(18, 5, agent_index=0)
        control = LearnedGatedDirichletControlPolicy(18, 5, agent_index=0)
        control.load_state_dict(active.state_dict())
        active.eval()
        control.eval()
        with torch.no_grad():
            active_logits = active(self.obs)
            control_logits = control(self.obs)
        torch.testing.assert_close(active_logits, control_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            active.last_energy_gate_mean, torch.zeros_like(active.last_energy_gate_mean)
        )

    def test_gated_dirichlet_gradient_reaches_both_gate_outputs(self):
        policy = LearnedGatedDirichletGSPPolicy(18, 5, agent_index=0)
        loss = policy(self.obs).square().mean()
        loss.backward()
        grad = policy.energy_gate[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertEqual(tuple(grad.shape), (2, 64))
        self.assertTrue((grad.abs().sum(dim=1) > 0).all())

    def test_gated_dirichlet_energy_correction_is_bounded(self):
        policy = LearnedGatedDirichletGSPPolicy(18, 5, agent_index=0)
        policy.energy_gate[-1].bias.data.fill_(100.0)
        policy.eval()
        with torch.no_grad():
            features = policy.action_features(self.obs)
            potentials = torch.tanh(features[:, :, :2] * policy.potential_scale)
            context = policy.context_encoder(self.obs).unsqueeze(1).expand(-1, 5, -1)
            potential_hidden = policy.potential_encoder(potentials)
            action_hidden = policy.action_embedding(policy.action_indices)
            action_hidden = action_hidden.unsqueeze(0).expand(len(self.obs), -1, -1)
            shared = torch.cat([context, potential_hidden, action_hidden], dim=2)
            gates = torch.tanh(policy.energy_gate(shared))
            energies = torch.tanh(features[:, :, 2:] * policy.energy_scale)
            correction = policy.energy_residual_scale * (gates * energies).sum(dim=2)
        self.assertLessEqual(float(correction.abs().max()), 0.2 + 1e-6)

    def test_gated_dirichlet_025_has_expected_bound(self):
        policy = LearnedGatedDirichletGSP025Policy(18, 5, agent_index=0)
        self.assertEqual(policy.energy_residual_scale, 0.25)
        self.assertEqual(2.0 * policy.energy_residual_scale, 0.5)

    def test_gated_dirichlet_local_only_shapes_and_raw_critic(self):
        env = make_parallel_env("simple_spread", 1, 555, True)
        try:
            maddpg = MADDPG.init_from_env(env, actor_model="learned_gated_dirichlet_gsp")
        finally:
            env.close()
        policy = maddpg.agents[0].policy
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            expected = policy(self.obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(policy.last_action_feature_shape, (12, 5, 4))
        self.assertEqual(policy.last_energy_gate_shape, (12, 5, 2))
        self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)
        self.assertEqual(maddpg.agents[0].target_critic.fc1.in_features, 69)

    def test_soft_matching_is_doubly_stochastic(self):
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(777)
            realistic_obs = torch.as_tensor(np.vstack(env.reset()), dtype=torch.float32)
            realistic_obs = realistic_obs.repeat(4, 1)
        finally:
            env.close()
        _, _, assignment = _candidate_soft_matching_tensor(realistic_obs, 0)
        ones = torch.ones(12, 5, 3)
        torch.testing.assert_close(assignment.sum(dim=2), ones, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(assignment.sum(dim=3), ones, atol=1e-4, rtol=1e-4)

    def test_spectral_matching_shape_noop_and_finite(self):
        for agent_index in range(3):
            features = _action_conditioned_spectral_matching_features_tensor(
                self.obs, agent_index
            )
            self.assertEqual(tuple(features.shape), (12, 5, 6))
            self.assertTrue(torch.isfinite(features).all())
            torch.testing.assert_close(
                features[:, 0, :], torch.zeros_like(features[:, 0, :]),
                atol=1e-6, rtol=1e-6
            )

    def test_spectral_matching_is_landmark_permutation_invariant(self):
        permuted = self.obs.clone()
        landmarks = self.obs[:, 4:10].reshape(12, 3, 2)
        permuted[:, 4:10] = landmarks[:, [2, 0, 1], :].reshape(12, 6)
        expected = _action_conditioned_spectral_matching_features_tensor(self.obs, 1)
        actual = _action_conditioned_spectral_matching_features_tensor(permuted, 1)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_matching_control_masks_only_spectral_channels(self):
        active = LearnedSpectralMatchingResidualPolicy(18, 5, agent_index=0)
        control = LearnedMatchingControlPolicy(18, 5, agent_index=0)
        with torch.no_grad():
            active_features = active.action_features(self.obs)
            control_features = control.action_features(self.obs)
        torch.testing.assert_close(active_features[:, :, :3], control_features[:, :, :3])
        torch.testing.assert_close(
            control_features[:, :, 3:], torch.zeros_like(control_features[:, :, 3:])
        )
        self.assertGreater(float(active_features[:, :, 3:].abs().sum()), 0.0)

    def test_spectral_matching_local_only_extreme_and_raw_critic(self):
        env = make_parallel_env("simple_spread", 1, 666, True)
        try:
            maddpg = MADDPG.init_from_env(env, actor_model="learned_spectral_matching_residual")
        finally:
            env.close()
        policy = maddpg.agents[0].policy
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            expected = policy(self.obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(policy.last_action_feature_shape, (12, 5, 6))
        self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)
        for magnitude in [0.0, 1e4]:
            obs = torch.zeros(64, 18)
            obs[:, 4:14] = magnitude
            features = _action_conditioned_spectral_matching_features_tensor(obs, 0)
            self.assertTrue(torch.isfinite(features).all())

    def test_linear_matching_initial_active_control_equivalence(self):
        active = LinearSpectralMatchingPolicy(18, 5, agent_index=0)
        control = LinearMatchingControlPolicy(18, 5, agent_index=0)
        control.load_state_dict(active.state_dict())
        active.eval()
        control.eval()
        with torch.no_grad():
            active_logits = active(self.obs)
            control_logits = control(self.obs)
        torch.testing.assert_close(active_logits, control_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            active.feature_weights, torch.zeros_like(active.feature_weights)
        )

    def test_linear_matching_all_active_weights_receive_gradient(self):
        policy = LinearSpectralMatchingPolicy(18, 5, agent_index=0)
        loss = policy(self.obs).square().mean()
        loss.backward()
        grad = policy.feature_weights.grad
        self.assertIsNotNone(grad)
        self.assertEqual(tuple(grad.shape), (6,))
        self.assertTrue((grad.abs() > 0).all())

    def test_linear_matching_local_only_and_raw_critic(self):
        env = make_parallel_env("simple_spread", 1, 888, True)
        try:
            maddpg = MADDPG.init_from_env(env, actor_model="linear_spectral_matching")
        finally:
            env.close()
        policy = maddpg.agents[0].policy
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            expected = policy(self.obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(policy.last_action_feature_shape, (12, 5, 6))
        self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)

    def test_matching_only_zero_logits_and_active_control_equivalence(self):
        active = SpectralMatchingOnlyPolicy(18, 5, agent_index=0)
        control = MatchingOnlyControlPolicy(18, 5, agent_index=0)
        control.load_state_dict(active.state_dict())
        with torch.no_grad():
            active_logits = active(self.obs)
            control_logits = control(self.obs)
        torch.testing.assert_close(active_logits, torch.zeros_like(active_logits))
        torch.testing.assert_close(active_logits, control_logits, rtol=0.0, atol=0.0)

    def test_matching_only_all_weights_receive_gradient(self):
        policy = SpectralMatchingOnlyPolicy(18, 5, agent_index=0)
        logits = policy(self.obs)
        target = torch.randn_like(logits)
        loss = ((logits - target) ** 2).mean()
        loss.backward()
        self.assertTrue((policy.feature_weights.grad.abs() > 0).all())

    def test_matching_only_maddpg_save_load_and_dimensions(self):
        env = make_parallel_env("simple_spread", 1, 999, True)
        try:
            maddpg = MADDPG.init_from_env(env, actor_model="spectral_matching_only")
        finally:
            env.close()
        policy = maddpg.agents[0].policy
        policy.eval()
        changed = self.obs.clone()
        changed[1:, :] += 1000.0
        with torch.no_grad():
            expected = policy(self.obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(policy.actor_input_dim, 18)
        self.assertEqual(policy.last_action_feature_shape, (12, 5, 6))
        self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)


if __name__ == "__main__":
    unittest.main()
