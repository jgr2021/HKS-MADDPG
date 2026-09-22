import tempfile
import unittest
from pathlib import Path

import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env


class ActorTeacherTrustRegionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.env = make_parallel_env("simple_spread", 1, 7007, True)
        self.model = MADDPG.init_from_env(
            self.env,
            actor_model="counterfactual_active_probe",
            actor_anchor={
                "mode": "fixed_teacher_kl",
                "max_kl": 0.002,
                "temperature": 1.0,
                "bisection_steps": 16,
            },
        )
        self.model.capture_policy_anchors()

    def tearDown(self):
        self.env.close()

    def test_projection_constrains_first_large_proposal(self):
        observations = torch.randn(64, 18)
        policy = self.model.agents[0].policy
        parameters = list(policy.parameters())
        old = [parameter.detach().clone() for parameter in parameters]
        with torch.no_grad():
            policy.scorer[-1].bias[0].add_(4.0)
            policy.scorer[-1].weight[0, 0].add_(8.0)
        proposed = [parameter.detach().clone() for parameter in parameters]
        report = self.model._project_actor_update(0, observations, old, proposed)
        self.assertGreater(report["proposed_kl"], 0.002)
        self.assertLess(report["accepted_scale"], 1.0)
        self.assertLessEqual(report["accepted_kl"], 0.002001)

    def test_anchor_round_trip_is_preserved_in_checkpoint(self):
        observations = torch.randn(5, 18)
        anchor_logits = self.model.policy_anchors[0](observations).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.model.init_dict = {
                **self.model.init_dict,
                "actor_anchor": self.model.actor_anchor,
            }
            self.model.save(path)
            restored = MADDPG.init_from_save(path)
            restored_logits = restored.policy_anchors[0](observations).detach()
        self.assertIsNotNone(restored.policy_anchors[0])
        torch.testing.assert_close(anchor_logits, restored_logits)

    def test_anchor_does_not_change_actor_or_critic_boundaries(self):
        self.assertEqual(self.model.agent_init_params[0]["num_in_pol"], 18)
        self.assertEqual(self.model.agent_init_params[0]["num_in_critic"], 69)
        self.assertEqual(self.model.agents[0].policy(torch.randn(3, 18)).shape, (3, 5))

    def test_real_maddpg_update_reports_bounded_anchor_kl(self):
        batch = 16
        obs = [torch.randn(batch, 18) for _ in range(3)]
        next_obs = [torch.randn(batch, 18) for _ in range(3)]
        actions = [
            torch.nn.functional.one_hot(
                torch.randint(0, 5, (batch,)), num_classes=5
            ).float()
            for _ in range(3)
        ]
        rewards = [torch.randn(batch) for _ in range(3)]
        dones = [torch.zeros(batch) for _ in range(3)]
        self.model.update((obs, actions, rewards, next_obs, dones), 0)
        report = self.model.last_anchor_update
        self.assertIsNotNone(report)
        self.assertLessEqual(report["accepted_kl"], 0.002001)

    def test_old_policy_projection_bounds_each_proposal_without_fixed_teacher(self):
        model = MADDPG.init_from_env(
            self.env,
            actor_model="counterfactual_active_probe",
            actor_anchor={
                "mode": "old_policy_kl",
                "max_kl": 0.002,
                "temperature": 1.0,
                "bisection_steps": 16,
            },
        )
        observations = torch.randn(64, 18)
        policy = model.agents[0].policy
        parameters = list(policy.parameters())
        old = [parameter.detach().clone() for parameter in parameters]
        old_probabilities = torch.softmax(policy(observations).detach(), dim=1)
        with torch.no_grad():
            policy.scorer[-1].weight[0, 0].add_(10.0)
            policy.scorer[-1].bias[0].add_(5.0)
        proposed = [parameter.detach().clone() for parameter in parameters]
        report = model._project_actor_update_against_probabilities(
            0,
            observations,
            old,
            proposed,
            old_probabilities,
            mode="old_policy_kl",
        )
        self.assertGreater(report["proposed_kl"], 0.002)
        self.assertLess(report["accepted_scale"], 1.0)
        self.assertLessEqual(report["accepted_kl"], 0.002001)

    def test_fixed_teacher_penalty_has_zero_initial_gradient(self):
        model = MADDPG.init_from_env(
            self.env,
            actor_model="counterfactual_active_probe",
            actor_anchor={
                "mode": "fixed_teacher_kl_penalty",
                "coefficient": 10.0,
                "temperature": 1.0,
            },
        )
        model.capture_policy_anchors()
        observations = torch.randn(64, 18)
        penalty = model._fixed_teacher_kl(
            0, observations, track_student_grad=True
        )
        model.agents[0].policy_optimizer.zero_grad()
        penalty.backward()
        gradient_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.agents[0].policy.parameters()
            if parameter.grad is not None
        )
        self.assertAlmostEqual(float(penalty.item()), 0.0, places=7)
        self.assertLess(gradient_norm, 1e-5)

    def test_fixed_parameter_l2_is_normalized_and_uses_initial_actor(self):
        model = MADDPG.init_from_env(
            self.env,
            actor_model="counterfactual_active_probe",
            actor_anchor={
                "mode": "fixed_parameter_l2_penalty",
                "coefficient": 10.0,
            },
        )
        model.capture_policy_anchors()
        initial = model._fixed_parameter_l2(0)
        self.assertAlmostEqual(float(initial.item()), 0.0, places=9)
        policy = model.agents[0].policy
        parameter = next(policy.parameters())
        with torch.no_grad():
            parameter.view(-1)[0].add_(2.0)
        displaced = model._fixed_parameter_l2(0)
        count = sum(value.numel() for value in policy.parameters())
        self.assertAlmostEqual(float(displaced.item()), 4.0 / count, places=9)
        displaced.backward()
        self.assertIsNotNone(parameter.grad)
        self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_fixed_parameter_l2_real_update_reports_displacement(self):
        model = MADDPG.init_from_env(
            self.env,
            actor_model="counterfactual_active_probe",
            actor_anchor={
                "mode": "fixed_parameter_l2_penalty",
                "coefficient": 100.0,
            },
        )
        model.capture_policy_anchors()
        batch = 16
        obs = [torch.randn(batch, 18) for _ in range(3)]
        next_obs = [torch.randn(batch, 18) for _ in range(3)]
        actions = [
            torch.nn.functional.one_hot(
                torch.randint(0, 5, (batch,)), num_classes=5
            ).float()
            for _ in range(3)
        ]
        rewards = [torch.randn(batch) for _ in range(3)]
        dones = [torch.zeros(batch) for _ in range(3)]
        model.update((obs, actions, rewards, next_obs, dones), 0)
        report = model.last_anchor_update
        self.assertEqual(report["mode"], "fixed_parameter_l2_penalty")
        self.assertGreater(report["parameter_l2"], 0.0)

    def test_separate_actor_and_critic_learning_rates(self):
        model = MADDPG.init_from_env(
            self.env,
            actor_model="counterfactual_active_probe",
            actor_lr=1e-4,
            critic_lr=1e-2,
        )
        self.assertEqual(model.agents[0].policy_optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(model.agents[0].critic_optimizer.param_groups[0]["lr"], 1e-2)

    def test_critic_only_update_leaves_actor_unchanged(self):
        batch = 16
        obs = [torch.randn(batch, 18) for _ in range(3)]
        next_obs = [torch.randn(batch, 18) for _ in range(3)]
        actions = [
            torch.nn.functional.one_hot(
                torch.randint(0, 5, (batch,)), num_classes=5
            ).float()
            for _ in range(3)
        ]
        rewards = [torch.randn(batch) for _ in range(3)]
        dones = [torch.zeros(batch) for _ in range(3)]
        before = [
            parameter.detach().clone()
            for parameter in self.model.agents[0].policy.parameters()
        ]
        self.model.update(
            (obs, actions, rewards, next_obs, dones),
            0,
            update_actor=False,
        )
        after = list(self.model.agents[0].policy.parameters())
        for initial, final in zip(before, after):
            torch.testing.assert_close(initial, final)


if __name__ == "__main__":
    unittest.main()
