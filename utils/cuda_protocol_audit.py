"""CUDA requirement and passive device/data-flow auditing for MADDPG updates."""

import json
from pathlib import Path

import torch

from algorithms.maddpg import MADDPG


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled for this experiment")
    value = torch.ones(8, device="cuda", requires_grad=True)
    value.square().sum().backward()
    torch.cuda.synchronize()
    return {"required_device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "tensor_device": str(value.device), "gradient_device": str(value.grad.device)}


class CudaAuditedMADDPG(MADDPG):
    """Call the unchanged update while checking actual tensors and gradients."""

    device_audit_path = None

    def update(self, sample, agent_i, *args, **kwargs):
        if not hasattr(self, "_cuda_update_count"):
            self._cuda_update_count = 0
            self._cuda_agent_audits = {}
        obs, actions, rewards, next_obs, dones = sample
        for family in sample:
            if any(value.device.type != "cuda" for value in family):
                raise RuntimeError("Training sample contains a non-CUDA tensor")
        agent = self.agents[agent_i]
        modules = {"actor": agent.policy, "critic": agent.critic,
                   "target_actor": agent.target_policy, "target_critic": agent.target_critic}
        if any(parameter.device.type != "cuda"
               for module in modules.values() for parameter in module.parameters()):
            raise RuntimeError("Training model contains a non-CUDA parameter")
        first = agent_i not in self._cuda_agent_audits
        record = {"agent": agent_i, "forwards": {},
                  "sample_devices": [[str(value.device) for value in family] for family in sample]}
        handles = []
        if first:
            raw_joint = torch.cat(obs, dim=1)
            next_joint = torch.cat(next_obs, dim=1)

            def check_input(name, expected, width):
                def hook(module, inputs):
                    value = inputs[0]
                    purpose = "sampled_transition"
                    matches = torch.equal(value[:, :expected.shape[1]], expected)
                    audit_obs = self.policy_audit_observations[agent_i] if name == "actor" else None
                    if not matches and audit_obs is not None and torch.equal(value, audit_obs):
                        matches = True
                        purpose = "existing_read_only_policy_audit"
                    if value.shape[1] != width or not matches:
                        raise RuntimeError(f"{name} received non-raw or mismatched observations")
                    record["forwards"].setdefault(name, []).append(
                        {"shape": list(value.shape), "device": str(value.device),
                         "raw_values_match": True, "purpose": purpose})
                return hook

            for name, expected, width in (
                ("actor", obs[agent_i], 18), ("target_actor", next_obs[agent_i], 18),
                ("critic", raw_joint, 69), ("target_critic", next_joint, 69),
            ):
                handles.append(modules[name].register_forward_pre_hook(check_input(name, expected, width)))
        try:
            result = super().update(sample, agent_i, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        self._cuda_update_count += 1
        if first:
            for name in ("actor", "critic"):
                gradients = [parameter.grad for parameter in modules[name].parameters()
                             if parameter.grad is not None]
                if not gradients or any(gradient.device.type != "cuda" for gradient in gradients):
                    raise RuntimeError(f"No CUDA backward gradient found for {name}")
                if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                    raise RuntimeError(f"Non-finite {name} gradients")
                record[f"{name}_gradient_devices"] = sorted({str(gradient.device) for gradient in gradients})
                record[f"{name}_parameter_device"] = str(next(modules[name].parameters()).device)
            if set(record["forwards"]) != set(modules):
                raise RuntimeError("The first update did not execute all actor/critic paths")
            record["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            self._cuda_agent_audits[agent_i] = record
        if first or self._cuda_update_count % 1000 == 0:
            self.write_device_audit()
        return result

    def write_device_audit(self):
        if self.device_audit_path is not None and hasattr(self, "_cuda_agent_audits"):
            path = Path(self.device_audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "gpu_name": torch.cuda.get_device_name(0), "cuda_updates_verified": self._cuda_update_count,
                "agents": self._cuda_agent_audits, "cpu_fallback": False,
                "scope": "every update: model/sample devices; first update per agent: raw forwards and CUDA gradients; environment rollout remains CPU",
            }, indent=2), encoding="utf-8")

    def save(self, filename):
        result = super().save(filename)
        self.write_device_audit()
        return result
