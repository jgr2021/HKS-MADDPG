"""Validate a completed topology screen and audit focal-star HKS degeneracy."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.maddpg import MADDPG
from experiments.run_d4_actor_gpu_screen import training_args, validate_run
from experiments.run_gsp_exploration_campaign import read_json, write_json, verify_snapshot, gate, run_directory
from utils.exploration_topology_policies import POLICIES, Topology4EgoPolicy, adjacency_from_local_obs, register_policies
from utils.make_env import make_env

METRICS = ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate", "final_coverage")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def feature_audit(campaign, output):
    torch.set_num_threads(1)
    env = make_env("simple_spread", discrete_action=True)
    bank, kinds = [], []
    try:
        for seed in range(345000, 345500):
            np.random.seed(seed)
            env.seed(seed)
            bank.extend(env.reset())
            kinds.extend(["reset"] * 3)
        checkpoint = campaign / "formal/explore_hks_4ego/seed_1/checkpoints/model_final_100000.pt"
        model = MADDPG.init_from_save(checkpoint)
        model.prep_rollouts(device="cpu")
        for seed in range(346000, 346040):
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.seed(seed)
            obs = env.reset()
            for _ in range(25):
                bank.extend(obs)
                kinds.extend(["star_policy_rollout"] * 3)
                tensors = [torch.as_tensor(row, dtype=torch.float32).unsqueeze(0) for row in obs]
                with torch.no_grad():
                    actions = [item.numpy().ravel() for item in model.step(tensors, explore=False)]
                obs, _, _, _ = env.step(actions)
    finally:
        env.close()
    raw = torch.tensor(np.array(bank), dtype=torch.float32)
    np.savez_compressed(output / "local_observation_audit_bank.npz", observations=raw.numpy(), kind=np.array(kinds))
    star = Topology4EgoPolicy(18, 5).eval()
    expected = .5 * (1 + torch.exp(-2 * star.hks_times))
    report = {"n_local_observations": len(bank), "analytic_star_hks": expected.tolist(),
              "formula": "For a positive weighted star with unfloored degrees, HKS_center(t)=(1+exp(-2t))/2",
              "qualification": "Degree floor/underflow can break the identity; report both reset and policy-state banks.",
              "training_interactions_added": 0, "state_source": "raw per-actor local observations; no world/state features", "banks": {}}
    for kind in sorted(set(kinds)):
        inputs = raw[np.array(kinds) == kind]
        values = star._hks_features(inputs)
        weights = adjacency_from_local_obs(inputs, star.topology, star.edge_mask, star.edge_sigma)
        degree_floored = (weights.sum(-1) < 1e-12).any(1)
        row = {"observations": len(inputs), "star_std": values.double().std(0, unbiased=False).tolist(),
               "star_max_absolute_error_vs_constant": (values - expected).abs().max().item(),
               "degree_floor_fraction": degree_floored.float().mean().item(), "descriptor_std_by_method": {}}
        for name, cls in POLICIES.items():
            value = cls(18, 5).eval()._hks_features(inputs)
            row["descriptor_std_by_method"][name] = value.double().std(0, unbiased=False).tolist()
        report["banks"][kind] = row
    write_json(output / "feature_information_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    campaign = options.campaign.resolve()
    if read_json(campaign / "queue_status.json")["status"] != "study_completed":
        raise RuntimeError("Only analyze fully validated, completed studies")
    manifest = verify_snapshot(campaign)
    register_policies()
    torch.set_num_threads(1)
    summaries, episode_arrays, timings, validations = {}, {}, [], {}
    reference_config = None
    ignore = {"model_name", "method", "actor_model", "actor_input_dim"}
    for method, spec in manifest["methods"].items():
        directory = run_directory(campaign, "formal", method, spec)
        state = read_json(campaign / f"formal_{method}.json")
        assert state["status"] == "completed"
        validation = validate_run(directory, training_args("formal"))
        assert state["validation"] == validation
        validations[method] = validation
        summary = read_json(directory / "summary.json")
        summaries[method] = summary
        config = {k: v for k, v in read_json(directory / "resolved_config.json").items() if k not in ignore}
        if reference_config is None:
            reference_config = config
        assert config == reference_config
        rows = read_csv(directory / "per_evaluation_episode_metrics.csv")
        assert len(rows) == 500 and [int(r["test_seed"]) for r in rows] == list(range(1000000, 1000500))
        episode_arrays[method] = {key: np.array([float(row[key]) for row in rows]) for key in METRICS}
        for key in METRICS:
            np.testing.assert_allclose(episode_arrays[method][key].mean(), summary[key], atol=1e-10, rtol=1e-10)
        curves = read_csv(directory / "learning_curve_eval.csv")
        assert [row["checkpoint"] for row in curves] == [f"model_step{s}" for s in (20000, 40000, 60000, 80000, 100000)]
        assert all(int(row["eval_episodes"]) == 100 for row in curves)
        if not spec.get("reuse_from"):
            elapsed = (datetime.strptime(state["finished_at"], "%Y-%m-%dT%H:%M:%S%z") - datetime.strptime(state["started_at"], "%Y-%m-%dT%H:%M:%S%z")).total_seconds()
            timings.append({"method": method, "training_minutes": summary["train_wall_time_sec"] / 60,
                            "worker_including_evaluation_minutes": elapsed / 60, "finished_at": state["finished_at"]})
    start = datetime.strptime(read_json(campaign / "controller_started.json")["started_at"], "%Y-%m-%dT%H:%M:%S%z")
    end = max(datetime.strptime(row["finished_at"], "%Y-%m-%dT%H:%M:%S%z") for row in timings)
    elapsed = (end - start).total_seconds() / 60
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    info = feature_audit(campaign, output)
    comparisons, paired = {}, []
    rng = np.random.default_rng(2026090602)
    bootstrap = rng.integers(0, 500, size=(3000, 500))
    interpretations = {}
    hypotheses = {
        "explore_geometric_stats3": "普通几何统计可能促使更积极靠近目标；碰撞权衡原因未证实。",
        "explore_hks_3aa": "图分支没有地标，可能更偏拥挤/避碰而不能表达目标分配。",
        "explore_hks_4ego": "已验证中心HKS在无degree-floor时为常数；收益不能归因为图几何信息，需同容量精确常数control。",
        "explore_hks_6al": "focal对角扩散返回值可能压缩了目标索引/方向；尺度或归一化效应仍待区分。",
        "explore_hks_6aal": "混合AA/AL可能改变覆盖与安全的权衡；单seed不能确认因果。",
        "explore_hks_6all": "LL上下文可能有益，但还没有排除单seed方差和更积极覆盖带来的碰撞代价。",
    }
    for method, spec in manifest["methods"].items():
        if not spec["control"]:
            continue
        names = list(dict.fromkeys([spec["control"], "raw_mlp"]))
        comparisons[method] = {name: gate(summaries[method], summaries[name]) for name in names}
        observed = "; ".join(f"vs {name}: {', '.join(item['failed_criteria'])}" for name, item in comparisons[method].items() if not item["passed"])
        interpretations[method] = {"observed_failure": observed, "possible_cause_not_proven": hypotheses[method]}
        for name in names:
            for key in METRICS:
                delta = episode_arrays[method][key] - episode_arrays[name][key]
                lo, hi = np.quantile(delta[bootstrap].mean(1), [.025, .975])
                paired.append({"method": method, "control": name, "metric": key, "delta": float(delta.mean()),
                               "episode_ci95_low": float(lo), "episode_ci95_high": float(hi)})
    write_json(output / "interpretations.json", interpretations)
    write_json(output / "paired_deltas.json", paired)
    write_json(output / "summary.json", {"runs": summaries, "comparisons": comparisons, "elapsed_minutes": elapsed,
                                          "timings": timings, "training_seed_count": 1, "paper_claim_allowed": False})
    write_json(output / "audit.json", {"passed": True, "validations": validations, "elapsed_minutes": elapsed,
               "started_at": start.isoformat(), "finished_at": end.isoformat(), "all_evaluation_seeds_matched": True,
               "all_configs_matched_except_actor_identity_width": True, "final_episodes": 500, "checkpoint_episodes": 100})
    lines = ["# 拓扑筛选2A结果", "", f"6个新方法依次完成smoke、100k及评估，总耗时 **{elapsed:.2f}分钟**。raw复用，不重复训练。", "",
             "| 方法 | Return | Hungarian距离 | AUC | Collision rate | Final coverage |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, summary in summaries.items():
        lines.append("| " + method + " | " + " | ".join(f"{summary[key]:.5f}" for key in METRICS) + " |")
    lines += ["", "## 结论", "",
              "没有方法同时通过对raw和21D几何control的全部预设门槛。6all对几何control四项均好，但对raw碰撞更高，属于值得保留的权衡候选，不是全面改进结论。",
              "4ego的return/Hungarian表现不能直接解释为目标几何的谱信息有效。对正权星形图，设归一化邻接为S，则中心偶数步返回概率=1、奇数步=0，因此HKS_center(t)=exp(-t)cosh(t)=(1+exp(-2t))/2，与具体边权无关。",
              "degree floor、underflow和float32舍入可能破坏上述精确恒等式，已分别记录reset及其训练后策略rollout中的偏离，而不是忽略它们。",
              "", "## 数值审计", "", "输入仅为每个actor自身raw18D观测。500个reset场景及40条冻结star策略rollout只用于特征诊断，不新增训练。"]
    for kind, row in info["banks"].items():
        lines.append(f"- {kind}: n={row['observations']}, star std={row['star_std']}, max deviation={row['star_max_absolute_error_vs_constant']:.8g}, degree-floor fraction={row['degree_floor_fraction']:.8g}.")
    lines += ["", "## 失败记录与下一组", ""]
    for method, item in interpretations.items():
        lines.append(f"- {method}: {item['observed_failure']}。原因/机制记录：{item['possible_cause_not_proven']}")
    lines += ["", "2B继续固定6AL，比较sigma倍率、bounded inverse、soft-radius和kNN union；补精确常数star3控制。固定二值K3,3图的HKS必为常数，取消其100k训练，改做数值负对照，因此2B仍只新增6个正式run。",
              "复用raw、几何control和默认6AL必须核对源码/参数/模型哈希，且禁止向旧目录回写新的comparison文件。",
              "", "## 限制", "", "只有一个training seed。配对bootstrap区间仅描述给定冻结策略的评估场景不确定性，不是训练seed区间。共用筛选bank，不能宣称独立测试或ICASSP级稳健性。",
              "各方法final3：" + ", ".join(f"{name}={row['final3']}" for name, row in summaries.items()) + "。平均reward改善不等于解决三地标完整覆盖。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "elapsed_minutes": elapsed, "feature_audit": info["banks"], "output": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
