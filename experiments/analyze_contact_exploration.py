"""Audit and summarize the contact x energy 2x2, with paired episode uncertainty."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run_gsp_exploration_campaign import read_json, write_json, verify_snapshot, gate

METRICS = ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate", "final_coverage")


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    phase1 = ROOT / "experiments/gsp_exploration_20260906_phase1"
    old = ROOT / "experiments/d4_actor_gpu_100k_seed1_20260905"
    if read_json(phase1 / "queue_status.json")["status"] != "phase1_completed":
        raise RuntimeError("Do not analyze unfinished runs")
    manifest = verify_snapshot(phase1)
    old_state = read_json(old / "status.json")
    assert old_state["status"] == "completed" and old_state["training"] == manifest["formal"]
    checks = {}
    for source, expected in old_state["source_hashes"].items():
        path = Path(source)
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("multiagent") / str(path).split("multiagent\\", 1)[1]
        checks[str(relative)] = hashlib.sha256((phase1 / "code" / relative).read_bytes()).hexdigest() == expected
    assert all(checks.values())
    paths = {
        "raw": phase1 / "formal/raw_mlp/seed_1",
        "legacy_geometry": old / "d4_geometric_control/seed_1",
        "legacy_energy": old / "d4_dirichlet_gsp/seed_1",
        "contact_geometry": phase1 / "formal/contact_d4_geometry/seed_1",
        "contact_energy": phase1 / "formal/contact_d4_energy/seed_1",
    }
    summaries, episodes, curves, configs, artifacts = {}, {}, {}, {}, {}
    ignored_config_keys = {"model_name", "method", "actor_model"}
    for name, directory in paths.items():
        summaries[name] = read_json(directory / "summary.json")
        configs[name] = {k: v for k, v in read_json(directory / "resolved_config.json").items() if k not in ignored_config_keys}
        counters = read_json(directory / "training_counters.json")
        assert counters["global_env_steps"] == 100000
        assert counters["saved_checkpoints"] == [20000, 40000, 60000, 80000, 100000]
        device = read_json(directory / "gpu_device_audit.json")
        assert device["cuda_updates_verified"] == 12000 and not device["cpu_fallback"]
        shapes = read_json(directory / "tensor_shape_report.json")
        assert shapes["critic_input_tensor_shape"] == [64, 69] and shapes["replay_buffer_obs_dims"] == [18] * 3
        rows = csv_rows(directory / "per_evaluation_episode_metrics.csv")
        assert len(rows) == 500
        assert [int(row["test_seed"]) for row in rows] == list(range(1000000, 1000500))
        episodes[name] = {metric: np.array([float(row[metric]) for row in rows]) for metric in METRICS}
        for metric in METRICS:
            np.testing.assert_allclose(episodes[name][metric].mean(), summaries[name][metric], rtol=1e-10, atol=1e-10)
        curves[name] = csv_rows(directory / "learning_curve_eval.csv")
        assert [row["checkpoint"] for row in curves[name]] == [f"model_step{s}" for s in (20000, 40000, 60000, 80000, 100000)]
        assert all(int(row["eval_episodes"]) == 100 for row in curves[name])
        artifacts[name] = {"directory": str(directory), "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (directory / "summary.json", directory / "per_evaluation_episode_metrics.csv",
                                    directory / "checkpoints/model_final_100000.pt")}}
    assert all(config == configs["raw"] for config in configs.values())
    rng = np.random.default_rng(20260906)
    indexes = rng.integers(0, 500, size=(4000, 500))
    paired = []
    for left, right in (("contact_geometry", "raw"), ("contact_energy", "contact_geometry"),
                        ("contact_geometry", "legacy_geometry"), ("contact_energy", "legacy_energy")):
        for metric in METRICS:
            difference = episodes[left][metric] - episodes[right][metric]
            low, high = np.quantile(difference[indexes].mean(1), [.025, .975])
            paired.append({"candidate": left, "control": right, "metric": metric,
                           "mean_delta": float(difference.mean()), "episode_bootstrap_ci95_low": float(low),
                           "episode_bootstrap_ci95_high": float(high), "n_training_seeds": 1,
                           "scope": "conditional policy evaluation-scene uncertainty only; NOT training-seed uncertainty"})
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", {"passed": True, "common_source_hashes": checks,
               "config_equal_excluding": sorted(ignored_config_keys), "final_eval_seeds": [1000000, 1000499],
               "checkpoints": [20000, 40000, 60000, 80000, 100000], "all_run_steps": 100000,
               "curve_eval_seed_base_from_verified_code": 780000, "curve_eval_episodes": 100,
               "final_eval_episodes": 500, "artifacts": artifacts})
    write_json(output / "paired_deltas.json", paired)
    with (output / "paired_deltas.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    gates = {"energy_vs_geometry": gate(summaries["contact_energy"], summaries["contact_geometry"]),
             "energy_vs_raw": gate(summaries["contact_energy"], summaries["raw"])}
    write_json(output / "summary.json", {"runs": summaries, "gates": gates, "paper_claim_allowed": False,
                                         "training_seed_count": 1})
    lines = ["# 接触修正与 Dirichlet 第一组结果", "", "三个新运行均完成100k，旧D4配对的源码、参数和评估场景核对通过。所有数字来自固定100k终点，不选最佳checkpoint。", "",
             "| 方法 | Return | Hungarian距离 | Coverage AUC | Collision rate | Final coverage |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, summary in summaries.items():
        lines.append("| " + name + " | " + " | ".join(f"{summary[m]:.5f}" for m in METRICS) + " |")
    lines += ["", "## 结论", "",
              "接触修正能量版相对接触修正几何control：return +2.06323，Hungarian距离 -0.12043，coverage AUC +0.02489，但collision rate +0.01360（1.36个百分点）。未通过预先固定的碰撞不增加门槛。",
              "接触修正能量版相对旧运动能量版：return -0.02821，距离 -0.04402，AUC -0.01248，collision rate +0.00664。更准确的接触预测没有带来全面的策略收益。",
              "几何版接触修正也不是全面提升：对旧几何版return/AUC略好，assignment距离略差。单seed变化不能解释为稳定效应。",
              "", "## 原因与后续", "",
              "已观察事实：能量通道带来更高覆盖和更近的一对一分配，同时更多碰撞。局部接触预测准确度改善与实际策略质量改善是两个不同问题。",
              "可能原因（尚未证实）：Dirichlet差分或其缩放偏向积极接近目标；缺少邻居速度/动作限制真实下一状态预测；原始奖励下的学习可能接受更多碰撞来换取覆盖收益。不能仅凭这些指标确定哪一种原因成立。",
              "下一步保留本组为负结果/权衡案例，进入固定3D focal HKS的拓扑组。后续必要时做固定checkpoint的mask与动作/碰撞轨迹诊断，不把mask后评估冒充新训练结果。",
              "", "## 限制", "",
              "仍只有一个training seed。paired_deltas中的95% bootstrap区间仅针对相同500个评估初始场景，不代表训练seed稳健性；多配置共用筛选bank也不构成独立测试。",
              "旧/新图编码外的训练配置一致；相同CUDA代码不保证跨进程逐位确定性。不同raw与graph容量也要求使用geometry匹配对照。",
              "完整覆盖指标应与其他指标一起阅读；本组不能声称解决了协作覆盖任务。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "passed", "gates": gates, "output": str(output)}))


if __name__ == "__main__":
    main()
