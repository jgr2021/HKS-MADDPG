"""Audit weight/operator/descriptor screens without touching run outputs."""

import argparse
import csv
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys


METRICS = ("return", "hungarian_assignment_distance", "coverage_radius_auc",
           "collision_step_rate", "final_coverage")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--study", choices=("weights", "operators", "descriptors"), default="weights")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    # Use the same immutable policy/validation definitions as this campaign.
    sys.path.insert(0, str(campaign / "code"))
    import numpy as np
    import torch
    from experiments.run_gsp_exploration_campaign import (
        gate, read_json, run_directory, verify_snapshot, write_json)
    from experiments.run_d4_actor_gpu_screen import training_args, validate_run

    torch.set_num_threads(1)
    manifest = verify_snapshot(campaign)
    if manifest["study"] != args.study:
        raise RuntimeError("Requested study does not match the frozen manifest")
    planned_new = 6 if args.study == "weights" else 4
    if sum(not spec.get("reuse_from") for spec in manifest["methods"].values()) != planned_new:
        raise RuntimeError("Unexpected number of new formal runs for this study")
    title = {"weights": "边权组", "operators": "算子组", "descriptors": "描述符组"}[args.study]
    next_action = {"weights": "整组完成后才审计并准备2C。",
                   "operators": "整组完成后才审计并准备2D。",
                   "descriptors": "这批最多20个新正式运行的预算到此结束。整组审计后总结并暂停自动跟进，等待用户决定，不再自动启动训练。"}[args.study]
    queue = read_json(campaign / "queue_status.json")
    complete = queue["status"] == "study_completed"
    if not complete and not args.allow_partial:
        raise RuntimeError("Study incomplete; use --allow-partial for an explicit progress snapshot")
    for module in manifest["registration_modules"]:
        importlib.import_module(module).register_policies()
    summaries, audit, sources, timings, episodes = {}, {}, {}, [], {}
    reference_config = None
    ignore = {"model_name", "method", "actor_model", "actor_input_dim"}
    for method, spec in manifest["methods"].items():
        marker = ("reused/" if spec.get("reuse_from") else "formal/") + method
        if marker not in queue["completed"]:
            continue
        directory = run_directory(campaign, "formal", method, spec)
        state = read_json(campaign / ("formal_" + method + ".json"))
        assert state["status"] == "completed"
        validation = validate_run(directory, training_args("formal"))
        assert state["validation"] == validation
        if spec.get("reuse_from"):
            old_campaign = directory.parents[2]
            old = verify_snapshot(old_campaign)
            assert old["formal"] == manifest["formal"]
            assert read_json(old_campaign / ("formal_" + method + ".json"))["validation"] == validation
            for key in ("actor_model", "actor_input_dim"):
                assert old["methods"][method].get(key, 18) == spec.get(key, 18)
            files = [name for name in old["source_hashes"] if name.startswith(("utils/", "algorithms/", "multiagent/"))]
            files += ["main.py", "experiments/run_passive_gsp_topology_pilot.py", "run_vector_signal_gsp_experiment.py",
                      "experiments/run_d4_actor_gpu_screen.py"]
            assert all(old["source_hashes"][name] == manifest["source_hashes"][name] for name in files)
        config = {key: value for key, value in read_json(directory / "resolved_config.json").items()
                  if key not in ignore}
        if reference_config is None:
            reference_config = config
        assert config == reference_config
        shapes = read_json(directory / "tensor_shape_report.json")
        width = 18 if method == "raw_mlp" else 21
        assert shapes["actor_augmented_tensor_shape"] == [64, width]
        assert shapes["target_actor_augmented_tensor_shape"] == [64, width]
        assert shapes["critic_input_tensor_shape"] == shapes["target_critic_input_tensor_shape"] == [64, 69]
        assert shapes["replay_obs_tensor_shapes"] == shapes["replay_next_obs_tensor_shapes"] == [[64, 18]] * 3
        summary = read_json(directory / "summary.json")
        rows = read_csv(directory / "per_evaluation_episode_metrics.csv")
        assert len(rows) == 500 and [int(row["test_seed"]) for row in rows] == list(range(1000000, 1000500))
        for key in METRICS:
            np.testing.assert_allclose(np.mean([float(row[key]) for row in rows]), summary[key], atol=1e-10, rtol=1e-10)
        episodes[method] = {key: np.array([float(row[key]) for row in rows]) for key in METRICS}
        curve = read_csv(directory / "learning_curve_eval.csv")
        assert [row["checkpoint"] for row in curve] == ["model_step" + str(s) for s in (20000, 40000, 60000, 80000, 100000)]
        assert all(int(row["eval_episodes"]) == 100 for row in curve)
        summaries[method] = summary
        audit[method] = dict(validation, actor_width=width, critic_width=69, replay_width=18)
        evidence = ("summary.json", "resolved_config.json", "per_evaluation_episode_metrics.csv",
                    "learning_curve_eval.csv", "tensor_shape_report.json", "gpu_device_audit.json")
        sources[method] = {"directory": str(directory), "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in evidence}}
        if not spec.get("reuse_from"):
            elapsed = (datetime.strptime(state["finished_at"], "%Y-%m-%dT%H:%M:%S%z") -
                       datetime.strptime(state["started_at"], "%Y-%m-%dT%H:%M:%S%z")).total_seconds()
            timings.append({"method": method, "worker_including_evaluation_sec": elapsed,
                            "training_sec": summary["train_wall_time_sec"], "finished_at": state["finished_at"]})
    elapsed = None
    if complete:
        assert len(timings) == planned_new and len(summaries) == len(manifest["methods"])
        start = datetime.strptime(read_json(campaign / "controller_started.json")["started_at"], "%Y-%m-%dT%H:%M:%S%z")
        end = max(datetime.strptime(row["finished_at"], "%Y-%m-%dT%H:%M:%S%z") for row in timings)
        elapsed = (end - start).total_seconds()
    comparisons, interpretations, screening = {}, {}, {}
    hypotheses = {
        "explore_constant_star3": "精确常数不含图信息。收益可能涉及输入参数化或优化轨迹；单seed不能区分具体原因。",
        "explore_hks_sigma075": "较窄权重可能削弱远处目标信息；尚未验证因果，不能据此否定全部边权方案。",
        "explore_hks_sigma125": "较宽权重可能压缩距离差异；需要特征变化与决策诊断验证。",
        "explore_hks_inverse": "反距离衰减较缓，可能平滑局部差异；未证明是性能变化原因。",
        "explore_hks_soft_radius": "软半径可能减少远处目标信息并改变度分布；原因待检验。",
        "explore_hks_knn2_union": "近邻筛选可能影响分配信息或连边稳定性；原因待检验。",
        "explore_hks_combinatorial": "同一数值t在组合L上对应不同扩散尺度；度尺度或focal读出压缩可能有影响，未证实因果。",
        "explore_hks_self_loop": "单位自环改变按度归一化和返回概率；小幅分配改善可能是单seed波动，机制尚未验证。",
        "explore_rwse_248": "偶数步返回概率不含地标索引，可能不足以表达目标分配；尚未证实因果。",
        "explore_lazy_rwse_248": "lazy walk混合不同步长并增加停留概率，可能平滑有用差异；原因待检验。",
        "explore_landmark_affinity3": "仅提供自身到各地标的三个高斯亲近度，没有谱信息。它不能直接编码其他agent占位，实际决策影响尚待验证。",
        "explore_wks3": "固定三个谱带可能压缩目标方向和地标索引信息；频带尺度或优化波动的影响尚未分离。",
        "explore_landmark_heat3": "地标对齐heat值混合全图扩散与目标亲近度，固定时间可能平滑分配差异；原因未证实。必须另对齐直接几何control。",
        "explore_regularized_resistance3": "normalized正则resolvent差异并非物理有效电阻，固定正则和缩放可能压缩差异；这只是候选原因，未验证因果。",
    }
    labels = {"raw_mlp": "raw", "explore_geometric_stats3": "几何", "explore_hks_6al": "默认6AL", "explore_constant_star3": "常数",
              "explore_landmark_affinity3": "地标对齐几何"}
    criteria = {"return": "return", "assignment distance": "分配距离", "coverage AUC": "AUC", "collision rate": "碰撞"}
    for method, summary in summaries.items():
        spec = manifest["methods"][method]
        if not spec["control"]:
            continue
        controls = list(dict.fromkeys([spec["control"], "raw_mlp"] + spec.get("additional_controls", [])))
        comparisons[method] = {name: gate(summary, summaries[name]) for name in controls if name in summaries}
        required = list(dict.fromkeys([spec["control"], "raw_mlp"] + spec.get("required_additional_controls", [])))
        if any(name not in comparisons[method] for name in required):
            raise RuntimeError("A completed candidate is missing a required control")
        screening[method] = {"required_controls": required,
                             "passed": all(comparisons[method][name]["passed"] for name in required)}
        if args.study == "descriptors" and not spec.get("reuse_from"):
            saved_gate = read_json(run_directory(campaign, "formal", method, spec) / "screening_gate.json")
            assert saved_gate["required_additional_controls"] == spec.get("required_additional_controls", [])
            assert saved_gate["matched_control"] == comparisons[method][spec["control"]]
            assert saved_gate["raw"] == comparisons[method]["raw_mlp"]
            assert saved_gate["additional_controls"] == {name: comparisons[method][name] for name in spec.get("additional_controls", [])}
        observed = "；".join("对" + labels.get(name, name) + ": " + "/".join(criteria[key] for key in result["failed_criteria"])
                            for name, result in comparisons[method].items() if not result["passed"])
        if method in hypotheses:
            interpretations[method] = {"observed_failure": observed, "possible_cause_not_proven": hypotheses[method]}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paired = []
    indices = np.random.default_rng(2026090604).integers(0, 500, size=(3000, 500))
    for method, controls in comparisons.items():
        if manifest["methods"][method].get("reuse_from"):
            continue
        for control in controls:
            for metric in METRICS:
                delta = episodes[method][metric] - episodes[control][metric]
                lo, hi = np.quantile(delta[indices].mean(1), [.025, .975])
                paired.append({"method": method, "control": control, "metric": metric,
                               "delta": float(delta.mean()), "episode_ci95_low": float(lo), "episode_ci95_high": float(hi)})
    write_json(output / "paired_deltas.json", {"comparisons": paired,
               "scope": "Conditional uncertainty over the 500 shared evaluation scenes, not training-seed uncertainty; no multiplicity adjustment. Screening bank, not independent test."})
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    result = {"passed": True, "study_complete": complete, "generated_at": generated_at,
              "new_completed": len(timings), "planned_new": planned_new, "study": args.study,
              "training_seed_count": 1, "elapsed_sec": elapsed,
              "queue_snapshot": queue, "validations": audit, "sources": sources,
              "all_final_evaluation_seeds_matched": True, "all_training_configs_matched_except_actor_identity_width": True}
    write_json(output / "audit.json", result)
    write_json(output / "summary.json", {"runs": summaries, "comparisons": comparisons, "timings": timings,
                                         "screening": screening, "study_complete": complete,
                                         "elapsed_sec": elapsed, "paper_claim_allowed": False})
    write_json(output / "interpretations.json", interpretations)
    with (output / "completed_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *METRICS, "final3", "new_run", "source"])
        writer.writeheader()
        for method, summary in summaries.items():
            writer.writerow(dict(method=method, **{key: summary[key] for key in (*METRICS, "final3")},
                                 new_run=not bool(manifest["methods"][method].get("reuse_from")), source=sources[method]["directory"]))
    lines = ["# " + title + "结果快照", "", "时间：" + generated_at,
             "新增正式运行完成：%s/%s。%s" % (len(timings), planned_new, "整组已完成。" if complete else "整组尚未完成，不提前给出组内赢家。"),
             "", "| 方法 | Return | Hungarian距离 | AUC | Collision rate | Final coverage |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, summary in summaries.items():
        lines.append("| " + method + " | " + " | ".join("%.5f" % summary[key] for key in METRICS) + " |")
    if complete:
        lines += ["", "全组含smoke、训练、评估耗时：%d分%d秒。" % divmod(int(elapsed), 60)]
    lines += ["", "综合主门槛同时比较raw和几何control。heat/resistance还必须通过地标对齐几何control；默认HKS和常数对照仍逐项报告，不凭raw提升宣称谱机制有效。" if args.study == "descriptors"
              else "综合主门槛同时比较raw和几何control，默认HKS和常数对照另外逐项报告。",
              "综合主门槛：" + ", ".join(name + "=" + str(item["passed"]) for name, item in screening.items())]
    lines += ["", "## 证据与限制", "", "已完成运行逐一核对100k计数、五个20k间隔checkpoint、每点100次和最终500次评估、原始观测数据流、CUDA更新、模型及源码哈希。复用比较没有回写旧目录。",
              "只有seed1，共用筛选场景。均值排序不证明训练稳定性，不把筛选集当独立测试集。配对bootstrap只反映冻结策略在评估场景上的不确定性，未作多重比较校正，不是training-seed区间。",
              "常数control没有图信息。其结果意味着不能仅用相对raw的提升证明GSP信息有用；也不意味着其他图方法必然无效。",
              "此前4ego只有数值舍入级特征变化。常数与4ego的训练结果差异尚不能归因于一个已验证机制。", "", "## 负结果与原因假说", ""]
    for method, item in interpretations.items():
        lines.append("- " + method + ": " + (item["observed_failure"] or "主比较没有退化项") + "。" + item["possible_cause_not_proven"])
    lines += ["", "final3：" + ", ".join(name + "=" + str(row["final3"]) for name, row in summaries.items()),
              "", "运行中方法不作结果推断，不修改其冻结源码或预算。" + next_action]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "study_complete": complete, "new_completed": len(timings), "output": str(output)}))


if __name__ == "__main__":
    main()
