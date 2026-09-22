"""Read-only collection of experiment evidence for the exploration workbook."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate", "final_coverage")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def record(source, cohort, data, seeds=None, count=1, suffix="", note="", cause=""):
    row = {"cohort": cohort, "method": data["method"], "seeds": str(seeds or data.get("seed", data.get("train_seed", ""))),
           "n_training_seeds": count, "budget_env_steps": number(data.get("global_env_steps", 100000)),
           "status": data.get("status", "completed"), "observed": note, "hypothesis_only": cause,
           "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    for key in METRICS:
        row[key] = number(data.get(key + suffix, data.get("final_coverage_r010") if key == "final_coverage" else None))
    return row


def collect(campaign):
    history = []
    passive = ROOT / "experiments/passive_gsp_topology_100k_seed1_corrected/run_20260710_193419"
    for method in ("raw_mlp", "gsp_3node_aa_hks", "gsp_6node_al_hks", "gsp_6node_aal_hks"):
        source = passive / method / "seed_1/summary.json"
        history.append(record(source, "passive_corrected_seed1", read(source),
                              note="Historical actor-only 100k. HU/AUC absent in this original summary; not zero.",
                              cause="Topology may change target/competition information. One seed cannot establish robustness."))
    d4 = ROOT / "experiments/d4_actor_gpu_100k_seed1_20260905"
    for method in ("d4_geometric_control", "d4_dirichlet_gsp"):
        source = d4 / method / "seed_1/summary.json"
        history.append(record(source, "d4_cuda_seed1", read(source),
                              note="Energy vs geometry: return/HU/AUC better, collision +0.00576. Original gate failed.",
                              cause="Possible collision-energy/action-ranking mismatch; feature masks are diagnostics, not retrained policies."))
    source = ROOT / "experiments/corrected_active_gsp_multiseed_20260711_124239/combined_seed_summaries.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for data in csv.DictReader(handle):
            history.append(record(source, "active_corrected_seeds123", data,
                                  note="Earlier 3-seed cohort. Do not pool with later training/evaluation protocols.",
                                  cause="Coverage/collision tradeoff; later cohorts are necessary to assess robustness."))
    source = ROOT / "experiments/multiscale_spectral_screening_eval_20260711/seed_metrics.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for data in csv.DictReader(handle):
            history.append(record(source, "multiscale_seeds11_13", data,
                                  note="Old active and multiscale failed HU/AUC paired screening against geometry; 500-episode final re-evaluation.",
                                  cause="Possible unhelpful scale mixing or feature magnitudes. Causality not established."))
    source = ROOT / "experiments/vector_signal_gsp_locked_screen/run_20260711_231009/aggregate_summary.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for data in csv.DictReader(handle):
            history.append(record(source, "vector_seeds21_23_aggregate", data, seeds="21,22,23", count=3, suffix="_mean",
                                  note="Three-training-seed mean, not an individual run. Vector GSP loses HU/AUC/return to vector geometry.",
                                  cause="More signal channels did not ensure useful action decisions; scaling/readout remain hypotheses."))
    catalog = []
    source = ROOT / "docs/maddpg_gsp_design_catalog_20260906.md"
    for line in source.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\| ([GEWOXDAR]\d+) \| (.*) \|$", line)
        if match:
            parts = match[2].split(" | ")
            catalog.append({"id": match[1], "design": parts[0], "rationale": parts[1],
                            "limitations_and_history": " | ".join(parts[2:]),
                            "state": "设计选项，非独立已完成实验", "source": str(source)})
    if len(catalog) != 76 or len({row["id"] for row in catalog}) != 76:
        raise RuntimeError("Design catalogue count changed; review parser and evidence before export")
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "campaign": str(campaign), "current": read(campaign / "exploration_master.json"),
            "history": history, "catalog": catalog,
            "plan": str(ROOT / "docs/gsp_sequential_exploration_plan_20260906.md"),
            "notes": [
                ["正式训练", "seed1，100k 环境交互步；每20k checkpoint；中间100、最终500 evaluation episodes。"],
                ["观测边界", "环境和replay为raw18D，critic/target critic 69D；仅actor按自身local obs构图。"],
                ["当前阶段", "raw、接触修正几何control、接触修正Dirichlet；每种先独立1k smoke，再fresh100k。"],
                ["后续阶段", "拓扑、边权、算子、描述符分组单因素筛选；实现/验证后另建冻结副本，最多20个新seed1正式运行。"],
                ["判定", "对matched control和raw同时比较：return上升、Hungarian距离下降、coverage-radius AUC上升、collision不增加。"],
                ["缺失值", "空白表示未运行或该源未报告，不代表0。历史组之间不得混排为同一受控实验。"],
                ["失败原因", "观测事实与可能原因分列。除非有专门消融验证，原因只能是待检验假说。"],
                ["统计限制", "单seed筛选不等于稳健改进；episode区间不代表训练seed方差；同bank多配置筛选有选择偏差。"],
                ["ICASSP", "有望的方法还需冻结方案、新训练/评估seeds、强对照、机制证据与合理新意，不保证发表。"],
                ["表格更新", "Excel是生成时刻快照；运行状态CSV/JSON由控制器更新，自动跟进重新汇总Excel。"],
                ["历史覆盖", "本版收录可直接核对的corrected100k核心结果；RWSE/SCF/门控20k、性能和mask审计还需补入明细，不能称全部已归档。"],
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    payload = collect(options.campaign.resolve())
    topology = ROOT / "experiments/gsp_exploration_20260906_phase2a/exploration_master.json"
    if topology.exists():
        payload["topology"] = read(topology)
        payload["notes"][2][1] = "接触修正组已完成，能量版因碰撞增加未通过综合门槛；拓扑组6个新方法依次smoke和100k，raw基线复用。"
        interpretation = topology.parent / "analysis_20260906/interpretations.json"
        if interpretation.exists():
            updates = read(interpretation)
            for row in payload["topology"]:
                if row["method"] in updates:
                    row.update(updates[row["method"]])
    weights = ROOT / "experiments/gsp_exploration_20260906_phase2b/exploration_master.json"
    if weights.exists():
        payload["weights"] = read(weights)
        updates = {}
        topology_interpretations = topology.parent / "analysis_20260906/interpretations.json"
        if topology_interpretations.exists():
            updates.update(read(topology_interpretations))
        for path in sorted(weights.parent.glob("analysis_*/interpretations.json")):
            if read(path.parent / "audit.json").get("passed"):
                updates.update(read(path))
        for row in payload["weights"]:
            if row["status"] == "completed" and row["method"] in updates:
                row.update(updates[row["method"]])
        payload["notes"][2][1] = "拓扑组已完成，含评估总耗时45分33秒；6all相对几何control四项均好，但对raw碰撞更高。当前为边权/常数对照组。"
        payload["notes"].append(["星形HKS审计", "4ego中心normalized HKS=(1+exp(-2t))/2，在4500条观测上仅舍入波动。2B用精确常数control替代二值图训练，不增加预算。"])
        constant = next((row for row in payload["weights"] if row["method"] == "explore_constant_star3"), None)
        if constant and constant["status"] == "completed":
            payload["notes"][-1][1] = "4ego中心HKS在4500条观测上仅舍入波动。精确常数control的100k结果对几何control四项均好，但碰撞仍高于raw。不能仅用相对raw的提升证明谱信息有效；只有seed1。"
    operators = ROOT / "experiments/gsp_exploration_20260906_phase2c/exploration_master.json"
    if operators.exists():
        payload["operators"] = read(operators)
        inherited = {row["method"]: row for row in payload.get("weights", [])}
        for row in payload["operators"]:
            if row.get("reused") and row["method"] in inherited:
                for key in ("observed_failure", "possible_cause_not_proven"):
                    row[key] = inherited[row["method"]][key]
        updates = {}
        for path in sorted(operators.parent.glob("analysis_*/interpretations.json")):
            audit = read(path.parent / "audit.json")
            if audit.get("passed") and audit.get("study") == "operators":
                updates.update(read(path))
        for row in payload["operators"]:
            if row["status"] == "completed" and row["method"] in updates:
                row.update(updates[row["method"]])
        payload["notes"][2][1] = "边权组已完成，完整结果见边权筛选；当前固定默认6AL图比较四种算子，四个基线复用。2A至2C共预留16个新正式run，后续至多4个。"
    descriptors = ROOT / "experiments/gsp_exploration_20260906_phase2d/exploration_master.json"
    if descriptors.exists():
        payload["descriptors"] = read(descriptors)
        inherited = {row["method"]: row for row in payload.get("operators", [])}
        methods = read(descriptors.parent / "manifest.json")["methods"]
        updates = {}
        for path in sorted(descriptors.parent.glob("analysis_*/interpretations.json")):
            audit = read(path.parent / "audit.json")
            if audit.get("passed") and audit.get("study") == "descriptors":
                updates.update(read(path))
        for row in payload["descriptors"]:
            if row.get("reused") and row["method"] in inherited:
                for key in ("observed_failure", "possible_cause_not_proven"):
                    row[key] = inherited[row["method"]][key]
            if row["status"] == "completed" and row["method"] in updates:
                row.update(updates[row["method"]])
            row["required_additional_controls"] = methods[row["method"]].get("required_additional_controls", [])
        complete = read(descriptors.parent / "queue_status.json")["status"] == "study_completed"
        payload["notes"][2][1] = ("描述符组已完成，等待完整审计与本批总结，不再自动训练。" if complete else
                                  "算子组4项完成，耗时35分56秒，均未通过综合门槛。最后描述符组4项依次smoke和fresh100k，四个旧对照复用。")
        payload["notes"][3][1] = "2A6+2B6+2C4+2D4，共20个新seed1正式run额度全部分配；最后组结束后审计、总结并暂停自动跟进，新增实验或seed需用户决定。"
        payload["notes"].append(["描述符边界", "固定6AL和normalized L，三维WKS/地标heat/正则resolvent对比地标affinity。heat/resolvent还必须通过地标对齐几何control。resolvent不是物理有效电阻；有限值预检不是策略收益证据。"])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: len(payload[key]) for key in ("current", "history", "catalog")}))


if __name__ == "__main__":
    main()
