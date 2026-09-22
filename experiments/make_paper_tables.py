import argparse
import shutil
import time
from pathlib import Path

import pandas as pd


ACTIVE_20K_REVIEW = Path("experiments/active_gsp_v3_residual_lowlr_20k_review_20260703_225610")
PASSIVE_100K_REVIEW = Path("experiments/formal100k_passive_combined_review_20260703_190622")
FASTPATH_PROFILE = Path("experiments/active_gsp_v3_profile_fastpath_20260703_020538")

METHOD_LABELS = {
    "raw_mlp": "MADDPG",
    "action_score_control": "Action scorer",
    "action_raw_potential_residual005": "Raw action prior",
    "v3_active_gsp_residual005": "Active GSP v3",
    "v1_full": "Passive GSP v1",
    "v2a_spectrum": "Passive spectrum",
    "none": "Raw rollout",
    "reference": "Reference v3",
    "fast": "Fast v3",
}


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_existing_dir(pattern):
    matches = [path for path in Path("experiments").glob(pattern) if path.is_dir()]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def label(name):
    return METHOD_LABELS.get(name, name.replace("_", "\\_"))


def fmt_mean_std(mean, std, digits=2):
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_num(value, digits=3):
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def write_latex_table(path, caption, label_name, columns, rows, align=None):
    if align is None:
        align = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label_name}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_csv(source, target_dir, prefix):
    if source.exists():
        target = target_dir / f"{prefix}_{source.parent.name}_{source.name}"
        shutil.copy2(source, target)
        return target
    return None


def active_summary_table(review_dir, output_dir, stem, caption, label_name):
    summary = pd.read_csv(review_dir / "summary.csv")
    order = ["raw_mlp", "action_raw_potential_residual005", "v3_active_gsp_residual005"]
    summary = summary[summary["experiment"].isin(order)].copy()
    summary["_rank"] = summary["experiment"].map({name: idx for idx, name in enumerate(order)})
    summary = summary.sort_values("_rank")
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                label(row["experiment"]),
                fmt_mean_std(row["return_mean"], row["return_std"]),
                fmt_mean_std(row["final_coverage_mean"], row["final_coverage_std"]),
                fmt_mean_std(row["max_coverage_mean"], row["max_coverage_std"]),
                fmt_mean_std(row["nearest_landmark_distance_mean"], row["nearest_landmark_distance_std"], 3),
                fmt_mean_std(row["collisions_mean"], row["collisions_std"], 3),
            ]
        )
    write_latex_table(
        output_dir / f"{stem}.tex",
        caption=caption,
        label_name=label_name,
        columns=["Method", "Return", "Final cov.", "Max cov.", "Dist.", "Coll."],
        rows=rows,
        align="lccccc",
    )
    summary.drop(columns=["_rank"]).to_csv(output_dir / f"{stem}.csv", index=False)


def active_delta_table(review_dir, output_dir, stem, caption, label_name):
    path = review_dir / "primary_vs_control_delta_means.csv"
    baseline_path = review_dir / "paired_delta_means_vs_baseline.csv"
    rows = []
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        v3 = baseline[baseline["experiment"].eq("v3_active_gsp_residual005")]
        if not v3.empty:
            row = v3.iloc[0]
            rows.append(
                [
                    "Active GSP v3 - MADDPG",
                    fmt_num(row["delta_return_vs_baseline"], 3),
                    fmt_num(row["delta_final_coverage_vs_baseline"], 3),
                    fmt_num(row["delta_max_coverage_vs_baseline"], 3),
                    fmt_num(row["delta_nearest_landmark_distance_vs_baseline"], 3),
                    fmt_num(row["delta_collisions_vs_baseline"], 3),
                ]
            )
    if path.exists():
        control = pd.read_csv(path).set_index("metric")["mean"]
        rows.append(
            [
                "Active GSP v3 - raw action prior",
                fmt_num(control.get("delta_return_v3_active_gsp_residual005_vs_action_raw_potential_residual005"), 3),
                fmt_num(control.get("delta_final_coverage_v3_active_gsp_residual005_vs_action_raw_potential_residual005"), 3),
                fmt_num(control.get("delta_max_coverage_v3_active_gsp_residual005_vs_action_raw_potential_residual005"), 3),
                fmt_num(control.get("delta_nearest_landmark_distance_v3_active_gsp_residual005_vs_action_raw_potential_residual005"), 3),
                fmt_num(control.get("delta_collisions_v3_active_gsp_residual005_vs_action_raw_potential_residual005"), 3),
            ]
        )
    write_latex_table(
        output_dir / f"{stem}.tex",
        caption=caption,
        label_name=label_name,
        columns=["Comparison", "$\\Delta$Return", "$\\Delta$Final", "$\\Delta$Max", "$\\Delta$Dist.", "$\\Delta$Coll."],
        rows=rows,
        align="lrrrrr",
    )
    pd.DataFrame(rows, columns=["comparison", "delta_return", "delta_final", "delta_max", "delta_distance", "delta_collisions"]).to_csv(
        output_dir / f"{stem}.csv", index=False
    )


def passive_table(output_dir):
    summary = pd.read_csv(PASSIVE_100K_REVIEW / "summary.csv")
    order = ["raw_mlp", "v1_full", "v2a_spectrum"]
    summary = summary[summary["experiment"].isin(order)].copy()
    summary["_rank"] = summary["experiment"].map({name: idx for idx, name in enumerate(order)})
    summary = summary.sort_values("_rank")
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                label(row["experiment"]),
                fmt_mean_std(row["return_mean"], row["return_std"]),
                fmt_mean_std(row["final_coverage_mean"], row["final_coverage_std"]),
                fmt_mean_std(row["max_coverage_mean"], row["max_coverage_std"]),
                fmt_mean_std(row["nearest_landmark_distance_mean"], row["nearest_landmark_distance_std"], 3),
            ]
        )
    write_latex_table(
        output_dir / "table_passive_gsp_100k_ablation.tex",
        caption="Passive GSP ablations at 100k episodes. Passive descriptors are retained as ablations rather than the main method.",
        label_name="tab:passive-gsp-ablation",
        columns=["Method", "Return", "Final cov.", "Max cov.", "Dist."],
        rows=rows,
        align="lcccc",
    )
    summary.drop(columns=["_rank"]).to_csv(output_dir / "table_passive_gsp_100k_ablation.csv", index=False)


def fastpath_table(output_dir):
    profile = pd.read_csv(FASTPATH_PROFILE / "profile.csv")
    latency = pd.read_csv(FASTPATH_PROFILE / "standalone_latency.csv").iloc[0]
    raw = profile[profile["feature_mode"].eq("none")].iloc[0]
    fast = profile[profile["feature_mode"].eq("fast")].iloc[0]
    reference = profile[profile["feature_mode"].eq("reference")].iloc[0]
    rows = [
        [
            "Rollout throughput",
            fmt_num(raw["env_steps_per_sec"], 1),
            fmt_num(reference["env_steps_per_sec"], 1),
            fmt_num(fast["env_steps_per_sec"], 1),
            fmt_num(fast["env_steps_per_sec"] / raw["env_steps_per_sec"], 3),
        ],
        [
            "Feature latency (ms/obs)",
            "--",
            fmt_num(latency["reference_ms_per_obs"], 4),
            fmt_num(latency["fast_ms_per_obs"], 4),
            f"{latency['standalone_speedup']:.1f}x",
        ],
    ]
    write_latex_table(
        output_dir / "table_active_gsp_v3_fastpath_profile.tex",
        caption="Active GSP v3 fast-path profiling. The vectorized local-observation-only implementation passes the training throughput gate.",
        label_name="tab:active-gsp-fastpath",
        columns=["Metric", "Raw", "Reference", "Fast path", "Ratio/speedup"],
        rows=rows,
        align="lrrrr",
    )
    profile.to_csv(output_dir / "table_active_gsp_v3_fastpath_profile.csv", index=False)


def final_eval_tables(final_eval_dir, output_dir):
    final_eval_dir = Path(final_eval_dir)
    summary_path = final_eval_dir / "summary_across_train_seeds.csv"
    delta_path = final_eval_dir / "paired_delta_summary.csv"
    if not summary_path.exists() or not delta_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    order = ["raw_mlp", "action_raw_potential_residual005", "v3_active_gsp_residual005"]
    summary = summary[summary["experiment"].isin(order)].copy()
    summary["_rank"] = summary["experiment"].map({name: idx for idx, name in enumerate(order)})
    summary = summary.sort_values("_rank")
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                label(row["experiment"]),
                fmt_mean_std(row["return_mean_mean"], row["return_mean_std"]),
                fmt_mean_std(row["final_coverage_mean_mean"], row["final_coverage_mean_std"]),
                fmt_mean_std(row["max_coverage_mean_mean"], row["max_coverage_mean_std"]),
                fmt_mean_std(row["nearest_landmark_distance_mean_mean"], row["nearest_landmark_distance_mean_std"], 3),
                fmt_mean_std(row["collisions_mean_mean"], row["collisions_mean_std"], 3),
            ]
        )
    write_latex_table(
        output_dir / "table_final_eval_summary.tex",
        caption="Final paired evaluation with fixed test seeds.",
        label_name="tab:final-eval-summary",
        columns=["Method", "Return", "Final cov.", "Max cov.", "Dist.", "Coll."],
        rows=rows,
        align="lccccc",
    )
    summary.drop(columns=["_rank"]).to_csv(output_dir / "table_final_eval_summary.csv", index=False)

    delta = pd.read_csv(delta_path)
    keep = delta[
        delta["left"].eq("v3_active_gsp_residual005")
        & delta["right"].isin(["raw_mlp", "action_raw_potential_residual005"])
    ].copy()
    rows = []
    for left, right, comparison in [
        ("v3_active_gsp_residual005", "raw_mlp", "Active GSP v3 - MADDPG"),
        ("v3_active_gsp_residual005", "action_raw_potential_residual005", "Active GSP v3 - raw action prior"),
    ]:
        subset = keep[keep["left"].eq(left) & keep["right"].eq(right)].set_index("metric")
        rows.append(
            [
                comparison,
                metric_ci(subset, "return"),
                metric_ci(subset, "final_coverage"),
                metric_ci(subset, "max_coverage"),
                metric_ci(subset, "nearest_landmark_distance"),
                metric_ci(subset, "collisions"),
            ]
        )
    write_latex_table(
        output_dir / "table_final_eval_paired_deltas.tex",
        caption="Final paired deltas with bootstrap confidence intervals over training seeds.",
        label_name="tab:final-eval-deltas",
        columns=["Comparison", "$\\Delta$Return", "$\\Delta$Final", "$\\Delta$Max", "$\\Delta$Dist.", "$\\Delta$Coll."],
        rows=rows,
        align="lccccc",
    )
    keep.to_csv(output_dir / "table_final_eval_paired_deltas.csv", index=False)
    return ["table_final_eval_summary.tex", "table_final_eval_paired_deltas.tex"]


def metric_ci(frame, metric):
    if metric not in frame.index:
        return "--"
    row = frame.loc[metric]
    return f"{row['mean_delta']:.3f} [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]"


def write_readme(output_dir, active_formal_review=None, final_eval_dir=None):
    lines = [
        "# Paper Table Bundle",
        "",
        f"Created: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "LaTeX tables use plain tabular/hline formatting so they can be pasted into the ICASSP template without extra packages.",
        "",
        "## Tables",
        "",
    ]
    for path in sorted(output_dir.glob("table_*.tex")):
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- active 20k: `{ACTIVE_20K_REVIEW}`",
            f"- passive 100k: `{PASSIVE_100K_REVIEW}`",
            f"- fast path: `{FASTPATH_PROFILE}`",
        ]
    )
    if active_formal_review:
        lines.append(f"- active formal review: `{active_formal_review}`")
    if final_eval_dir:
        lines.append(f"- final evaluation: `{final_eval_dir}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--active-formal-review", default=None)
    parser.add_argument("--final-eval-dir", default=None)
    parser.add_argument("--auto-latest-final", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    active_formal_review = Path(args.active_formal_review) if args.active_formal_review else None
    final_eval_dir = Path(args.final_eval_dir) if args.final_eval_dir else None
    if args.auto_latest_final:
        active_formal_review = active_formal_review or latest_existing_dir("active_gsp_v3_residual_lowlr_100k_formal_review_*")
        final_eval_dir = final_eval_dir or latest_existing_dir("active_gsp_v3_final_eval_*")
    kind = "final" if active_formal_review or final_eval_dir else "prelim"
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path("experiments") / f"paper_tables_{kind}_{time.strftime('%Y%m%d_%H%M%S')}")

    active_summary_table(
        ACTIVE_20K_REVIEW,
        output_dir,
        "table_active_gsp_v3_20k_summary",
        "Active GSP v3 20k screening results.",
        "tab:active-gsp-20k",
    )
    active_delta_table(
        ACTIVE_20K_REVIEW,
        output_dir,
        "table_active_gsp_v3_20k_paired_deltas",
        "Paired 20k deltas for Active GSP v3.",
        "tab:active-gsp-20k-deltas",
    )
    if active_formal_review is not None:
        active_summary_table(
            active_formal_review,
            output_dir,
            "table_active_gsp_v3_100k_formal_summary",
            "Active GSP v3 100k formal validation results.",
            "tab:active-gsp-100k",
        )
        active_delta_table(
            active_formal_review,
            output_dir,
            "table_active_gsp_v3_100k_formal_paired_deltas",
            "Paired 100k formal deltas for Active GSP v3.",
            "tab:active-gsp-100k-deltas",
        )
    passive_table(output_dir)
    fastpath_table(output_dir)
    if final_eval_dir is not None:
        final_eval_tables(final_eval_dir, output_dir)
    for source in [
        ACTIVE_20K_REVIEW / "summary.csv",
        ACTIVE_20K_REVIEW / "paired_delta_means_vs_baseline.csv",
        PASSIVE_100K_REVIEW / "summary.csv",
        FASTPATH_PROFILE / "profile.csv",
    ]:
        copy_csv(source, output_dir, "source")
    write_readme(output_dir, active_formal_review=active_formal_review, final_eval_dir=final_eval_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
