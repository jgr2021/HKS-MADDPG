import argparse
import time
from pathlib import Path

import pandas as pd


ACTIVE_20K_REVIEW = Path("experiments/active_gsp_v3_residual_lowlr_20k_review_20260703_225610")

METHOD_LABELS = {
    "raw_mlp": "MADDPG",
    "action_raw_potential_residual005": "Raw action prior",
    "v3_active_gsp_residual005": "Active GSP v3",
}


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_existing_dir(pattern):
    matches = [path for path in Path("experiments").glob(pattern) if path.is_dir()]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def fmt_ci(mean, low, high, digits=3):
    return f"{fmt(mean, digits)} [{fmt(low, digits)}, {fmt(high, digits)}]"


def method_row(summary, experiment):
    rows = summary[summary["experiment"].eq(experiment)]
    if rows.empty:
        raise KeyError(f"missing experiment in summary: {experiment}")
    return rows.iloc[0]


def delta_row(delta_summary, metric, left, right):
    rows = delta_summary[
        delta_summary["metric"].eq(metric)
        & delta_summary["left"].eq(left)
        & delta_summary["right"].eq(right)
    ]
    if rows.empty:
        raise KeyError(f"missing delta: {metric} {left} vs {right}")
    return rows.iloc[0]


def macro_line(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_prelim_snippets(output_dir, active_review):
    summary = pd.read_csv(active_review / "summary.csv")
    deltas = pd.read_csv(active_review / "paired_delta_means_vs_baseline.csv")
    control = pd.read_csv(active_review / "primary_vs_control_delta_means.csv").set_index("metric")["mean"]

    raw = method_row(summary, "raw_mlp")
    active = method_row(summary, "v3_active_gsp_residual005")
    raw_prior = method_row(summary, "action_raw_potential_residual005")
    active_vs_raw = deltas[deltas["experiment"].eq("v3_active_gsp_residual005")].iloc[0]

    macros = [
        "% Preliminary 20k screening macros. Replace with final macros after 100k final evaluation.",
        macro_line("PrelimRawReturn", fmt(raw["return_mean"], 2)),
        macro_line("PrelimRawFinalCoverage", fmt(raw["final_coverage_mean"], 2)),
        macro_line("PrelimRawMaxCoverage", fmt(raw["max_coverage_mean"], 2)),
        macro_line("PrelimRawDistance", fmt(raw["nearest_landmark_distance_mean"], 3)),
        macro_line("PrelimActiveReturn", fmt(active["return_mean"], 2)),
        macro_line("PrelimActiveFinalCoverage", fmt(active["final_coverage_mean"], 2)),
        macro_line("PrelimActiveMaxCoverage", fmt(active["max_coverage_mean"], 2)),
        macro_line("PrelimActiveDistance", fmt(active["nearest_landmark_distance_mean"], 3)),
        macro_line("PrelimActiveDeltaReturnVsRaw", fmt(active_vs_raw["delta_return_vs_baseline"], 3)),
        macro_line("PrelimActiveDeltaFinalVsRaw", fmt(active_vs_raw["delta_final_coverage_vs_baseline"], 3)),
        macro_line("PrelimActiveDeltaMaxVsRaw", fmt(active_vs_raw["delta_max_coverage_vs_baseline"], 3)),
        macro_line("PrelimActiveDeltaDistanceVsRaw", fmt(active_vs_raw["delta_nearest_landmark_distance_vs_baseline"], 3)),
        macro_line(
            "PrelimActiveDeltaReturnVsRawPrior",
            fmt(control["delta_return_v3_active_gsp_residual005_vs_action_raw_potential_residual005"], 3),
        ),
        macro_line(
            "PrelimActiveDeltaFinalVsRawPrior",
            fmt(control["delta_final_coverage_v3_active_gsp_residual005_vs_action_raw_potential_residual005"], 3),
        ),
    ]
    (output_dir / "paper_result_macros_prelim.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    paragraph = f"""# Preliminary Result Snippets

Use these snippets only before the formal 100k final evaluation is available.

At 20k episodes over three training seeds, Active GSP v3 achieved the best mean return ({fmt(active['return_mean'], 2)}), final coverage ({fmt(active['final_coverage_mean'], 2)}), max coverage ({fmt(active['max_coverage_mean'], 2)}), and nearest-landmark distance ({fmt(active['nearest_landmark_distance_mean'], 3)}). The MADDPG baseline obtained return {fmt(raw['return_mean'], 2)}, final coverage {fmt(raw['final_coverage_mean'], 2)}, max coverage {fmt(raw['max_coverage_mean'], 2)}, and distance {fmt(raw['nearest_landmark_distance_mean'], 3)}. Active GSP v3 improved paired return by {fmt(active_vs_raw['delta_return_vs_baseline'], 3)}, final coverage by {fmt(active_vs_raw['delta_final_coverage_vs_baseline'], 3)}, max coverage by {fmt(active_vs_raw['delta_max_coverage_vs_baseline'], 3)}, and nearest-landmark distance by {fmt(active_vs_raw['delta_nearest_landmark_distance_vs_baseline'], 3)} relative to MADDPG.

Compared with the raw action prior, Active GSP v3 improved return by {fmt(control['delta_return_v3_active_gsp_residual005_vs_action_raw_potential_residual005'], 3)}, final coverage by {fmt(control['delta_final_coverage_v3_active_gsp_residual005_vs_action_raw_potential_residual005'], 3)}, max coverage by {fmt(control['delta_max_coverage_v3_active_gsp_residual005_vs_action_raw_potential_residual005'], 3)}, and nearest-landmark distance by {fmt(control['delta_nearest_landmark_distance_v3_active_gsp_residual005_vs_action_raw_potential_residual005'], 3)}.

Raw action prior mean return for context: {fmt(raw_prior['return_mean'], 2)}.
"""
    (output_dir / "paper_result_snippets_prelim.md").write_text(paragraph, encoding="utf-8")


def final_summary_row(summary, experiment):
    rows = summary[summary["experiment"].eq(experiment)]
    if rows.empty:
        raise KeyError(f"missing final summary experiment: {experiment}")
    return rows.iloc[0]


def write_final_snippets(output_dir, final_eval_dir, formal_review=None):
    summary = pd.read_csv(final_eval_dir / "summary_across_train_seeds.csv")
    delta_summary = pd.read_csv(final_eval_dir / "paired_delta_summary.csv")

    raw = final_summary_row(summary, "raw_mlp")
    active = final_summary_row(summary, "v3_active_gsp_residual005")
    prior = final_summary_row(summary, "action_raw_potential_residual005")

    metrics = ["return", "final_coverage", "max_coverage", "nearest_landmark_distance", "collisions"]
    labels = {
        "return": "Return",
        "final_coverage": "FinalCoverage",
        "max_coverage": "MaxCoverage",
        "nearest_landmark_distance": "Distance",
        "collisions": "Collisions",
    }
    methods = {
        "Raw": raw,
        "Active": active,
        "RawPrior": prior,
    }
    macros = ["% Final evaluation macros generated from fixed-test-seed paired evaluation."]
    for method_name, row in methods.items():
        for metric in metrics:
            prefix = f"{metric}_mean"
            macros.append(macro_line(f"Final{method_name}{labels[metric]}", fmt(row[f"{prefix}_mean"], 3)))
            macros.append(macro_line(f"Final{method_name}{labels[metric]}Std", fmt(row[f"{prefix}_std"], 3)))

    for right, right_label in [("raw_mlp", "VsRaw"), ("action_raw_potential_residual005", "VsRawPrior")]:
        for metric in metrics:
            row = delta_row(delta_summary, metric, "v3_active_gsp_residual005", right)
            macros.append(macro_line(f"FinalActiveDelta{labels[metric]}{right_label}", fmt(row["mean_delta"], 3)))
            macros.append(
                macro_line(
                    f"FinalActiveDelta{labels[metric]}{right_label}CI",
                    fmt_ci(row["mean_delta"], row["bootstrap_ci_low"], row["bootstrap_ci_high"], 3),
                )
            )
    (output_dir / "paper_result_macros_final.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    def ci(metric, right):
        row = delta_row(delta_summary, metric, "v3_active_gsp_residual005", right)
        return fmt_ci(row["mean_delta"], row["bootstrap_ci_low"], row["bootstrap_ci_high"], 3)

    paragraph = f"""# Final Result Snippets

Source final evaluation:

`{final_eval_dir}`

In the fixed-test-seed final evaluation, Active GSP v3 achieved return {fmt(active['return_mean_mean'], 3)}, final coverage {fmt(active['final_coverage_mean_mean'], 3)}, max coverage {fmt(active['max_coverage_mean_mean'], 3)}, nearest-landmark distance {fmt(active['nearest_landmark_distance_mean_mean'], 3)}, and collisions {fmt(active['collisions_mean_mean'], 3)}. MADDPG achieved return {fmt(raw['return_mean_mean'], 3)}, final coverage {fmt(raw['final_coverage_mean_mean'], 3)}, max coverage {fmt(raw['max_coverage_mean_mean'], 3)}, nearest-landmark distance {fmt(raw['nearest_landmark_distance_mean_mean'], 3)}, and collisions {fmt(raw['collisions_mean_mean'], 3)}.

Relative to MADDPG, Active GSP v3 changed return by {ci('return', 'raw_mlp')}, final coverage by {ci('final_coverage', 'raw_mlp')}, max coverage by {ci('max_coverage', 'raw_mlp')}, nearest-landmark distance by {ci('nearest_landmark_distance', 'raw_mlp')}, and collisions by {ci('collisions', 'raw_mlp')}. Negative values are improvements for distance and collisions.

Relative to the raw action prior, Active GSP v3 changed return by {ci('return', 'action_raw_potential_residual005')}, final coverage by {ci('final_coverage', 'action_raw_potential_residual005')}, max coverage by {ci('max_coverage', 'action_raw_potential_residual005')}, nearest-landmark distance by {ci('nearest_landmark_distance', 'action_raw_potential_residual005')}, and collisions by {ci('collisions', 'action_raw_potential_residual005')}.
"""
    if formal_review is not None:
        paragraph += f"\nFormal training review:\n\n`{formal_review}`\n"
    (output_dir / "paper_result_snippets_final.md").write_text(paragraph, encoding="utf-8")


def write_readme(output_dir, final_eval_dir=None, formal_review=None):
    lines = [
        "# Paper Result Snippets",
        "",
        f"Created: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "This bundle contains copy-ready result paragraphs and LaTeX numeric macros.",
        "",
        "## Files",
        "",
    ]
    for path in sorted(output_dir.glob("*")):
        if path.name != "README.md":
            lines.append(f"- `{path.name}`")
    lines.extend(["", "## Sources", "", f"- active 20k: `{ACTIVE_20K_REVIEW}`"])
    if formal_review is not None:
        lines.append(f"- formal review: `{formal_review}`")
    if final_eval_dir is not None:
        lines.append(f"- final evaluation: `{final_eval_dir}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formal-review", default=None)
    parser.add_argument("--final-eval-dir", default=None)
    parser.add_argument("--auto-latest-final", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    formal_review = Path(args.formal_review) if args.formal_review else None
    final_eval_dir = Path(args.final_eval_dir) if args.final_eval_dir else None
    if args.auto_latest_final:
        formal_review = formal_review or latest_existing_dir("active_gsp_v3_residual_lowlr_100k_formal_review_*")
        final_eval_dir = final_eval_dir or latest_existing_dir("active_gsp_v3_final_eval_*")
    if final_eval_dir is not None and not (final_eval_dir / "summary_across_train_seeds.csv").exists():
        raise FileNotFoundError(final_eval_dir / "summary_across_train_seeds.csv")
    kind = "final" if final_eval_dir is not None else "prelim"
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path("experiments") / f"paper_snippets_{kind}_{time.strftime('%Y%m%d_%H%M%S')}")
    write_prelim_snippets(output_dir, ACTIVE_20K_REVIEW)
    if final_eval_dir is not None:
        write_final_snippets(output_dir, final_eval_dir, formal_review=formal_review)
    write_readme(output_dir, final_eval_dir=final_eval_dir, formal_review=formal_review)
    print(output_dir)


if __name__ == "__main__":
    main()
