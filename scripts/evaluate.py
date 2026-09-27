"""Evaluate one checkpoint, or select a run's best checkpoint by curve-bank return."""
import argparse
import csv
import json
import math
from pathlib import Path

from _common import FINAL_SEED, load_runtime, positive_int, summarize


def select_checkpoint(run_dir):
    with (run_dir / "learning_curve_eval.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not all(math.isfinite(float(row["return"])) for row in rows):
        raise ValueError("learning curve must contain finite mean returns")
    # CSV checkpoint order is chronological; ties select the earlier checkpoint.
    best = max(rows, key=lambda row: float(row["return"]))
    checkpoint = run_dir / "checkpoints" / (best["checkpoint"] + ".pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    seed = int(best["seed"])
    curve_start = protocol["curve_eval_seed_base"] + seed * 10000
    curve_count = protocol["curve_eval_episodes"]
    return checkpoint, best, (curve_start, curve_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-dir", type=Path, help="select highest mean curve-bank return")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=positive_int, default=500)
    parser.add_argument("--seed-start", type=int, default=FINAL_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    output = options.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    metadata = {"selection": "provided_checkpoint"}
    if options.run_dir:
        checkpoint, selected, (curve_start, curve_count) = select_checkpoint(options.run_dir.resolve())
        if max(curve_start, options.seed_start) < min(curve_start + curve_count,
                                                    options.seed_start + options.episodes):
            parser.error("selection and final evaluation banks overlap")
        metadata.update(selection="highest_mean_curve_return", method=selected["method"],
                        seed=int(selected["seed"]), curve_mean_return=float(selected["return"]),
                        curve_seed_start=curve_start, curve_episodes=curve_count)
    else:
        checkpoint = options.checkpoint.resolve()
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
    protocol, evaluation = load_runtime(options.device)
    import torch
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    rows = evaluation.evaluate_model("simple_spread", checkpoint, options.episodes, 25, options.seed_start)
    output.mkdir(parents=True)
    protocol.write_csv(output / "episodes.csv", rows)
    metadata.update(checkpoint=str(checkpoint), episode_length=25,
                    seed_start=options.seed_start, device=options.device, **summarize(rows))
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
