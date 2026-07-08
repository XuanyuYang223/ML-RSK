#!/usr/bin/env python3
"""Run shape-count feature ablations and record logs by configuration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


FEATURE_SETS = ["rows", "rows_cols", "rows_cols_hooks"]
TASKS = ["syt", "ssyt"]


FINAL_PATTERN = re.compile(
    r"test_log_mae=(?P<mae>[0-9.]+).*?"
    r"test_log_rmse=(?P<rmse>[0-9.]+).*?"
    r"median_multiplicative_error=(?P<median>[0-9.]+)x.*?"
    r"mean_relative_error=(?P<mean_rel>[0-9.]+)",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/shape_count_features_n30_e120"))
    parser.add_argument("--python", type=Path, default=Path("/home/yangx/miniforge3/envs/sage/bin/python"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--alphabet-size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, task: str, feature_set: str, run_dir: Path) -> list[str]:
    return [
        str(args.python),
        "-u",
        "shape_counts/train_shape_counts_mlp.py",
        "--task",
        task,
        "--n",
        str(args.n),
        "--alphabet-size",
        str(args.alphabet_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--feature-set",
        feature_set,
        "--device",
        args.device,
        "--model-out",
        str(run_dir / "model.pt"),
    ]


def parse_final_metrics(log_text: str) -> dict[str, float]:
    final_text = log_text.rsplit("final:", maxsplit=1)[-1]
    match = FINAL_PATTERN.search(final_text)
    if match is None:
        return {}
    return {key: float(value) for key, value in match.groupdict().items()}


def run_command(command: list[str], run_dir: Path) -> str:
    lines = []
    with (run_dir / "train.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("subprocess stdout was not captured")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            lines.append(line)
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)
    return "".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for task in args.tasks:
        for feature_set in args.feature_sets:
            name = f"{task}_{feature_set}"
            run_dir = args.out_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(args, task, feature_set, run_dir)
            config = {
                "task": task,
                "feature_set": feature_set,
                "n": args.n,
                "alphabet_size": args.alphabet_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "layers": args.layers,
                "device": args.device,
                "model_out": str(run_dir / "model.pt"),
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
            print(f"prepared {run_dir}", flush=True)

            if args.run:
                log_text = run_command(command, run_dir)
                metrics = parse_final_metrics(log_text)
                rows.append({**config, **metrics})

    if rows:
        summary_lines = [
            "# Shape Count Feature Ablation",
            "",
            "Feature sets:",
            "",
            "- `rows`: padded normalized row lengths.",
            "- `rows_cols`: rows plus conjugate partition column lengths.",
            "- `rows_cols_hooks`: rows, columns, and hook-length summary statistics. This is formula-inspired.",
            "",
            "| task | feature set | test log MAE | test log RMSE | median mult err | mean rel err |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            summary_lines.append(
                f"| {row['task']} | {row['feature_set']} | "
                f"{row.get('mae', float('nan')):.6f} | {row.get('rmse', float('nan')):.6f} | "
                f"{row.get('median', float('nan')):.6f}x | {row.get('mean_rel', float('nan')):.6f} |"
            )
        summary_lines.extend(
            [
                "",
                "Takeaway:",
                "",
                "Hook summaries are expected to be very strong because the exact formulas depend on hook lengths.",
            ]
        )
        (args.out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    elif not args.run:
        print("\nDry run only. Add --run to train.", flush=True)


if __name__ == "__main__":
    main()
