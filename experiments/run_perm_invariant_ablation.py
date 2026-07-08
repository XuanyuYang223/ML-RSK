#!/usr/bin/env python3
"""Run permutation invariant ablations and record each configuration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


CONFIGS = [
    {"name": "length_classification_one_line", "task": "length", "target_mode": "classification", "repr": "one_line"},
    {"name": "length_classification_lehmer", "task": "length", "target_mode": "classification", "repr": "lehmer"},
    {"name": "length_regression_one_line", "task": "length", "target_mode": "regression", "repr": "one_line"},
    {"name": "length_regression_lehmer", "task": "length", "target_mode": "regression", "repr": "lehmer"},
    {"name": "sign_classification_one_line", "task": "sign", "target_mode": "classification", "repr": "one_line"},
    {"name": "sign_classification_lehmer", "task": "sign", "target_mode": "classification", "repr": "lehmer"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/rsk_shape/rsk_n30_m100000_seed2.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/perm_invariants_ablation_20k_e8"))
    parser.add_argument("--python", type=Path, default=Path("/home/yangx/miniforge3/envs/sage/bin/python"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, config: dict[str, str], run_dir: Path) -> list[str]:
    command = [
        str(args.python),
        "-u",
        "perm_invariants/train_perm_invariants_transformer.py",
        str(args.dataset),
        "--task",
        config["task"],
        "--target-mode",
        config["target_mode"],
        "--perm-representation",
        config["repr"],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--d-model",
        str(args.d_model),
        "--num-heads",
        str(args.num_heads),
        "--num-layers",
        str(args.num_layers),
        "--dim-feedforward",
        str(args.dim_feedforward),
        "--max-samples",
        str(args.max_samples),
        "--device",
        args.device,
        "--model-out",
        str(run_dir / "model.pt"),
    ]
    return command


def parse_last_epoch(log_text: str) -> dict[str, float]:
    epoch_lines = [line for line in log_text.splitlines() if line.startswith("epoch=")]
    if not epoch_lines:
        return {}
    line = epoch_lines[-1]
    metrics = {}
    for key, value in re.findall(r"(test_[a-z_]+)=([0-9.]+)", line):
        metrics[key] = float(value)
    return metrics


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

    for config in CONFIGS:
        run_dir = args.out_dir / config["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, config, run_dir)
        metadata = {
            **config,
            "dataset": str(args.dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "max_samples": args.max_samples,
            "device": args.device,
            "model_out": str(run_dir / "model.pt"),
        }
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        print(f"prepared {run_dir}", flush=True)

        if args.run:
            log_text = run_command(command, run_dir)
            rows.append({**metadata, **parse_last_epoch(log_text)})

    if rows:
        lines = [
            "# Permutation Invariant Ablation",
            "",
            "| name | task | mode | representation | test acc | test MAE | test RMSE | rounded acc |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['name']} | {row['task']} | {row['target_mode']} | {row['repr']} | "
                f"{row.get('test_acc', float('nan')):.4f} | {row.get('test_mae', float('nan')):.3f} | "
                f"{row.get('test_rmse', float('nan')):.3f} | {row.get('test_rounded_acc', float('nan')):.4f} |"
            )
        lines.extend(
            [
                "",
                "Notes:",
                "",
                "- Lehmer code is a deterministic representation of the permutation.",
                "- For Coxeter length, `sum(Lehmer code)` equals the target, so this is a strong inductive bias.",
            ]
        )
        (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    elif not args.run:
        print("\nDry run only. Add --run to train.", flush=True)


if __name__ == "__main__":
    main()
