#!/usr/bin/env python3
"""Prepare or run RSK shape ablations in separate experiment folders."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    {
        "name": "baseline_one_line_encdec_rows_argmax",
        "perm_representation": "one_line",
        "architecture": "encoder_decoder",
        "output_mode": "rows",
        "inference": "argmax",
    },
    {
        "name": "inverse_encdec_rows_argmax",
        "perm_representation": "inverse",
        "architecture": "encoder_decoder",
        "output_mode": "rows",
        "inference": "argmax",
    },
    {
        "name": "lehmer_encdec_rows_argmax",
        "perm_representation": "lehmer",
        "architecture": "encoder_decoder",
        "output_mode": "rows",
        "inference": "argmax",
    },
    {
        "name": "one_line_encoder_only_rows_argmax",
        "perm_representation": "one_line",
        "architecture": "encoder_only",
        "output_mode": "rows",
        "inference": "argmax",
    },
    {
        "name": "one_line_encdec_deltas_argmax",
        "perm_representation": "one_line",
        "architecture": "encoder_decoder",
        "output_mode": "deltas",
        "inference": "argmax",
    },
    {
        "name": "one_line_encdec_rows_monotone",
        "perm_representation": "one_line",
        "architecture": "encoder_decoder",
        "output_mode": "rows",
        "inference": "monotone",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/rsk_shape/rsk_n30_m10000_seed0.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/rsk_shape_ablation"))
    parser.add_argument("--python", type=Path, default=Path("/home/yangx/miniforge3/envs/sage/bin/python"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run", action="store_true", help="Actually train. Without this, only writes configs/commands.")
    return parser.parse_args()


def build_command(args: argparse.Namespace, config: dict[str, str], run_dir: Path) -> list[str]:
    model_out = run_dir / "model.pt"
    command = [
        str(args.python),
        "-u",
        "rsk_shape/train_shape_transformer.py",
        str(args.dataset),
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
        "--perm-representation",
        config["perm_representation"],
        "--architecture",
        config["architecture"],
        "--output-mode",
        config["output_mode"],
        "--inference",
        config["inference"],
        "--device",
        args.device,
        "--model-out",
        str(model_out),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    return command


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for config in DEFAULT_CONFIGS:
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

        if not args.run:
            continue

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
            return_code = process.wait()
        if return_code != 0:
            raise SystemExit(return_code)
        print(f"finished {run_dir}", flush=True)

    if not args.run:
        print("\nDry run only. Add --run to train these configurations.", flush=True)


if __name__ == "__main__":
    main()
