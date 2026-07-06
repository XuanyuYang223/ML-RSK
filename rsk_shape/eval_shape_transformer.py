#!/usr/bin/env python3
"""Evaluate a trained Transformer shape model on an RSK dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_shape_transformer import RSKShapeDataset, ShapeTransformer, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/rsk_shape/shape_transformer.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(checkpoint: dict, n: int, device: torch.device) -> ShapeTransformer:
    train_args = checkpoint.get("args", {})
    model = ShapeTransformer(
        n=n,
        d_model=int(train_args.get("d_model", 128)),
        num_heads=int(train_args.get("num_heads", 4)),
        num_layers=int(train_args.get("num_layers", 3)),
        dim_feedforward=int(train_args.get("dim_feedforward", 512)),
        dropout=float(train_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(
    model: ShapeTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    total_examples = 0
    metric_sums = {"row_acc": 0.0, "exact_acc": 0.0, "row_mae": 0.0, "total_box_mae": 0.0}
    sample_perms = []
    sample_targets = []
    sample_preds = []

    with torch.inference_mode():
        for permutations, shapes in loader:
            permutations = permutations.to(device)
            shapes = shapes.to(device)
            logits = model(permutations)
            preds = logits.argmax(dim=-1)
            metrics = compute_metrics(logits, shapes)

            batch_size = permutations.shape[0]
            total_examples += batch_size
            for key, value in metrics.items():
                metric_sums[key] += value * batch_size

            if len(sample_perms) < 1:
                sample_perms.append(permutations.cpu())
                sample_targets.append(shapes.cpu())
                sample_preds.append(preds.cpu())

    mean_metrics = {key: value / total_examples for key, value in metric_sums.items()}
    return mean_metrics, sample_perms[0], sample_targets[0], sample_preds[0]


def trim_shape(shape: list[int]) -> list[int]:
    while shape and shape[-1] == 0:
        shape.pop()
    return shape


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    shapes = data["shapes"].astype(np.int64)
    n = int(data["n"])

    # The checkpoint is produced locally by train_shape_transformer.py and stores
    # argparse values such as pathlib.Path, so it needs full pickle loading.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_n = int(checkpoint.get("n", n))
    if checkpoint_n != n:
        raise ValueError(f"checkpoint n={checkpoint_n} does not match dataset n={n}")

    model = build_model(checkpoint, n, device)
    loader = DataLoader(RSKShapeDataset(permutations, shapes), batch_size=args.batch_size, shuffle=False)
    metrics, sample_perms, sample_targets, sample_preds = evaluate(model, loader, device)

    print(f"dataset={args.dataset}")
    print(f"checkpoint={args.checkpoint}")
    print(f"n={n} examples={len(permutations)} device={device}")
    print(f"row_acc={metrics['row_acc']:.4f}")
    print(f"exact_acc={metrics['exact_acc']:.4f}")
    print(f"row_mae={metrics['row_mae']:.4f}")
    print(f"total_box_mae={metrics['total_box_mae']:.4f}")

    print("\nexamples:")
    for i in range(min(args.examples, sample_perms.shape[0])):
        perm = sample_perms[i].tolist()
        true_shape = trim_shape(sample_targets[i].tolist())
        pred_shape = trim_shape(sample_preds[i].tolist())
        print(f"{i + 1}. perm={perm}")
        print(f"   true={true_shape}")
        print(f"   pred={pred_shape}")


if __name__ == "__main__":
    main()
