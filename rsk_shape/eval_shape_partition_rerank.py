#!/usr/bin/env python3
"""Rerank partition-classifier top-k candidates with row-model scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_shape_partition_classifier import PartitionClassifier, shape_labels
from train_shape_transformer import RSKShapeDataset, ShapeTransformer, encode_permutations, encode_shapes


class RerankDataset(Dataset):
    def __init__(
        self,
        classifier_permutations: np.ndarray,
        row_permutations: np.ndarray,
        labels: np.ndarray,
        shapes: np.ndarray,
    ) -> None:
        self.classifier_permutations = torch.as_tensor(classifier_permutations, dtype=torch.long)
        self.row_permutations = torch.as_tensor(row_permutations, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.classifier_permutations[idx],
            self.row_permutations[idx],
            self.labels[idx],
            self.shapes[idx],
        )


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_classifier(checkpoint: dict, n: int, device: torch.device) -> PartitionClassifier:
    train_args = checkpoint.get("args", {})
    partitions = checkpoint["partitions"]
    model = PartitionClassifier(
        n=n,
        num_classes=int(checkpoint.get("num_classes", len(partitions))),
        d_model=int(train_args.get("d_model", 128)),
        num_heads=int(train_args.get("num_heads", 4)),
        num_layers=int(train_args.get("num_layers", 2)),
        dim_feedforward=int(train_args.get("dim_feedforward", 512)),
        dropout=float(train_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_row_model(checkpoint: dict, n: int, device: torch.device) -> ShapeTransformer:
    train_args = checkpoint.get("args", {})
    model = ShapeTransformer(
        n=n,
        d_model=int(train_args.get("d_model", 128)),
        num_heads=int(train_args.get("num_heads", 4)),
        num_layers=int(train_args.get("num_layers", 3)),
        dim_feedforward=int(train_args.get("dim_feedforward", 512)),
        dropout=float(train_args.get("dropout", 0.1)),
        architecture=str(train_args.get("architecture", "encoder_decoder")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def score_candidates_with_rows(row_logits: torch.Tensor, candidate_shapes: torch.Tensor) -> torch.Tensor:
    batch_size, top_k, n = candidate_shapes.shape
    row_idx = torch.arange(n, device=row_logits.device)[None, None, :].expand(batch_size, top_k, -1)
    batch_idx = torch.arange(batch_size, device=row_logits.device)[:, None, None].expand(-1, top_k, n)
    token_scores = row_logits.log_softmax(dim=-1)[batch_idx, row_idx, candidate_shapes]
    return token_scores.sum(dim=2)


def compute_shape_metrics(pred_shapes: torch.Tensor, true_shapes: torch.Tensor, pred_labels: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    correct_rows = pred_shapes.eq(true_shapes)
    return {
        "exact_acc": pred_labels.eq(labels).float().mean().item(),
        "row_acc": correct_rows.float().mean().item(),
        "row_mae": (pred_shapes - true_shapes).abs().float().mean().item(),
        "total_box_mae": (pred_shapes.sum(dim=1) - true_shapes.sum(dim=1)).abs().float().mean().item(),
    }


def evaluate(
    classifier: PartitionClassifier,
    row_model: ShapeTransformer,
    loader: DataLoader,
    candidates: torch.Tensor,
    top_k: int,
    alphas: list[float],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    metric_sums: dict[str, dict[str, float]] = {}
    for name in ["classifier_argmax", "row_rerank", *[f"mix_alpha_{alpha:g}" for alpha in alphas], "oracle_topk"]:
        metric_sums[name] = {"exact_acc": 0.0, "row_acc": 0.0, "row_mae": 0.0, "total_box_mae": 0.0}
    total_examples = 0

    with torch.inference_mode():
        for classifier_perms, row_perms, labels, true_shapes in loader:
            classifier_perms = classifier_perms.to(device)
            row_perms = row_perms.to(device)
            labels = labels.to(device)
            true_shapes = true_shapes.to(device)

            classifier_logits = classifier(classifier_perms)
            classifier_log_probs = classifier_logits.log_softmax(dim=1)
            top_scores, top_labels = classifier_log_probs.topk(k=top_k, dim=1)
            top_shapes = candidates[top_labels]

            row_logits = row_model(row_perms)
            row_scores = score_candidates_with_rows(row_logits, top_shapes)

            predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            classifier_pred_labels = top_labels[:, 0]
            predictions["classifier_argmax"] = (classifier_pred_labels, candidates[classifier_pred_labels])

            row_choice = row_scores.argmax(dim=1)
            row_pred_labels = top_labels.gather(1, row_choice[:, None]).squeeze(1)
            predictions["row_rerank"] = (row_pred_labels, candidates[row_pred_labels])

            for alpha in alphas:
                mix_choice = (top_scores + alpha * row_scores).argmax(dim=1)
                mix_pred_labels = top_labels.gather(1, mix_choice[:, None]).squeeze(1)
                predictions[f"mix_alpha_{alpha:g}"] = (mix_pred_labels, candidates[mix_pred_labels])

            oracle_hit = top_labels.eq(labels[:, None])
            oracle_pred_labels = torch.where(oracle_hit.any(dim=1), labels, classifier_pred_labels)
            predictions["oracle_topk"] = (oracle_pred_labels, candidates[oracle_pred_labels])

            batch_size = labels.shape[0]
            total_examples += batch_size
            for name, (pred_labels, pred_shapes) in predictions.items():
                metrics = compute_shape_metrics(pred_shapes, true_shapes, pred_labels, labels)
                for key, value in metrics.items():
                    metric_sums[name][key] += value * batch_size

    return {
        name: {key: value / total_examples for key, value in sums.items()}
        for name, sums in metric_sums.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--row-checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    shapes = data["shapes"].astype(np.int64)
    n = int(data["n"])
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
        shapes = shapes[: args.max_samples]

    classifier_checkpoint = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=False)
    row_checkpoint = torch.load(args.row_checkpoint, map_location="cpu", weights_only=False)
    if int(classifier_checkpoint.get("n", n)) != n:
        raise ValueError("classifier checkpoint n does not match dataset")
    if int(row_checkpoint.get("n", n)) != n:
        raise ValueError("row checkpoint n does not match dataset")

    candidates_np = classifier_checkpoint["partitions"].astype(np.int64)
    labels = shape_labels(shapes, candidates_np)
    classifier_representation = str(classifier_checkpoint.get("args", {}).get("perm_representation", "lehmer"))
    row_args = row_checkpoint.get("args", {})
    row_representation = str(row_args.get("perm_representation", "lehmer"))
    row_output_mode = str(row_args.get("output_mode", "rows"))
    if row_output_mode != "rows":
        raise ValueError(f"row reranking expects rows output, got {row_output_mode}")

    classifier_perms = encode_permutations(permutations, classifier_representation)
    row_perms = encode_permutations(permutations, row_representation)
    _ = encode_shapes(shapes, row_output_mode)

    classifier = build_classifier(classifier_checkpoint, n, device)
    row_model = build_row_model(row_checkpoint, n, device)
    candidates = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
    if args.top_k > candidates.shape[0]:
        raise ValueError(f"top-k={args.top_k} exceeds num classes={candidates.shape[0]}")

    loader = DataLoader(
        RerankDataset(classifier_perms, row_perms, labels, shapes),
        batch_size=args.batch_size,
        shuffle=False,
    )
    results = evaluate(classifier, row_model, loader, candidates, args.top_k, args.alphas, device)

    print(f"dataset={args.dataset}")
    print(f"classifier_checkpoint={args.classifier_checkpoint}")
    print(f"row_checkpoint={args.row_checkpoint}")
    print(
        f"n={n} examples={len(permutations)} top_k={args.top_k} "
        f"classifier_representation={classifier_representation} row_representation={row_representation} "
        f"device={device}"
    )
    for name, metrics in results.items():
        print(
            f"{name} "
            f"exact_acc={metrics['exact_acc']:.4f} "
            f"row_acc={metrics['row_acc']:.4f} "
            f"row_mae={metrics['row_mae']:.4f} "
            f"total_box_mae={metrics['total_box_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
