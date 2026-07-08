#!/usr/bin/env python3
"""Fine-tune the partition classifier with hard-negative top-k losses."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_shape_partition_classifier import PartitionClassDataset, PartitionClassifier, shape_labels
from train_shape_transformer import encode_permutations, split_indices


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return model


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    true_shapes: torch.Tensor,
    candidates: torch.Tensor,
    top_ks: tuple[int, ...] = (5, 10, 50, 100),
) -> dict[str, float]:
    pred_labels = logits.argmax(dim=1)
    pred_shapes = candidates[pred_labels]
    correct_rows = pred_shapes.eq(true_shapes)
    metrics = {
        "acc": pred_labels.eq(labels).float().mean().item(),
        "row_acc": correct_rows.float().mean().item(),
        "row_mae": (pred_shapes - true_shapes).abs().float().mean().item(),
        "total_box_mae": (pred_shapes.sum(dim=1) - true_shapes.sum(dim=1)).abs().float().mean().item(),
    }
    max_k = min(max(top_ks), logits.shape[1])
    topk = logits.topk(k=max_k, dim=1).indices
    for k in top_ks:
        use_k = min(k, max_k)
        metrics[f"top{use_k}_acc"] = topk[:, :use_k].eq(labels[:, None]).any(dim=1).float().mean().item()
    return metrics


def hard_negative_candidates(logits: torch.Tensor, labels: torch.Tensor, num_negatives: int) -> torch.Tensor:
    top_indices = logits.detach().topk(k=num_negatives + 1, dim=1).indices
    rows = []
    for row, label in zip(top_indices, labels, strict=True):
        negatives = row[row != label]
        if negatives.numel() < num_negatives:
            raise RuntimeError("not enough hard negatives; increase top-k extraction")
        rows.append(negatives[:num_negatives])
    return torch.stack(rows, dim=0)


def aux_topk_loss(logits: torch.Tensor, labels: torch.Tensor, num_negatives: int) -> torch.Tensor:
    negatives = hard_negative_candidates(logits, labels, num_negatives)
    candidate_labels = torch.cat([labels[:, None], negatives], dim=1)
    candidate_logits = logits.gather(1, candidate_labels)
    targets = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
    return nn.functional.cross_entropy(candidate_logits, targets)


def run_epoch(
    model: PartitionClassifier,
    loader: DataLoader,
    candidates: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    aux_weight: float,
    num_negatives: int,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_ce_loss = 0.0
    total_aux_loss = 0.0
    total_examples = 0
    metric_sums = {
        "acc": 0.0,
        "top5_acc": 0.0,
        "top10_acc": 0.0,
        "top50_acc": 0.0,
        "top100_acc": 0.0,
        "row_acc": 0.0,
        "row_mae": 0.0,
        "total_box_mae": 0.0,
    }

    for permutations, labels, shapes in loader:
        permutations = permutations.to(device)
        labels = labels.to(device)
        shapes = shapes.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(permutations)
            ce_loss = nn.functional.cross_entropy(logits, labels)
            hard_loss = aux_topk_loss(logits, labels, num_negatives)
            loss = ce_loss + aux_weight * hard_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_ce_loss += float(ce_loss.item()) * batch_size
        total_aux_loss += float(hard_loss.item()) * batch_size
        total_examples += batch_size
        metrics = compute_metrics(logits.detach(), labels, shapes, candidates)
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    metrics = {key: value / total_examples for key, value in metric_sums.items()}
    metrics["ce_loss"] = total_ce_loss / total_examples
    metrics["aux_loss"] = total_aux_loss / total_examples
    return total_loss / total_examples, metrics


def evaluate_dataset(
    model: PartitionClassifier,
    permutations: np.ndarray,
    labels: np.ndarray,
    shapes: np.ndarray,
    candidates: torch.Tensor,
    batch_size: int,
    device: torch.device,
    num_negatives: int,
) -> dict[str, float]:
    loader = DataLoader(PartitionClassDataset(permutations, labels, shapes), batch_size=batch_size, shuffle=False)
    _loss, metrics = run_epoch(
        model,
        loader,
        candidates,
        device,
        optimizer=None,
        aux_weight=0.0,
        num_negatives=num_negatives,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=int, default=50)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=Path("experiments/rsk_shape_classifier_aux_topk/model.pt"))
    parser.add_argument("--results-out", type=Path, default=Path("experiments/rsk_shape_classifier_aux_topk/results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    checkpoint = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=False)
    train_data = np.load(args.dataset)
    permutations = train_data["permutations"].astype(np.int64)
    shapes = train_data["shapes"].astype(np.int64)
    n = int(train_data["n"])
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
        shapes = shapes[: args.max_samples]
    if int(checkpoint.get("n", n)) != n:
        raise ValueError("classifier checkpoint n does not match training dataset")

    candidates_np = checkpoint["partitions"].astype(np.int64)
    labels = shape_labels(shapes, candidates_np)
    representation = str(checkpoint.get("args", {}).get("perm_representation", "lehmer"))
    encoded = encode_permutations(permutations, representation)
    train_idx, valid_idx = split_indices(len(encoded), args.test_frac, args.seed)

    eval_data = np.load(args.eval_dataset)
    eval_permutations = eval_data["permutations"].astype(np.int64)
    eval_shapes = eval_data["shapes"].astype(np.int64)
    eval_n = int(eval_data["n"])
    if eval_n != n:
        raise ValueError("eval dataset n does not match training dataset")
    eval_labels = shape_labels(eval_shapes, candidates_np)
    eval_encoded = encode_permutations(eval_permutations, representation)

    model = build_classifier(checkpoint, n, device)
    candidates = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        PartitionClassDataset(encoded[train_idx], labels[train_idx], shapes[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        PartitionClassDataset(encoded[valid_idx], labels[valid_idx], shapes[valid_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(
        f"dataset={args.dataset} eval_dataset={args.eval_dataset} n={n} classes={candidates_np.shape[0]} "
        f"train={len(train_idx)} valid={len(valid_idx)} representation={representation} "
        f"hard_negatives={args.hard_negatives} aux_weight={args.aux_weight} "
        f"device={device} parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    best_score = (-1.0, -1.0)
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            candidates,
            device,
            optimizer,
            args.aux_weight,
            args.hard_negatives,
        )
        valid_loss, valid_metrics = run_epoch(
            model,
            valid_loader,
            candidates,
            device,
            optimizer=None,
            aux_weight=args.aux_weight,
            num_negatives=args.hard_negatives,
        )
        score = (valid_metrics["acc"], valid_metrics["top50_acc"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_metrics['acc']:.4f} "
            f"valid_top10={valid_metrics['top10_acc']:.4f} "
            f"valid_top50={valid_metrics['top50_acc']:.4f} "
            f"valid_top100={valid_metrics['top100_acc']:.4f} "
            f"valid_row_acc={valid_metrics['row_acc']:.4f} "
            f"valid_row_mae={valid_metrics['row_mae']:.4f} "
            f"valid_ce={valid_metrics['ce_loss']:.4f} "
            f"valid_aux={valid_metrics['aux_loss']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_metrics = evaluate_dataset(
        model,
        eval_encoded,
        eval_labels,
        eval_shapes,
        candidates,
        args.batch_size,
        device,
        args.hard_negatives,
    )
    print(
        f"eval_dataset={args.eval_dataset} "
        f"eval_acc={eval_metrics['acc']:.4f} "
        f"eval_top5={eval_metrics['top5_acc']:.4f} "
        f"eval_top10={eval_metrics['top10_acc']:.4f} "
        f"eval_top50={eval_metrics['top50_acc']:.4f} "
        f"eval_top100={eval_metrics['top100_acc']:.4f} "
        f"eval_row_acc={eval_metrics['row_acc']:.4f} "
        f"eval_row_mae={eval_metrics['row_mae']:.4f} "
        f"eval_total_box_mae={eval_metrics['total_box_mae']:.4f}",
        flush=True,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "n": n,
            "num_classes": candidates_np.shape[0],
            "partitions": candidates_np,
            "args": vars(args),
            "best_acc": best_score[0],
            "best_top50_acc": best_score[1],
            "eval_metrics": eval_metrics,
        },
        args.model_out,
    )
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "best_valid_acc": best_score[0],
        "best_valid_top50_acc": best_score[1],
        "history": history,
        "eval_metrics": eval_metrics,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.model_out}")
    print(f"wrote {args.results_out}")


if __name__ == "__main__":
    main()
