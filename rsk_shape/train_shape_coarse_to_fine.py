#!/usr/bin/env python3
"""Train a coarse-to-fine multitask classifier for RSK shape prediction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_shape_partition_classifier import PartitionClassifier, shape_labels
from train_shape_transformer import encode_permutations, split_indices


class CoarseShapeDataset(Dataset):
    def __init__(
        self,
        permutations: np.ndarray,
        labels: np.ndarray,
        shapes: np.ndarray,
        first_rows: np.ndarray,
        num_rows: np.ndarray,
        first2_labels: np.ndarray,
    ) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)
        self.first_rows = torch.as_tensor(first_rows, dtype=torch.long)
        self.num_rows = torch.as_tensor(num_rows, dtype=torch.long)
        self.first2_labels = torch.as_tensor(first2_labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.permutations[idx],
            self.labels[idx],
            self.shapes[idx],
            self.first_rows[idx],
            self.num_rows[idx],
            self.first2_labels[idx],
        )


class CoarseToFineClassifier(nn.Module):
    def __init__(
        self,
        n: int,
        num_classes: int,
        num_first2: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n = n
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.token_embedding = nn.Embedding(n + 1, d_model)
        self.pos_embedding = nn.Embedding(n + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
        self.first_row_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n + 1))
        self.num_rows_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n + 1))
        self.first2_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_first2))

    def encode(self, permutations: torch.Tensor) -> torch.Tensor:
        batch_size, n = permutations.shape
        if n != self.n:
            raise ValueError(f"model was built for n={self.n}, got n={n}")
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, self.token_embedding(permutations)], dim=1)
        positions = torch.arange(n + 1, device=permutations.device)
        encoded = self.encoder(x + self.pos_embedding(positions)[None, :, :])
        return encoded[:, 0]

    def forward(self, permutations: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.encode(permutations)
        return {
            "partition": self.classifier(pooled),
            "first_row": self.first_row_head(pooled),
            "num_rows": self.num_rows_head(pooled),
            "first2": self.first2_head(pooled),
        }


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


def build_first2_maps(candidates: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    values = sorted({tuple(row[:2].tolist()) for row in candidates})
    label_by_pair = {pair: idx for idx, pair in enumerate(values)}
    partition_labels = np.asarray([label_by_pair[tuple(row[:2].tolist())] for row in candidates], dtype=np.int64)
    return label_by_pair, partition_labels


def coarse_labels(shapes: np.ndarray, first2_map: dict[tuple[int, int], int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_rows = shapes[:, 0].astype(np.int64)
    num_rows = (shapes > 0).sum(axis=1).astype(np.int64)
    first2 = np.asarray([first2_map[tuple(row[:2].tolist())] for row in shapes], dtype=np.int64)
    return first_rows, num_rows, first2


def build_model_from_checkpoint(
    checkpoint: dict,
    n: int,
    num_first2: int,
    device: torch.device,
) -> CoarseToFineClassifier:
    train_args = checkpoint.get("args", {})
    partitions = checkpoint["partitions"]
    model = CoarseToFineClassifier(
        n=n,
        num_classes=int(checkpoint.get("num_classes", len(partitions))),
        num_first2=num_first2,
        d_model=int(train_args.get("d_model", 128)),
        num_heads=int(train_args.get("num_heads", 4)),
        num_layers=int(train_args.get("num_layers", 2)),
        dim_feedforward=int(train_args.get("dim_feedforward", 512)),
        dropout=float(train_args.get("dropout", 0.1)),
    )
    state = checkpoint["model_state_dict"]
    own_state = model.state_dict()
    copied = {
        key: value
        for key, value in state.items()
        if key in own_state and own_state[key].shape == value.shape
    }
    model.load_state_dict(copied, strict=False)
    model.to(device)
    return model


def topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    use_k = min(k, logits.shape[1])
    return logits.topk(k=use_k, dim=1).indices.eq(labels[:, None]).any(dim=1).float().mean().item()


def shape_metrics(pred_labels: torch.Tensor, labels: torch.Tensor, true_shapes: torch.Tensor, candidates: torch.Tensor) -> dict[str, float]:
    pred_shapes = candidates[pred_labels]
    correct_rows = pred_shapes.eq(true_shapes)
    return {
        "exact_acc": pred_labels.eq(labels).float().mean().item(),
        "row_acc": correct_rows.float().mean().item(),
        "row_mae": (pred_shapes - true_shapes).abs().float().mean().item(),
        "total_box_mae": (pred_shapes.sum(dim=1) - true_shapes.sum(dim=1)).abs().float().mean().item(),
    }


def masked_argmax(logits: torch.Tensor, allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(~allowed, -torch.inf)
    has_allowed = allowed.any(dim=1)
    pred = masked.argmax(dim=1)
    fallback = logits.argmax(dim=1)
    pred = torch.where(has_allowed, pred, fallback)
    return pred, has_allowed


def coarse_decode_predictions(
    outputs: dict[str, torch.Tensor],
    partition_first_rows: torch.Tensor,
    partition_num_rows: torch.Tensor,
    partition_first2: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    logits = outputs["partition"]
    batch = logits.shape[0]
    device = logits.device
    ones = torch.ones((batch, logits.shape[1]), dtype=torch.bool, device=device)
    decoded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {"argmax": (logits.argmax(dim=1), torch.ones(batch, dtype=torch.bool, device=device))}

    for k in [1, 3, 5]:
        first_values = outputs["first_row"].topk(k=min(k, outputs["first_row"].shape[1]), dim=1).indices
        first_allowed = (partition_first_rows[None, :, None] == first_values[:, None, :]).any(dim=2)
        decoded[f"first_row_top{k}"] = masked_argmax(logits, first_allowed)

        num_values = outputs["num_rows"].topk(k=min(k, outputs["num_rows"].shape[1]), dim=1).indices
        num_allowed = (partition_num_rows[None, :, None] == num_values[:, None, :]).any(dim=2)
        decoded[f"num_rows_top{k}"] = masked_argmax(logits, num_allowed)

        first2_values = outputs["first2"].topk(k=min(k, outputs["first2"].shape[1]), dim=1).indices
        first2_allowed = (partition_first2[None, :, None] == first2_values[:, None, :]).any(dim=2)
        decoded[f"first2_top{k}"] = masked_argmax(logits, first2_allowed)

        decoded[f"first_row_and_num_rows_top{k}"] = masked_argmax(logits, first_allowed & num_allowed)
        decoded[f"first_row_and_first2_top{k}"] = masked_argmax(logits, first_allowed & first2_allowed)

    _ = ones
    return decoded


def run_epoch(
    model: CoarseToFineClassifier,
    loader: DataLoader,
    candidates: torch.Tensor,
    partition_first_rows: torch.Tensor,
    partition_num_rows: torch.Tensor,
    partition_first2: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    coarse_weight: float,
) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_examples = 0
    loss_sum = 0.0
    metric_sums = {
        "partition_acc": 0.0,
        "partition_top5": 0.0,
        "partition_top10": 0.0,
        "partition_top50": 0.0,
        "partition_top100": 0.0,
        "first_row_acc": 0.0,
        "num_rows_acc": 0.0,
        "first2_acc": 0.0,
    }
    decode_sums: dict[str, dict[str, float]] = {}

    for batch in loader:
        permutations, labels, shapes, first_rows, num_rows, first2_labels = [item.to(device) for item in batch]
        with torch.set_grad_enabled(is_train):
            outputs = model(permutations)
            partition_loss = nn.functional.cross_entropy(outputs["partition"], labels)
            coarse_loss = (
                nn.functional.cross_entropy(outputs["first_row"], first_rows)
                + nn.functional.cross_entropy(outputs["num_rows"], num_rows)
                + nn.functional.cross_entropy(outputs["first2"], first2_labels)
            )
            loss = partition_loss + coarse_weight * coarse_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = labels.shape[0]
        total_examples += batch_size
        loss_sum += float(loss.item()) * batch_size
        partition_logits = outputs["partition"].detach()
        metric_sums["partition_acc"] += partition_logits.argmax(dim=1).eq(labels).float().mean().item() * batch_size
        metric_sums["partition_top5"] += topk_acc(partition_logits, labels, 5) * batch_size
        metric_sums["partition_top10"] += topk_acc(partition_logits, labels, 10) * batch_size
        metric_sums["partition_top50"] += topk_acc(partition_logits, labels, 50) * batch_size
        metric_sums["partition_top100"] += topk_acc(partition_logits, labels, 100) * batch_size
        metric_sums["first_row_acc"] += outputs["first_row"].detach().argmax(dim=1).eq(first_rows).float().mean().item() * batch_size
        metric_sums["num_rows_acc"] += outputs["num_rows"].detach().argmax(dim=1).eq(num_rows).float().mean().item() * batch_size
        metric_sums["first2_acc"] += outputs["first2"].detach().argmax(dim=1).eq(first2_labels).float().mean().item() * batch_size

        decoded = coarse_decode_predictions(
            {key: value.detach() for key, value in outputs.items()},
            partition_first_rows,
            partition_num_rows,
            partition_first2,
        )
        for name, (pred_labels, has_allowed) in decoded.items():
            metrics = shape_metrics(pred_labels, labels, shapes, candidates)
            if name not in decode_sums:
                decode_sums[name] = {key: 0.0 for key in metrics}
                decode_sums[name]["coverage"] = 0.0
            for key, value in metrics.items():
                decode_sums[name][key] += value * batch_size
            decode_sums[name]["coverage"] += has_allowed.float().mean().item() * batch_size

    metrics = {key: value / total_examples for key, value in metric_sums.items()}
    decode_metrics = {
        name: {key: value / total_examples for key, value in sums.items()}
        for name, sums in decode_sums.items()
    }
    return loss_sum / total_examples, metrics, decode_metrics


def evaluate(
    model: CoarseToFineClassifier,
    permutations: np.ndarray,
    labels: np.ndarray,
    shapes: np.ndarray,
    first_rows: np.ndarray,
    num_rows: np.ndarray,
    first2_labels: np.ndarray,
    candidates: torch.Tensor,
    partition_first_rows: torch.Tensor,
    partition_num_rows: torch.Tensor,
    partition_first2: torch.Tensor,
    batch_size: int,
    device: torch.device,
    coarse_weight: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    loader = DataLoader(
        CoarseShapeDataset(permutations, labels, shapes, first_rows, num_rows, first2_labels),
        batch_size=batch_size,
        shuffle=False,
    )
    _loss, metrics, decode_metrics = run_epoch(
        model,
        loader,
        candidates,
        partition_first_rows,
        partition_num_rows,
        partition_first2,
        device,
        optimizer=None,
        coarse_weight=coarse_weight,
    )
    return metrics, decode_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--coarse-weight", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=Path("experiments/rsk_shape_coarse_to_fine/model.pt"))
    parser.add_argument("--results-out", type=Path, default=Path("experiments/rsk_shape_coarse_to_fine/results.json"))
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
    first2_map, partition_first2_np = build_first2_maps(candidates_np)
    labels = shape_labels(shapes, candidates_np)
    first_rows, num_rows, first2_labels = coarse_labels(shapes, first2_map)
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
    eval_first_rows, eval_num_rows, eval_first2_labels = coarse_labels(eval_shapes, first2_map)
    eval_encoded = encode_permutations(eval_permutations, representation)

    model = build_model_from_checkpoint(checkpoint, n, len(first2_map), device)
    candidates = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
    partition_first_rows = candidates[:, 0]
    partition_num_rows = (candidates > 0).sum(dim=1)
    partition_first2 = torch.as_tensor(partition_first2_np, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        CoarseShapeDataset(
            encoded[train_idx],
            labels[train_idx],
            shapes[train_idx],
            first_rows[train_idx],
            num_rows[train_idx],
            first2_labels[train_idx],
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        CoarseShapeDataset(
            encoded[valid_idx],
            labels[valid_idx],
            shapes[valid_idx],
            first_rows[valid_idx],
            num_rows[valid_idx],
            first2_labels[valid_idx],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(
        f"dataset={args.dataset} eval_dataset={args.eval_dataset} n={n} classes={candidates_np.shape[0]} "
        f"first2_classes={len(first2_map)} train={len(train_idx)} valid={len(valid_idx)} "
        f"representation={representation} coarse_weight={args.coarse_weight} device={device}",
        flush=True,
    )

    best_score = (-1.0, -1.0)
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _train_decode = run_epoch(
            model,
            train_loader,
            candidates,
            partition_first_rows,
            partition_num_rows,
            partition_first2,
            device,
            optimizer,
            args.coarse_weight,
        )
        valid_loss, valid_metrics, valid_decode = run_epoch(
            model,
            valid_loader,
            candidates,
            partition_first_rows,
            partition_num_rows,
            partition_first2,
            device,
            optimizer=None,
            coarse_weight=args.coarse_weight,
        )
        best_decode_name, best_decode_metrics = max(
            valid_decode.items(),
            key=lambda item: (item[1]["exact_acc"], item[1]["row_acc"]),
        )
        score = (best_decode_metrics["exact_acc"], valid_metrics["partition_top50"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train": train_metrics,
            "valid": valid_metrics,
            "valid_decode": valid_decode,
            "best_decode": best_decode_name,
        }
        history.append(row)
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_argmax={valid_decode['argmax']['exact_acc']:.4f} "
            f"valid_best_decode={best_decode_name}:{best_decode_metrics['exact_acc']:.4f} "
            f"valid_top50={valid_metrics['partition_top50']:.4f} "
            f"valid_top100={valid_metrics['partition_top100']:.4f} "
            f"first_row_acc={valid_metrics['first_row_acc']:.4f} "
            f"num_rows_acc={valid_metrics['num_rows_acc']:.4f} "
            f"first2_acc={valid_metrics['first2_acc']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_metrics, eval_decode = evaluate(
        model,
        eval_encoded,
        eval_labels,
        eval_shapes,
        eval_first_rows,
        eval_num_rows,
        eval_first2_labels,
        candidates,
        partition_first_rows,
        partition_num_rows,
        partition_first2,
        args.batch_size,
        device,
        args.coarse_weight,
    )
    best_eval_name, best_eval_decode = max(eval_decode.items(), key=lambda item: (item[1]["exact_acc"], item[1]["row_acc"]))
    print(
        f"eval_dataset={args.eval_dataset} "
        f"eval_argmax={eval_decode['argmax']['exact_acc']:.4f} "
        f"eval_best_decode={best_eval_name}:{best_eval_decode['exact_acc']:.4f} "
        f"eval_top50={eval_metrics['partition_top50']:.4f} "
        f"eval_top100={eval_metrics['partition_top100']:.4f} "
        f"eval_row_acc={best_eval_decode['row_acc']:.4f} "
        f"eval_row_mae={best_eval_decode['row_mae']:.4f} "
        f"eval_total_box_mae={best_eval_decode['total_box_mae']:.4f}",
        flush=True,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "n": n,
            "num_classes": candidates_np.shape[0],
            "partitions": candidates_np,
            "partition_first2": partition_first2_np,
            "args": vars(args),
            "best_valid_exact_acc": best_score[0],
            "best_valid_top50_acc": best_score[1],
            "eval_metrics": eval_metrics,
            "eval_decode": eval_decode,
        },
        args.model_out,
    )
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "best_valid_exact_acc": best_score[0],
        "best_valid_top50_acc": best_score[1],
        "history": history,
        "eval_metrics": eval_metrics,
        "eval_decode": eval_decode,
        "best_eval_decode": best_eval_name,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.model_out}")
    print(f"wrote {args.results_out}")


if __name__ == "__main__":
    main()
