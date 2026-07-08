#!/usr/bin/env python3
"""Train a classifier from permutations to RSK partition classes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_shape_transformer import encode_permutations, generate_partitions, split_indices


class PartitionClassDataset(Dataset):
    def __init__(self, permutations: np.ndarray, labels: np.ndarray, shapes: np.ndarray) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.permutations.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.labels[idx], self.shapes[idx]


class PartitionClassifier(nn.Module):
    def __init__(
        self,
        n: int,
        num_classes: int,
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
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, permutations: torch.Tensor) -> torch.Tensor:
        batch_size, n = permutations.shape
        if n != self.n:
            raise ValueError(f"model was built for n={self.n}, got n={n}")
        token_embeddings = self.token_embedding(permutations)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, token_embeddings], dim=1)
        positions = torch.arange(n + 1, device=permutations.device)
        x = x + self.pos_embedding(positions)[None, :, :]
        encoded = self.encoder(x)
        return self.classifier(encoded[:, 0])


def padded_partitions(n: int) -> np.ndarray:
    rows = []
    for partition in generate_partitions(n):
        rows.append(partition + [0] * (n - len(partition)))
    return np.asarray(rows, dtype=np.int64)


def shape_labels(shapes: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    class_by_shape = {tuple(shape.tolist()): idx for idx, shape in enumerate(candidates)}
    labels = np.empty(shapes.shape[0], dtype=np.int64)
    for idx, shape in enumerate(shapes):
        labels[idx] = class_by_shape[tuple(shape.tolist())]
    return labels


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    true_shapes: torch.Tensor,
    candidates: torch.Tensor,
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
    max_k = min(10, logits.shape[1])
    topk = logits.topk(k=max_k, dim=1).indices
    metrics["top5_acc"] = topk[:, : min(5, max_k)].eq(labels[:, None]).any(dim=1).float().mean().item()
    metrics["top10_acc"] = topk.eq(labels[:, None]).any(dim=1).float().mean().item()
    return metrics


def run_epoch(
    model: PartitionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    candidates: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    metric_sums = {
        "acc": 0.0,
        "top5_acc": 0.0,
        "top10_acc": 0.0,
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
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = permutations.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        metrics = compute_metrics(logits.detach(), labels, shapes, candidates)
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    return total_loss / total_examples, {key: value / total_examples for key, value in metric_sums.items()}


def evaluate_dataset(
    model: PartitionClassifier,
    permutations: np.ndarray,
    labels: np.ndarray,
    shapes: np.ndarray,
    candidates: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(PartitionClassDataset(permutations, labels, shapes), batch_size=batch_size, shuffle=False)
    _loss, metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), candidates, device, optimizer=None)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--eval-dataset", type=Path, default=None)
    parser.add_argument("--perm-representation", choices=["one_line", "inverse", "lehmer"], default="lehmer")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=Path("models/rsk_shape/partition_classifier.pt"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    shapes = data["shapes"].astype(np.int64)
    n = int(data["n"])
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
        shapes = shapes[: args.max_samples]

    candidates_np = padded_partitions(n)
    labels = shape_labels(shapes, candidates_np)
    encoded_permutations = encode_permutations(permutations, args.perm_representation)
    train_idx, test_idx = split_indices(len(encoded_permutations), args.test_frac, args.seed)

    train_loader = DataLoader(
        PartitionClassDataset(encoded_permutations[train_idx], labels[train_idx], shapes[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        PartitionClassDataset(encoded_permutations[test_idx], labels[test_idx], shapes[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    candidates = torch.as_tensor(candidates_np, dtype=torch.long, device=device)

    model = PartitionClassifier(
        n=n,
        num_classes=candidates_np.shape[0],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"dataset={args.dataset} n={n} classes={candidates_np.shape[0]} "
        f"train={len(train_idx)} test={len(test_idx)} representation={args.perm_representation} "
        f"device={device} parameters={sum(p.numel() for p in model.parameters())}"
    )

    best_score = (-1.0, -1.0)
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, candidates, device, optimizer)
        test_loss, test_metrics = run_epoch(model, test_loader, criterion, candidates, device, optimizer=None)
        score = (test_metrics["acc"], test_metrics["row_acc"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"test_top5_acc={test_metrics['top5_acc']:.4f} "
            f"test_top10_acc={test_metrics['top10_acc']:.4f} "
            f"test_row_acc={test_metrics['row_acc']:.4f} "
            f"test_row_mae={test_metrics['row_mae']:.4f} "
            f"test_total_box_mae={test_metrics['total_box_mae']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.eval_dataset is not None:
        eval_data = np.load(args.eval_dataset)
        eval_permutations = eval_data["permutations"].astype(np.int64)
        eval_shapes = eval_data["shapes"].astype(np.int64)
        eval_labels = shape_labels(eval_shapes, candidates_np)
        eval_encoded = encode_permutations(eval_permutations, args.perm_representation)
        eval_metrics = evaluate_dataset(
            model,
            eval_encoded,
            eval_labels,
            eval_shapes,
            candidates,
            args.batch_size,
            device,
        )
        print(
            f"eval_dataset={args.eval_dataset} "
            f"eval_acc={eval_metrics['acc']:.4f} "
            f"eval_top5_acc={eval_metrics['top5_acc']:.4f} "
            f"eval_top10_acc={eval_metrics['top10_acc']:.4f} "
            f"eval_row_acc={eval_metrics['row_acc']:.4f} "
            f"eval_row_mae={eval_metrics['row_mae']:.4f} "
            f"eval_total_box_mae={eval_metrics['total_box_mae']:.4f}"
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
            "best_row_acc": best_score[1],
        },
        args.model_out,
    )
    print(f"wrote {args.model_out}")


if __name__ == "__main__":
    main()
