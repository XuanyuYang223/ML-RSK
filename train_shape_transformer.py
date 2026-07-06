#!/usr/bin/env python3
"""Train a Transformer to predict the padded RSK shape from a permutation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class RSKShapeDataset(Dataset):
    def __init__(self, permutations: np.ndarray, shapes: np.ndarray) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.permutations.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.shapes[idx]


class ShapeTransformer(nn.Module):
    def __init__(
        self,
        n: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n = n
        self.perm_embedding = nn.Embedding(n + 1, d_model)
        self.input_pos_embedding = nn.Embedding(n, d_model)
        self.shape_query_embedding = nn.Embedding(n, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n + 1)

    def forward(self, permutations: torch.Tensor) -> torch.Tensor:
        batch_size, n = permutations.shape
        if n != self.n:
            raise ValueError(f"model was built for n={self.n}, got n={n}")

        positions = torch.arange(n, device=permutations.device)
        src = self.perm_embedding(permutations) + self.input_pos_embedding(positions)[None, :, :]
        memory = self.encoder(src)

        shape_queries = self.shape_query_embedding(positions)[None, :, :].expand(batch_size, -1, -1)
        decoded = self.decoder(shape_queries, memory)
        return self.classifier(decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=Path("models/shape_transformer.pt"))
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


def split_arrays(
    permutations: np.ndarray,
    shapes: np.ndarray,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(permutations) < 2:
        raise ValueError("need at least two examples to make a train/test split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(permutations))
    test_size = max(1, int(round(len(order) * test_frac)))
    train_size = len(order) - test_size
    if train_size < 1:
        train_size = len(order) - 1
        test_size = 1
    train_idx = order[:train_size]
    test_idx = order[train_size : train_size + test_size]
    return permutations[train_idx], shapes[train_idx], permutations[test_idx], shapes[test_idx]


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    correct = pred.eq(targets)
    row_acc = correct.float().mean().item()
    exact_acc = correct.all(dim=1).float().mean().item()
    row_mae = (pred - targets).abs().float().mean().item()
    total_box_mae = (pred.sum(dim=1) - targets.sum(dim=1)).abs().float().mean().item()
    return {
        "row_acc": row_acc,
        "exact_acc": exact_acc,
        "row_mae": row_mae,
        "total_box_mae": total_box_mae,
    }


def run_epoch(
    model: ShapeTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    metric_sums = {"row_acc": 0.0, "exact_acc": 0.0, "row_mae": 0.0, "total_box_mae": 0.0}

    for permutations, shapes in loader:
        permutations = permutations.to(device)
        shapes = shapes.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(permutations)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), shapes.reshape(-1))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = permutations.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        metrics = compute_metrics(logits.detach(), shapes)
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    mean_metrics = {key: value / total_examples for key, value in metric_sums.items()}
    return total_loss / total_examples, mean_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    shapes = data["shapes"].astype(np.int64)
    n = int(data["n"])

    train_x, train_y, test_x, test_y = split_arrays(permutations, shapes, args.test_frac, args.seed)
    train_loader = DataLoader(
        RSKShapeDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        RSKShapeDataset(test_x, test_y),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ShapeTransformer(
        n=n,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"dataset={args.dataset} n={n} train={len(train_x)} test={len(test_x)} "
        f"device={device} parameters={sum(p.numel() for p in model.parameters())}"
    )
    best_exact_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_metrics = run_epoch(model, test_loader, criterion, device)
        if test_metrics["exact_acc"] > best_exact_acc:
            best_exact_acc = test_metrics["exact_acc"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"train_row_acc={train_metrics['row_acc']:.3f} "
            f"test_row_acc={test_metrics['row_acc']:.3f} "
            f"test_exact_acc={test_metrics['exact_acc']:.3f} "
            f"test_row_mae={test_metrics['row_mae']:.3f} "
            f"test_total_box_mae={test_metrics['total_box_mae']:.3f}"
        )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "n": n,
            "args": vars(args),
            "best_exact_acc": best_exact_acc,
        },
        args.model_out,
    )
    print(f"wrote {args.model_out}")


if __name__ == "__main__":
    main()
