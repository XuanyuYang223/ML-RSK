#!/usr/bin/env python3
"""Train an MLP to predict log #SYT or log #SSYT from a partition shape."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def integer_partitions(n: int, max_part: int | None = None) -> list[list[int]]:
    if n == 0:
        return [[]]
    if max_part is None or max_part > n:
        max_part = n
    out = []
    for first in range(max_part, 0, -1):
        for rest in integer_partitions(n - first, min(first, n - first) if n - first else 0):
            out.append([first] + rest)
    return out


def hook_lengths(shape: list[int]) -> list[int]:
    hooks = []
    for i, row_len in enumerate(shape):
        for j in range(row_len):
            below = sum(1 for lower_row_len in shape[i + 1 :] if lower_row_len > j)
            hooks.append((row_len - j) + below)
    return hooks


def log_syt_count(shape: list[int]) -> float:
    total_boxes = sum(shape)
    return math.lgamma(total_boxes + 1) - sum(math.log(hook) for hook in hook_lengths(shape))


def log_ssyt_count(shape: list[int], alphabet_size: int) -> float:
    total = 0.0
    for i, row_len in enumerate(shape, start=1):
        for j in range(1, row_len + 1):
            total += math.log(alphabet_size + j - i)
    total -= sum(math.log(hook) for hook in hook_lengths(shape))
    return total


class ShapeCountDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)[:, None]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ShapeCountMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        dim = input_dim
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--task", choices=["syt", "ssyt"], default="syt")
    parser.add_argument("--alphabet-size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=None)
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


def build_arrays(n: int, task: str, alphabet_size: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    shapes = integer_partitions(n)
    x = np.zeros((len(shapes), n), dtype=np.float32)
    y = np.zeros(len(shapes), dtype=np.float32)
    for idx, shape in enumerate(shapes):
        x[idx, : len(shape)] = np.array(shape, dtype=np.float32) / n
        if task == "syt":
            y[idx] = log_syt_count(shape)
        else:
            y[idx] = log_ssyt_count(shape, alphabet_size)
    return x, y, shapes


def split_indices(total: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    test_size = max(1, int(round(total * test_frac)))
    return order[:-test_size], order[-test_size:]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_abs = 0.0
    total_rel = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = criterion(pred, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch = x.shape[0]
        log_abs = (pred.detach() - y).abs()
        total_loss += float(loss.item()) * batch
        total_abs += float(log_abs.sum().item())
        total_rel += float(torch.expm1(log_abs).sum().item())
        total += batch
    return total_loss / total, total_abs / total, total_rel / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    x, y, shapes = build_arrays(args.n, args.task, args.alphabet_size)
    train_idx, test_idx = split_indices(len(x), args.test_frac, args.seed)
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() + 1e-8)
    y_scaled = (y - y_mean) / y_std

    train_loader = DataLoader(
        ShapeCountDataset(x[train_idx], y_scaled[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ShapeCountDataset(x[test_idx], y_scaled[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ShapeCountMLP(input_dim=args.n, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"task={args.task} n={args.n} partitions={len(shapes)} train={len(train_idx)} "
        f"test={len(test_idx)} device={device} parameters={sum(p.numel() for p in model.parameters())}"
    )
    print(f"target_log_mean={y_mean:.4f} target_log_std={y_std:.4f}")

    best_test_log_mae = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_scaled_mae, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_scaled_mae, _ = run_epoch(model, test_loader, criterion, device)
        test_log_mae = test_scaled_mae * y_std
        train_log_mae = train_scaled_mae * y_std
        if test_log_mae < best_test_log_mae:
            best_test_log_mae = test_log_mae
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.6f} test_loss={test_loss:.6f} "
                f"train_log_mae={train_log_mae:.4f} test_log_mae={test_log_mae:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.inference_mode():
        x_test = torch.as_tensor(x[test_idx], dtype=torch.float32, device=device)
        pred_scaled = model(x_test).cpu().numpy().reshape(-1)
    pred_log = pred_scaled * y_std + y_mean
    true_log = y[test_idx]
    log_abs = np.abs(pred_log - true_log)
    rel_factor = np.expm1(log_abs)

    print("final:")
    print(f"test_log_mae={float(log_abs.mean()):.6f}")
    print(f"test_log_rmse={float(np.sqrt(np.mean((pred_log - true_log) ** 2))):.6f}")
    print(f"median_multiplicative_error={float(np.exp(np.median(log_abs))):.6f}x")
    print(f"mean_relative_error={float(rel_factor.mean()):.6f}")

    print("\nexamples:")
    for local_idx in np.argsort(log_abs)[-5:][::-1]:
        global_idx = int(test_idx[local_idx])
        print(
            f"shape={shapes[global_idx]} true_log={true_log[local_idx]:.4f} "
            f"pred_log={pred_log[local_idx]:.4f} abs_log_err={log_abs[local_idx]:.4f}"
        )

    model_out = args.model_out or Path(f"models/shape_counts/{args.task}_n{args.n}_mlp.pt")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "n": args.n,
            "task": args.task,
            "alphabet_size": args.alphabet_size,
            "target_log_mean": y_mean,
            "target_log_std": y_std,
            "args": vars(args),
            "best_test_log_mae": best_test_log_mae,
        },
        model_out,
    )
    print(f"wrote {model_out}")


if __name__ == "__main__":
    main()
