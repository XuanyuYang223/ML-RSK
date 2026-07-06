#!/usr/bin/env python3
"""Train a small NumPy MLP to predict the RSK shape from a permutation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--model-out", type=Path, default=Path("models/shape_mlp.npz"))
    return parser.parse_args()


def init_weights(n: int, hidden: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / n), size=(n, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, hidden)),
        "b2": np.zeros(hidden),
        "w3": rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, n)),
        "b3": np.zeros(n),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ params["w1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["w2"] + params["b2"]
    a2 = relu(z2)
    y = a2 @ params["w3"] + params["b3"]
    return y, {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}


def train_step(params: dict[str, np.ndarray], x: np.ndarray, y_true: np.ndarray, lr: float) -> float:
    batch = x.shape[0]
    y_pred, cache = forward(params, x)
    diff = y_pred - y_true
    loss = float(np.mean(diff * diff))

    dy = (2.0 / diff.size) * diff
    dw3 = cache["a2"].T @ dy
    db3 = dy.sum(axis=0)
    da2 = dy @ params["w3"].T
    dz2 = da2 * relu_grad(cache["z2"])
    dw2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ params["w2"].T
    dz1 = da1 * relu_grad(cache["z1"])
    dw1 = cache["x"].T @ dz1
    db1 = dz1.sum(axis=0)

    for name, grad in {
        "w1": dw1,
        "b1": db1,
        "w2": dw2,
        "b2": db2,
        "w3": dw3,
        "b3": db3,
    }.items():
        params[name] -= lr * grad
    return loss


def evaluate(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, n: int) -> dict[str, float]:
    pred, _ = forward(params, x)
    pred_boxes = np.clip(np.rint(pred * n), 0, n)
    true_boxes = y * n
    return {
        "mse": float(np.mean((pred - y) ** 2)),
        "mae_rows": float(np.mean(np.abs(pred_boxes - true_boxes))),
        "mae_total_boxes": float(np.mean(np.sum(np.abs(pred_boxes - true_boxes), axis=1))),
    }


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.float64)
    shapes = data["shapes"].astype(np.float64)
    n = int(data["n"])

    x = (permutations - 1.0) / max(1.0, n - 1.0)
    y = shapes / n

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))
    split = int(len(x) * (1.0 - args.test_frac))
    train_idx, test_idx = order[:split], order[split:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    params = init_weights(n, args.hidden, rng)
    for epoch in range(1, args.epochs + 1):
        batch_order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(batch_order), args.batch_size):
            idx = batch_order[start : start + args.batch_size]
            losses.append(train_step(params, x_train[idx], y_train[idx], args.lr))
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            metrics = evaluate(params, x_test, y_test, n)
            print(
                f"epoch={epoch:04d} "
                f"train_mse={np.mean(losses):.6f} "
                f"test_mse={metrics['mse']:.6f} "
                f"test_row_mae={metrics['mae_rows']:.3f} "
                f"test_total_box_mae={metrics['mae_total_boxes']:.3f}"
            )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.model_out, **params, n=np.array(n, dtype=np.int64))
    print(f"wrote {args.model_out}")


if __name__ == "__main__":
    main()
