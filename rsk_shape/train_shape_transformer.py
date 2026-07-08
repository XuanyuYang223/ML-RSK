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
    def __init__(self, permutations: np.ndarray, shapes: np.ndarray, true_rows: np.ndarray | None = None) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)
        self.true_rows = torch.as_tensor(shapes if true_rows is None else true_rows, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.permutations.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.shapes[idx], self.true_rows[idx]


def inverse_permutations(permutations: np.ndarray) -> np.ndarray:
    inverse = np.zeros_like(permutations)
    positions = np.arange(1, permutations.shape[1] + 1, dtype=permutations.dtype)
    for row_idx, permutation in enumerate(permutations):
        inverse[row_idx, permutation - 1] = positions
    return inverse


def lehmer_codes(permutations: np.ndarray) -> np.ndarray:
    codes = np.zeros_like(permutations)
    n = permutations.shape[1]
    for i in range(n - 1):
        codes[:, i] = (permutations[:, i, None] > permutations[:, i + 1 :]).sum(axis=1)
    return codes


def encode_permutations(permutations: np.ndarray, representation: str) -> np.ndarray:
    if representation == "one_line":
        return permutations
    if representation == "inverse":
        return inverse_permutations(permutations)
    if representation == "lehmer":
        return lehmer_codes(permutations)
    raise ValueError(f"unknown permutation representation: {representation}")


def rows_to_deltas(shapes: np.ndarray) -> np.ndarray:
    next_rows = np.zeros_like(shapes)
    next_rows[:, :-1] = shapes[:, 1:]
    return shapes - next_rows


def deltas_to_rows(deltas: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(deltas, dims=[1]), dim=1), dims=[1])


def generate_partitions(n: int, max_part: int | None = None) -> list[list[int]]:
    if n == 0:
        return [[]]
    if max_part is None or max_part > n:
        max_part = n

    partitions = []
    for first in range(max_part, 0, -1):
        for rest in generate_partitions(n - first, first):
            partitions.append([first, *rest])
    return partitions


_PARTITION_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def partition_table(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    if key not in _PARTITION_CACHE:
        rows = []
        for partition in generate_partitions(n):
            padded = partition + [0] * (n - len(partition))
            rows.append(padded)
        _PARTITION_CACHE[key] = torch.tensor(rows, dtype=torch.long, device=device)
    return _PARTITION_CACHE[key]


def encode_shapes(shapes: np.ndarray, output_mode: str) -> np.ndarray:
    if output_mode == "rows":
        return shapes
    if output_mode == "deltas":
        return rows_to_deltas(shapes)
    raise ValueError(f"unknown output mode: {output_mode}")


def decode_predictions(pred: torch.Tensor, output_mode: str, inference: str) -> torch.Tensor:
    if output_mode == "deltas":
        rows = deltas_to_rows(pred)
    elif output_mode == "rows":
        rows = pred
    else:
        raise ValueError(f"unknown output mode: {output_mode}")

    if inference == "argmax":
        return rows
    if inference == "monotone":
        projected = rows.clone()
        for i in range(1, projected.shape[1]):
            projected[:, i] = torch.minimum(projected[:, i], projected[:, i - 1])
        return projected
    if inference == "partition":
        raise ValueError("partition inference needs logits; use predict_rows")
    raise ValueError(f"unknown inference mode: {inference}")


def predict_rows(logits: torch.Tensor, output_mode: str, inference: str) -> torch.Tensor:
    if inference != "partition":
        return decode_predictions(logits.argmax(dim=-1), output_mode, inference)
    if output_mode != "rows":
        raise ValueError("partition inference currently supports only rows output")

    n = logits.shape[1]
    partitions = partition_table(n, logits.device)
    scores = torch.zeros((logits.shape[0], partitions.shape[0]), dtype=logits.dtype, device=logits.device)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)[:, None]
    for row_idx in range(n):
        scores += logits[batch_idx, row_idx, partitions[:, row_idx][None, :]]
    best_idx = scores.argmax(dim=1)
    return partitions[best_idx]


class ShapeTransformer(nn.Module):
    def __init__(
        self,
        n: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        architecture: str = "encoder_decoder",
    ) -> None:
        super().__init__()
        self.n = n
        self.architecture = architecture
        self.perm_embedding = nn.Embedding(n + 1, d_model)
        self.input_pos_embedding = nn.Embedding(n, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if architecture == "encoder_decoder":
            self.shape_query_embedding = nn.Embedding(n, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        elif architecture != "encoder_only":
            raise ValueError(f"unknown architecture: {architecture}")

        self.classifier = nn.Linear(d_model, n + 1)

    def forward(self, permutations: torch.Tensor) -> torch.Tensor:
        batch_size, n = permutations.shape
        if n != self.n:
            raise ValueError(f"model was built for n={self.n}, got n={n}")

        positions = torch.arange(n, device=permutations.device)
        src = self.perm_embedding(permutations) + self.input_pos_embedding(positions)[None, :, :]
        memory = self.encoder(src)

        if self.architecture == "encoder_only":
            return self.classifier(memory)

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
    parser.add_argument("--perm-representation", choices=["one_line", "inverse", "lehmer"], default="one_line")
    parser.add_argument("--architecture", choices=["encoder_decoder", "encoder_only"], default="encoder_decoder")
    parser.add_argument("--output-mode", choices=["rows", "deltas"], default="rows")
    parser.add_argument("--inference", choices=["argmax", "monotone", "partition"], default="argmax")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=Path("models/rsk_shape/shape_transformer.pt"))
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


def split_indices(total: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if total < 2:
        raise ValueError("need at least two examples to make a train/test split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    test_size = max(1, int(round(len(order) * test_frac)))
    train_size = len(order) - test_size
    if train_size < 1:
        train_size = len(order) - 1
        test_size = 1
    return order[:train_size], order[train_size : train_size + test_size]


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    true_rows: torch.Tensor,
    output_mode: str,
    inference: str,
) -> dict[str, float]:
    pred = predict_rows(logits, output_mode, inference)
    correct = pred.eq(true_rows)
    row_acc = correct.float().mean().item()
    exact_acc = correct.all(dim=1).float().mean().item()
    row_mae = (pred - true_rows).abs().float().mean().item()
    total_box_mae = (pred.sum(dim=1) - true_rows.sum(dim=1)).abs().float().mean().item()
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
    output_mode: str = "rows",
    inference: str = "argmax",
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    metric_sums = {"row_acc": 0.0, "exact_acc": 0.0, "row_mae": 0.0, "total_box_mae": 0.0}

    for permutations, encoded_shapes, true_rows in loader:
        permutations = permutations.to(device)
        encoded_shapes = encoded_shapes.to(device)
        true_rows = true_rows.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(permutations)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), encoded_shapes.reshape(-1))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = permutations.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        metrics = compute_metrics(logits.detach(), encoded_shapes, true_rows, output_mode, inference)
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
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
        shapes = shapes[: args.max_samples]

    encoded_permutations = encode_permutations(permutations, args.perm_representation)
    encoded_shapes = encode_shapes(shapes, args.output_mode)

    train_idx, test_idx = split_indices(len(encoded_permutations), args.test_frac, args.seed)
    train_x = encoded_permutations[train_idx]
    train_y = encoded_shapes[train_idx]
    train_rows = shapes[train_idx]
    test_x = encoded_permutations[test_idx]
    test_y = encoded_shapes[test_idx]
    test_rows = shapes[test_idx]
    train_loader = DataLoader(
        RSKShapeDataset(train_x, train_y, train_rows),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        RSKShapeDataset(test_x, test_y, test_rows),
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
        architecture=args.architecture,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"dataset={args.dataset} n={n} train={len(train_x)} test={len(test_x)} "
        f"representation={args.perm_representation} architecture={args.architecture} "
        f"output={args.output_mode} inference={args.inference} "
        f"device={device} parameters={sum(p.numel() for p in model.parameters())}"
    )
    best_exact_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.output_mode, args.inference
        )
        test_loss, test_metrics = run_epoch(
            model, test_loader, criterion, device, None, args.output_mode, args.inference
        )
        if test_metrics["exact_acc"] > best_exact_acc:
            best_exact_acc = test_metrics["exact_acc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
