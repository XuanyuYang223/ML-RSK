#!/usr/bin/env python3
"""Train a Transformer on permutation invariants.

Supported tasks:
  - length: Coxeter length / inversion number, classified as 0..n(n-1)/2
  - sign: permutation parity, classified as 0 for even and 1 for odd
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class PermInvariantDataset(Dataset):
    def __init__(self, permutations: np.ndarray, labels: np.ndarray, regression: bool) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        label_dtype = torch.float32 if regression else torch.long
        self.labels = torch.as_tensor(labels, dtype=label_dtype)

    def __len__(self) -> int:
        return int(self.permutations.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.labels[idx]


class PermInvariantTransformer(nn.Module):
    def __init__(
        self,
        n: int,
        output_dim: int,
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
            nn.Linear(d_model, output_dim),
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


def inversion_counts(permutations: np.ndarray) -> np.ndarray:
    counts = np.zeros(permutations.shape[0], dtype=np.int64)
    n = permutations.shape[1]
    for i in range(n - 1):
        counts += (permutations[:, i, None] > permutations[:, i + 1 :]).sum(axis=1)
    return counts


def lehmer_codes(permutations: np.ndarray) -> np.ndarray:
    codes = np.zeros_like(permutations)
    n = permutations.shape[1]
    for i in range(n - 1):
        codes[:, i] = (permutations[:, i, None] > permutations[:, i + 1 :]).sum(axis=1)
    return codes


def encode_permutations(permutations: np.ndarray, representation: str) -> np.ndarray:
    if representation == "one_line":
        return permutations
    if representation == "lehmer":
        return lehmer_codes(permutations)
    raise ValueError(f"unknown permutation representation: {representation}")


def make_labels(permutations: np.ndarray, task: str, target_mode: str) -> tuple[np.ndarray, int]:
    n = permutations.shape[1]
    lengths = inversion_counts(permutations)
    if task == "length":
        if target_mode == "regression":
            max_len = n * (n - 1) // 2
            return (lengths.astype(np.float32) / max_len)[:, None], 1
        return lengths, n * (n - 1) // 2 + 1
    if task == "sign":
        return lengths % 2, 2
    raise ValueError(f"unknown task: {task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--task", choices=["length", "sign"], default="length")
    parser.add_argument("--target-mode", choices=["classification", "regression"], default="classification")
    parser.add_argument("--perm-representation", choices=["one_line", "lehmer"], default="one_line")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
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


def split_arrays(
    permutations: np.ndarray,
    labels: np.ndarray,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(permutations))
    test_size = max(1, int(round(len(order) * test_frac)))
    train_size = max(1, len(order) - test_size)
    train_idx = order[:train_size]
    test_idx = order[train_size:]
    return permutations[train_idx], labels[train_idx], permutations[test_idx], labels[test_idx]


def compute_metrics(
    output: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    target_mode: str,
    max_len: int,
) -> dict[str, float]:
    if target_mode == "regression":
        pred_len = output.reshape(-1).clamp(0.0, 1.0) * max_len
        true_len = labels.reshape(-1) * max_len
        rounded = pred_len.round().long()
        true_rounded = true_len.round().long()
        return {
            "mae": (pred_len - true_len).abs().float().mean().item(),
            "rmse": torch.sqrt(torch.mean((pred_len - true_len) ** 2)).item(),
            "rounded_acc": rounded.eq(true_rounded).float().mean().item(),
        }

    pred = output.argmax(dim=-1)
    acc = pred.eq(labels).float().mean().item()
    metrics = {"acc": acc}
    if task == "length":
        metrics["mae"] = (pred - labels).abs().float().mean().item()
    return metrics


def run_epoch(
    model: PermInvariantTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    target_mode: str,
    max_len: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    metric_sums = {"acc": 0.0, "mae": 0.0, "rmse": 0.0, "rounded_acc": 0.0}

    for permutations, labels in loader:
        permutations = permutations.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            output = model(permutations)
            if target_mode == "regression":
                loss = criterion(output, labels)
            else:
                loss = criterion(output, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = permutations.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        metrics = compute_metrics(output.detach(), labels, task, target_mode, max_len)
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    mean_metrics = {key: value / total_examples for key, value in metric_sums.items()}
    if target_mode == "regression":
        mean_metrics.pop("acc", None)
    else:
        mean_metrics.pop("rmse", None)
        mean_metrics.pop("rounded_acc", None)
        if task != "length":
            mean_metrics.pop("mae", None)
    return total_loss / total_examples, mean_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    n = int(data["n"])
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
    if args.task != "length" and args.target_mode != "classification":
        raise ValueError("--target-mode regression is only supported for --task length")
    labels, output_dim = make_labels(permutations, args.task, args.target_mode)
    encoded_permutations = encode_permutations(permutations, args.perm_representation)
    max_len = n * (n - 1) // 2

    train_x, train_y, test_x, test_y = split_arrays(encoded_permutations, labels, args.test_frac, args.seed)
    is_regression = args.target_mode == "regression"
    train_loader = DataLoader(
        PermInvariantDataset(train_x, train_y, is_regression),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        PermInvariantDataset(test_x, test_y, is_regression),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PermInvariantTransformer(
        n=n,
        output_dim=output_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"task={args.task} target_mode={args.target_mode} representation={args.perm_representation} "
        f"dataset={args.dataset} n={n} outputs={output_dim} "
        f"train={len(train_x)} test={len(test_x)} device={device} "
        f"parameters={sum(p.numel() for p in model.parameters())}"
    )

    best_score = float("inf") if is_regression else -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, args.task, args.target_mode, max_len, optimizer
        )
        test_loss, test_metrics = run_epoch(
            model, test_loader, criterion, device, args.task, args.target_mode, max_len
        )
        score = test_metrics["mae"] if is_regression else test_metrics["acc"]
        is_better = score < best_score if is_regression else score > best_score
        if is_better:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if is_regression:
            metric_text = (
                f"test_mae={test_metrics['mae']:.3f} test_rmse={test_metrics['rmse']:.3f} "
                f"test_rounded_acc={test_metrics['rounded_acc']:.4f}"
            )
            train_metric_text = f"train_mae={train_metrics['mae']:.3f}"
        else:
            metric_text = f"test_acc={test_metrics['acc']:.4f}"
            train_metric_text = f"train_acc={train_metrics['acc']:.4f}"
        if args.task == "length" and not is_regression:
            metric_text += f" test_mae={test_metrics['mae']:.3f}"
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"{train_metric_text} {metric_text}"
        )

    model_out = args.model_out or Path(f"models/perm_invariants/{args.task}_{args.target_mode}_transformer.pt")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "n": n,
            "task": args.task,
            "target_mode": args.target_mode,
            "perm_representation": args.perm_representation,
            "output_dim": output_dim,
            "args": vars(args),
            "best_score": best_score,
        },
        model_out,
    )
    print(f"wrote {model_out}")


if __name__ == "__main__":
    main()
