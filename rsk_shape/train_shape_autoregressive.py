#!/usr/bin/env python3
"""Train an autoregressive Transformer for permutation -> RSK shape."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_shape_transformer import encode_permutations, split_indices


class RSKShapeDataset(Dataset):
    def __init__(self, permutations: np.ndarray, shapes: np.ndarray) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.permutations.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.shapes[idx]


class AutoregressiveShapeTransformer(nn.Module):
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
        self.start_token = n + 1
        self.perm_embedding = nn.Embedding(n + 1, d_model)
        self.row_embedding = nn.Embedding(n + 2, d_model)
        self.input_pos_embedding = nn.Embedding(n, d_model)
        self.output_pos_embedding = nn.Embedding(n, d_model)

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

    def encode(self, permutations: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.n, device=permutations.device)
        src = self.perm_embedding(permutations) + self.input_pos_embedding(positions)[None, :, :]
        return self.encoder(src)

    def forward(self, permutations: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        if permutations.shape[1] != self.n or decoder_input.shape[1] != self.n:
            raise ValueError(f"model was built for n={self.n}")
        memory = self.encode(permutations)
        positions = torch.arange(self.n, device=permutations.device)
        tgt = self.row_embedding(decoder_input) + self.output_pos_embedding(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.full((self.n, self.n), float("-inf"), device=permutations.device),
            diagonal=1,
        )
        decoded = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.classifier(decoded)

    def greedy_decode(self, permutations: torch.Tensor, constrained: bool) -> torch.Tensor:
        batch_size = permutations.shape[0]
        device = permutations.device
        memory = self.encode(permutations)
        generated = torch.zeros((batch_size, self.n), dtype=torch.long, device=device)
        decoder_input = torch.full((batch_size, self.n), self.start_token, dtype=torch.long, device=device)
        remaining = torch.full((batch_size,), self.n, dtype=torch.long, device=device)
        previous = torch.full((batch_size,), self.n, dtype=torch.long, device=device)

        for row_idx in range(self.n):
            positions = torch.arange(self.n, device=device)
            tgt = self.row_embedding(decoder_input) + self.output_pos_embedding(positions)[None, :, :]
            causal_mask = torch.triu(torch.full((self.n, self.n), float("-inf"), device=device), diagonal=1)
            logits = self.decoder(tgt, memory, tgt_mask=causal_mask)
            row_logits = self.classifier(logits)[:, row_idx, :]

            if constrained:
                row_logits = mask_invalid_rows(row_logits, previous, remaining, self.n - row_idx)

            next_row = row_logits.argmax(dim=1)
            generated[:, row_idx] = next_row
            if row_idx + 1 < self.n:
                decoder_input[:, row_idx + 1] = next_row
            remaining = remaining - next_row
            previous = next_row

        return generated


def mask_invalid_rows(logits: torch.Tensor, previous: torch.Tensor, remaining: torch.Tensor, slots_left: int) -> torch.Tensor:
    values = torch.arange(logits.shape[1], device=logits.device)[None, :]
    valid = values <= previous[:, None]
    valid &= values <= remaining[:, None]
    if slots_left == 1:
        valid &= values == remaining[:, None]
    else:
        valid &= (remaining[:, None] - values) <= (slots_left - 1) * values
    return logits.masked_fill(~valid, float("-inf"))


def make_decoder_input(shapes: torch.Tensor, start_token: int) -> torch.Tensor:
    decoder_input = torch.empty_like(shapes)
    decoder_input[:, 0] = start_token
    decoder_input[:, 1:] = shapes[:, :-1]
    return decoder_input


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    correct = pred.eq(target)
    return {
        "row_acc": correct.float().mean().item(),
        "exact_acc": correct.all(dim=1).float().mean().item(),
        "row_mae": (pred - target).abs().float().mean().item(),
        "total_box_mae": (pred.sum(dim=1) - target.sum(dim=1)).abs().float().mean().item(),
    }


def run_epoch(
    model: AutoregressiveShapeTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    constrained: bool,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    metric_sums = {"row_acc": 0.0, "exact_acc": 0.0, "row_mae": 0.0, "total_box_mae": 0.0}

    for permutations, shapes in loader:
        permutations = permutations.to(device)
        shapes = shapes.to(device)
        decoder_input = make_decoder_input(shapes, model.start_token)

        with torch.set_grad_enabled(is_train):
            logits = model(permutations, decoder_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), shapes.reshape(-1))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        with torch.inference_mode():
            if is_train:
                pred = logits.argmax(dim=-1)
            else:
                pred = model.greedy_decode(permutations, constrained=constrained)
            metrics = compute_metrics(pred, shapes)

        batch_size = permutations.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    return total_loss / total_examples, {key: value / total_examples for key, value in metric_sums.items()}


def evaluate_dataset(
    model: AutoregressiveShapeTransformer,
    permutations: np.ndarray,
    shapes: np.ndarray,
    batch_size: int,
    device: torch.device,
    constrained: bool,
) -> dict[str, float]:
    loader = DataLoader(RSKShapeDataset(permutations, shapes), batch_size=batch_size, shuffle=False)
    metric_sums = {"row_acc": 0.0, "exact_acc": 0.0, "row_mae": 0.0, "total_box_mae": 0.0}
    total_examples = 0
    model.eval()
    with torch.inference_mode():
        for permutations_batch, shapes_batch in loader:
            permutations_batch = permutations_batch.to(device)
            shapes_batch = shapes_batch.to(device)
            pred = model.greedy_decode(permutations_batch, constrained=constrained)
            metrics = compute_metrics(pred, shapes_batch)
            batch_size_actual = permutations_batch.shape[0]
            total_examples += batch_size_actual
            for key, value in metrics.items():
                metric_sums[key] += value * batch_size_actual
    return {key: value / total_examples for key, value in metric_sums.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--eval-dataset", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--perm-representation", choices=["one_line", "inverse", "lehmer"], default="lehmer")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--unconstrained", action="store_true")
    parser.add_argument("--model-out", type=Path, default=Path("models/rsk_shape/shape_autoregressive.pt"))
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
    constrained = not args.unconstrained

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    shapes = data["shapes"].astype(np.int64)
    n = int(data["n"])
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
        shapes = shapes[: args.max_samples]
    encoded_permutations = encode_permutations(permutations, args.perm_representation)

    train_idx, test_idx = split_indices(len(encoded_permutations), args.test_frac, args.seed)
    train_loader = DataLoader(
        RSKShapeDataset(encoded_permutations[train_idx], shapes[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        RSKShapeDataset(encoded_permutations[test_idx], shapes[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = AutoregressiveShapeTransformer(
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
        f"dataset={args.dataset} n={n} train={len(train_idx)} test={len(test_idx)} "
        f"representation={args.perm_representation} constrained={constrained} "
        f"device={device} parameters={sum(p.numel() for p in model.parameters())}"
    )
    best_score = (-1.0, -1.0)
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, constrained)
        test_loss, test_metrics = run_epoch(model, test_loader, criterion, device, None, constrained)
        score = (test_metrics["exact_acc"], test_metrics["row_acc"])
        if score > best_score:
            best_score = score
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

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.eval_dataset is not None:
        eval_data = np.load(args.eval_dataset)
        eval_permutations = encode_permutations(
            eval_data["permutations"].astype(np.int64),
            args.perm_representation,
        )
        eval_shapes = eval_data["shapes"].astype(np.int64)
        eval_metrics = evaluate_dataset(model, eval_permutations, eval_shapes, args.batch_size, device, constrained)
        print(
            f"eval_dataset={args.eval_dataset} "
            f"eval_row_acc={eval_metrics['row_acc']:.4f} "
            f"eval_exact_acc={eval_metrics['exact_acc']:.4f} "
            f"eval_row_mae={eval_metrics['row_mae']:.4f} "
            f"eval_total_box_mae={eval_metrics['total_box_mae']:.4f}"
        )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "n": n,
            "args": vars(args),
            "best_exact_acc": best_score[0],
            "best_row_acc": best_score[1],
        },
        args.model_out,
    )
    print(f"wrote {args.model_out}")


if __name__ == "__main__":
    main()
