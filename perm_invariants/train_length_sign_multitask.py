#!/usr/bin/env python3
"""Train a Transformer with joint Coxeter length regression and sign classification."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_perm_invariants_transformer import encode_permutations, inversion_counts, pick_device, set_seed, split_arrays


class LengthSignDataset(Dataset):
    def __init__(self, permutations: np.ndarray, lengths: np.ndarray, signs: np.ndarray) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.float32)[:, None]
        self.signs = torch.as_tensor(signs, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.permutations.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.lengths[idx], self.signs[idx]


class LengthSignTransformer(nn.Module):
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
        self.norm = nn.LayerNorm(d_model)
        self.length_head = nn.Linear(d_model, 1)
        self.sign_head = nn.Linear(d_model, 2)

    def forward(self, permutations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n = permutations.shape
        if n != self.n:
            raise ValueError(f"model was built for n={self.n}, got n={n}")
        token_embeddings = self.token_embedding(permutations)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, token_embeddings], dim=1)
        positions = torch.arange(n + 1, device=permutations.device)
        x = x + self.pos_embedding(positions)[None, :, :]
        pooled = self.norm(self.encoder(x)[:, 0])
        return self.length_head(pooled), self.sign_head(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--perm-representation", choices=["one_line", "lehmer"], default="one_line")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--length-loss-weight", type=float, default=1.0)
    parser.add_argument("--sign-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=Path("models/perm_invariants/length_sign_multitask.pt"))
    return parser.parse_args()


def run_epoch(
    model: LengthSignTransformer,
    loader: DataLoader,
    device: torch.device,
    max_len: int,
    length_criterion: nn.Module,
    sign_criterion: nn.Module,
    length_loss_weight: float,
    sign_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_len_abs = 0.0
    total_sign_correct = 0.0
    total = 0
    for permutations, lengths, signs in loader:
        permutations = permutations.to(device)
        lengths = lengths.to(device)
        signs = signs.to(device)
        with torch.set_grad_enabled(is_train):
            pred_lengths, sign_logits = model(permutations)
            length_loss = length_criterion(pred_lengths, lengths)
            sign_loss = sign_criterion(sign_logits, signs)
            loss = length_loss_weight * length_loss + sign_loss_weight * sign_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch = permutations.shape[0]
        pred_len = pred_lengths.detach().reshape(-1).clamp(0.0, 1.0) * max_len
        true_len = lengths.reshape(-1) * max_len
        total_loss += float(loss.item()) * batch
        total_len_abs += float((pred_len - true_len).abs().sum().item())
        total_sign_correct += float(sign_logits.detach().argmax(dim=-1).eq(signs).sum().item())
        total += batch
    return total_loss / total, {"length_mae": total_len_abs / total, "sign_acc": total_sign_correct / total}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    n = int(data["n"])
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
    max_len = n * (n - 1) // 2
    lengths_int = inversion_counts(permutations)
    lengths = lengths_int.astype(np.float32) / max_len
    signs = lengths_int % 2
    encoded = encode_permutations(permutations, args.perm_representation)

    train_x, train_lengths, test_x, test_lengths = split_arrays(encoded, lengths, args.test_frac, args.seed)
    _, train_signs, _, test_signs = split_arrays(encoded, signs, args.test_frac, args.seed)
    train_loader = DataLoader(LengthSignDataset(train_x, train_lengths, train_signs), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(LengthSignDataset(test_x, test_lengths, test_signs), batch_size=args.batch_size, shuffle=False)

    model = LengthSignTransformer(
        n=n,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    length_criterion = nn.MSELoss()
    sign_criterion = nn.CrossEntropyLoss()

    print(
        f"task=length_sign representation={args.perm_representation} dataset={args.dataset} "
        f"n={n} train={len(train_x)} test={len(test_x)} device={device} "
        f"parameters={sum(p.numel() for p in model.parameters())}"
    )

    best_sign_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            device,
            max_len,
            length_criterion,
            sign_criterion,
            args.length_loss_weight,
            args.sign_loss_weight,
            optimizer,
        )
        test_loss, test_metrics = run_epoch(
            model,
            test_loader,
            device,
            max_len,
            length_criterion,
            sign_criterion,
            args.length_loss_weight,
            args.sign_loss_weight,
        )
        if test_metrics["sign_acc"] > best_sign_acc:
            best_sign_acc = test_metrics["sign_acc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"train_length_mae={train_metrics['length_mae']:.3f} test_length_mae={test_metrics['length_mae']:.3f} "
            f"train_sign_acc={train_metrics['sign_acc']:.4f} test_sign_acc={test_metrics['sign_acc']:.4f}"
        )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "n": n,
            "args": vars(args),
            "best_sign_acc": best_sign_acc,
        },
        args.model_out,
    )
    print(f"wrote {args.model_out}")


if __name__ == "__main__":
    main()
