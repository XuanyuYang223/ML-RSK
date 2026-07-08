#!/usr/bin/env python3
"""Evaluate an autoregressive shape model by scoring every partition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_shape_autoregressive import AutoregressiveShapeTransformer, compute_metrics, make_decoder_input
from train_shape_transformer import encode_permutations, generate_partitions


class RSKShapeDataset(Dataset):
    def __init__(self, permutations: np.ndarray, shapes: np.ndarray) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.permutations.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.shapes[idx]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def partition_tensor(n: int, device: torch.device) -> torch.Tensor:
    rows = []
    for partition in generate_partitions(n):
        rows.append(partition + [0] * (n - len(partition)))
    return torch.tensor(rows, dtype=torch.long, device=device)


def build_model(checkpoint: dict, n: int, device: torch.device) -> AutoregressiveShapeTransformer:
    train_args = checkpoint.get("args", {})
    model = AutoregressiveShapeTransformer(
        n=n,
        d_model=int(train_args.get("d_model", 64)),
        num_heads=int(train_args.get("num_heads", 4)),
        num_layers=int(train_args.get("num_layers", 2)),
        dim_feedforward=int(train_args.get("dim_feedforward", 256)),
        dropout=float(train_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def score_partition_chunk(
    model: AutoregressiveShapeTransformer,
    memory: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    batch_size = memory.shape[0]
    candidate_count = candidates.shape[0]
    expanded_memory = memory[:, None, :, :].expand(-1, candidate_count, -1, -1)
    expanded_memory = expanded_memory.reshape(batch_size * candidate_count, model.n, -1)
    expanded_candidates = candidates[None, :, :].expand(batch_size, -1, -1)
    expanded_candidates = expanded_candidates.reshape(batch_size * candidate_count, model.n)
    decoder_input = make_decoder_input(expanded_candidates, model.start_token)

    positions = torch.arange(model.n, device=memory.device)
    tgt = model.row_embedding(decoder_input) + model.output_pos_embedding(positions)[None, :, :]
    causal_mask = torch.triu(torch.full((model.n, model.n), float("-inf"), device=memory.device), diagonal=1)
    decoded = model.decoder(tgt, expanded_memory, tgt_mask=causal_mask)
    logits = model.classifier(decoded)
    log_probs = logits.log_softmax(dim=-1)
    token_scores = log_probs.gather(dim=-1, index=expanded_candidates[:, :, None]).squeeze(-1)
    return token_scores.sum(dim=1).reshape(batch_size, candidate_count)


def predict_by_partition_scoring(
    model: AutoregressiveShapeTransformer,
    permutations: torch.Tensor,
    candidates: torch.Tensor,
    candidate_chunk_size: int,
) -> torch.Tensor:
    memory = model.encode(permutations)
    best_scores = torch.full((permutations.shape[0],), float("-inf"), device=permutations.device)
    best_indices = torch.zeros((permutations.shape[0],), dtype=torch.long, device=permutations.device)

    for start in range(0, candidates.shape[0], candidate_chunk_size):
        end = min(start + candidate_chunk_size, candidates.shape[0])
        scores = score_partition_chunk(model, memory, candidates[start:end])
        chunk_scores, chunk_indices = scores.max(dim=1)
        improved = chunk_scores > best_scores
        best_scores = torch.where(improved, chunk_scores, best_scores)
        best_indices = torch.where(improved, chunk_indices + start, best_indices)

    return candidates[best_indices]


def evaluate(
    model: AutoregressiveShapeTransformer,
    loader: DataLoader,
    candidates: torch.Tensor,
    candidate_chunk_size: int,
    device: torch.device,
) -> dict[str, float]:
    metric_sums = {"row_acc": 0.0, "exact_acc": 0.0, "row_mae": 0.0, "total_box_mae": 0.0}
    total_examples = 0

    with torch.inference_mode():
        for permutations, shapes in loader:
            permutations = permutations.to(device)
            shapes = shapes.to(device)
            pred = predict_by_partition_scoring(model, permutations, candidates, candidate_chunk_size)
            metrics = compute_metrics(pred, shapes)
            batch_size = permutations.shape[0]
            total_examples += batch_size
            for key, value in metrics.items():
                metric_sums[key] += value * batch_size

    return {key: value / total_examples for key, value in metric_sums.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-chunk-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    data = np.load(args.dataset)
    permutations = data["permutations"].astype(np.int64)
    shapes = data["shapes"].astype(np.int64)
    n = int(data["n"])
    if args.max_samples is not None:
        permutations = permutations[: args.max_samples]
        shapes = shapes[: args.max_samples]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_n = int(checkpoint.get("n", n))
    if checkpoint_n != n:
        raise ValueError(f"checkpoint n={checkpoint_n} does not match dataset n={n}")
    train_args = checkpoint.get("args", {})
    representation = str(train_args.get("perm_representation", "lehmer"))
    encoded_permutations = encode_permutations(permutations, representation)

    model = build_model(checkpoint, n, device)
    candidates = partition_tensor(n, device)
    loader = DataLoader(
        RSKShapeDataset(encoded_permutations, shapes),
        batch_size=args.batch_size,
        shuffle=False,
    )
    metrics = evaluate(model, loader, candidates, args.candidate_chunk_size, device)

    print(f"dataset={args.dataset}")
    print(f"checkpoint={args.checkpoint}")
    print(
        f"n={n} examples={len(permutations)} representation={representation} "
        f"partitions={candidates.shape[0]} batch_size={args.batch_size} "
        f"candidate_chunk_size={args.candidate_chunk_size} device={device}"
    )
    print(f"row_acc={metrics['row_acc']:.4f}")
    print(f"exact_acc={metrics['exact_acc']:.4f}")
    print(f"row_mae={metrics['row_mae']:.4f}")
    print(f"total_box_mae={metrics['total_box_mae']:.4f}")


if __name__ == "__main__":
    main()
