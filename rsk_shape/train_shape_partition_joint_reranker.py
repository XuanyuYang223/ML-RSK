#!/usr/bin/env python3
"""Train a joint/listwise reranker over partition-classifier top-k candidates."""

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


class RerankerDataset(Dataset):
    def __init__(self, permutations: np.ndarray, labels: np.ndarray, shapes: np.ndarray) -> None:
        self.permutations = torch.as_tensor(permutations, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.shapes = torch.as_tensor(shapes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.permutations[idx], self.labels[idx], self.shapes[idx]


class JointPartitionReranker(nn.Module):
    def __init__(
        self,
        n: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        candidate_layers: int,
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

        self.row_embedding = nn.Embedding(n + 1, d_model)
        self.row_pos_embedding = nn.Embedding(n, d_model)
        self.rank_projection = nn.Linear(1, d_model)
        self.logprob_projection = nn.Linear(1, d_model)
        self.partition_projection = nn.Sequential(
            nn.LayerNorm(d_model + n),
            nn.Linear(d_model + n, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.candidate_encoder = nn.TransformerEncoder(candidate_layer, num_layers=candidate_layers)
        self.scorer = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )
        self.classifier_logprob_weight = nn.Parameter(torch.tensor(1.0))

    def load_classifier_encoder(self, classifier_state: dict[str, torch.Tensor]) -> None:
        own_state = self.state_dict()
        keys = ["cls_token", "token_embedding.weight", "pos_embedding.weight"]
        keys += [key for key in classifier_state if key.startswith("encoder.")]
        copied = {}
        for key in keys:
            if key in classifier_state and key in own_state and classifier_state[key].shape == own_state[key].shape:
                copied[key] = classifier_state[key]
        self.load_state_dict(copied, strict=False)

    def encode_permutation(self, permutations: torch.Tensor) -> torch.Tensor:
        batch_size, n = permutations.shape
        if n != self.n:
            raise ValueError(f"model was built for n={self.n}, got n={n}")
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, self.token_embedding(permutations)], dim=1)
        positions = torch.arange(n + 1, device=permutations.device)
        encoded = self.encoder(x + self.pos_embedding(positions)[None, :, :])
        return encoded[:, 0]

    def encode_partitions(self, candidate_shapes: torch.Tensor) -> torch.Tensor:
        batch_size, top_k, n = candidate_shapes.shape
        if n != self.n:
            raise ValueError(f"model was built for n={self.n}, got candidate length={n}")
        row_positions = torch.arange(n, device=candidate_shapes.device)
        row_tokens = self.row_embedding(candidate_shapes) + self.row_pos_embedding(row_positions)[None, None, :, :]
        embedded_rows = row_tokens.mean(dim=2)
        normalized_rows = candidate_shapes.float() / float(n)
        return self.partition_projection(torch.cat([embedded_rows, normalized_rows], dim=-1))

    def forward(
        self,
        permutations: torch.Tensor,
        candidate_shapes: torch.Tensor,
        classifier_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, top_k, _n = candidate_shapes.shape
        perm_repr = self.encode_permutation(permutations)
        part_repr = self.encode_partitions(candidate_shapes)

        ranks = torch.linspace(0.0, 1.0, steps=top_k, device=permutations.device)[None, :, None]
        ranks = ranks.expand(batch_size, -1, -1)
        candidate_tokens = (
            part_repr
            + perm_repr[:, None, :]
            + self.rank_projection(ranks)
            + self.logprob_projection(classifier_log_probs[..., None])
        )
        list_repr = self.candidate_encoder(candidate_tokens)
        expanded_perm = perm_repr[:, None, :].expand_as(list_repr)
        features = torch.cat(
            [
                list_repr,
                part_repr,
                expanded_perm * list_repr,
                (expanded_perm - part_repr).abs(),
            ],
            dim=-1,
        )
        learned_scores = self.scorer(features).squeeze(-1)
        return learned_scores + self.classifier_logprob_weight * classifier_log_probs


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


def build_classifier(checkpoint: dict, n: int, device: torch.device) -> PartitionClassifier:
    train_args = checkpoint.get("args", {})
    partitions = checkpoint["partitions"]
    model = PartitionClassifier(
        n=n,
        num_classes=int(checkpoint.get("num_classes", len(partitions))),
        d_model=int(train_args.get("d_model", 128)),
        num_heads=int(train_args.get("num_heads", 4)),
        num_layers=int(train_args.get("num_layers", 2)),
        dim_feedforward=int(train_args.get("dim_feedforward", 512)),
        dropout=float(train_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def compute_shape_metrics(
    pred_shapes: torch.Tensor,
    true_shapes: torch.Tensor,
    pred_labels: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    correct_rows = pred_shapes.eq(true_shapes)
    return {
        "exact_acc": pred_labels.eq(labels).float().mean().item(),
        "row_acc": correct_rows.float().mean().item(),
        "row_mae": (pred_shapes - true_shapes).abs().float().mean().item(),
        "total_box_mae": (pred_shapes.sum(dim=1) - true_shapes.sum(dim=1)).abs().float().mean().item(),
    }


def get_topk(
    classifier: PartitionClassifier,
    permutations: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        classifier_log_probs = classifier(permutations).log_softmax(dim=1)
        return classifier_log_probs.topk(k=top_k, dim=1)


def run_epoch(
    reranker: JointPartitionReranker,
    classifier: PartitionClassifier,
    loader: DataLoader,
    candidates: torch.Tensor,
    top_k: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    reranker.train(is_train)
    total_loss = 0.0
    total_examples = 0
    total_hit_examples = 0
    metric_sums = {
        "classifier_argmax_exact_acc": 0.0,
        "reranker_exact_acc": 0.0,
        "topk_oracle_acc": 0.0,
        "row_acc": 0.0,
        "row_mae": 0.0,
        "total_box_mae": 0.0,
    }

    for permutations, labels, true_shapes in loader:
        permutations = permutations.to(device)
        labels = labels.to(device)
        true_shapes = true_shapes.to(device)

        top_scores, top_labels = get_topk(classifier, permutations, top_k)
        top_shapes = candidates[top_labels]
        hit_mask = top_labels.eq(labels[:, None])
        has_hit = hit_mask.any(dim=1)

        with torch.set_grad_enabled(is_train):
            scores = reranker(permutations, top_shapes, top_scores)
            if has_hit.any():
                loss = nn.functional.cross_entropy(scores[has_hit], hit_mask[has_hit].float().argmax(dim=1))
            else:
                loss = scores.sum() * 0.0
            if optimizer is not None and has_hit.any():
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(reranker.parameters(), max_norm=1.0)
                optimizer.step()

        pred_choice = scores.detach().argmax(dim=1)
        pred_labels = top_labels.gather(1, pred_choice[:, None]).squeeze(1)
        pred_shapes = candidates[pred_labels]
        classifier_pred_labels = top_labels[:, 0]

        batch_size = labels.shape[0]
        hit_count = int(has_hit.sum().item())
        total_examples += batch_size
        total_hit_examples += hit_count
        total_loss += float(loss.item()) * max(hit_count, 1)

        metrics = compute_shape_metrics(pred_shapes, true_shapes, pred_labels, labels)
        metric_sums["classifier_argmax_exact_acc"] += classifier_pred_labels.eq(labels).float().mean().item() * batch_size
        metric_sums["reranker_exact_acc"] += metrics["exact_acc"] * batch_size
        metric_sums["topk_oracle_acc"] += has_hit.float().mean().item() * batch_size
        metric_sums["row_acc"] += metrics["row_acc"] * batch_size
        metric_sums["row_mae"] += metrics["row_mae"] * batch_size
        metric_sums["total_box_mae"] += metrics["total_box_mae"] * batch_size

    metrics = {key: value / total_examples for key, value in metric_sums.items()}
    metrics["candidate_hit_rate"] = total_hit_examples / total_examples
    return total_loss / max(total_hit_examples, 1), metrics


def evaluate(
    reranker: JointPartitionReranker,
    classifier: PartitionClassifier,
    permutations: np.ndarray,
    labels: np.ndarray,
    shapes: np.ndarray,
    candidates: torch.Tensor,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(RerankerDataset(permutations, labels, shapes), batch_size=batch_size, shuffle=False)
    _loss, metrics = run_epoch(reranker, classifier, loader, candidates, top_k, device, optimizer=None)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--candidate-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--freeze-perm-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-out", type=Path, default=Path("experiments/rsk_shape_joint_reranker/top50_model.pt"))
    parser.add_argument("--results-out", type=Path, default=Path("experiments/rsk_shape_joint_reranker/top50_results.json"))
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
    if args.top_k > candidates_np.shape[0]:
        raise ValueError(f"top-k={args.top_k} exceeds num classes={candidates_np.shape[0]}")
    labels = shape_labels(shapes, candidates_np)
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
    eval_encoded = encode_permutations(eval_permutations, representation)

    classifier = build_classifier(checkpoint, n, device)
    candidates = torch.as_tensor(candidates_np, dtype=torch.long, device=device)
    reranker = JointPartitionReranker(
        n=n,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        candidate_layers=args.candidate_layers,
    ).to(device)
    reranker.load_classifier_encoder(checkpoint["model_state_dict"])
    if args.freeze_perm_encoder:
        for module in [reranker.token_embedding, reranker.pos_embedding, reranker.encoder]:
            for param in module.parameters():
                param.requires_grad_(False)
        reranker.cls_token.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [param for param in reranker.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        RerankerDataset(encoded[train_idx], labels[train_idx], shapes[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        RerankerDataset(encoded[valid_idx], labels[valid_idx], shapes[valid_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(
        f"dataset={args.dataset} eval_dataset={args.eval_dataset} n={n} classes={candidates_np.shape[0]} "
        f"train={len(train_idx)} valid={len(valid_idx)} representation={representation} top_k={args.top_k} "
        f"device={device} trainable_parameters={sum(p.numel() for p in reranker.parameters() if p.requires_grad)}",
        flush=True,
    )

    best_score = (-1.0, -1.0)
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            reranker, classifier, train_loader, candidates, args.top_k, device, optimizer
        )
        valid_loss, valid_metrics = run_epoch(
            reranker, classifier, valid_loader, candidates, args.top_k, device, optimizer=None
        )
        score = (valid_metrics["reranker_exact_acc"], valid_metrics["row_acc"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in reranker.state_dict().items()}
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train": train_metrics,
            "valid": valid_metrics,
            "classifier_logprob_weight": float(reranker.classifier_logprob_weight.detach().cpu().item()),
        }
        history.append(row)
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"train_hit={train_metrics['candidate_hit_rate']:.4f} valid_hit={valid_metrics['candidate_hit_rate']:.4f} "
            f"valid_classifier_exact={valid_metrics['classifier_argmax_exact_acc']:.4f} "
            f"valid_reranker_exact={valid_metrics['reranker_exact_acc']:.4f} "
            f"valid_oracle={valid_metrics['topk_oracle_acc']:.4f} "
            f"valid_row_acc={valid_metrics['row_acc']:.4f} "
            f"valid_row_mae={valid_metrics['row_mae']:.4f} "
            f"logprob_weight={row['classifier_logprob_weight']:.4f}",
            flush=True,
        )

    if best_state is not None:
        reranker.load_state_dict(best_state)

    eval_metrics = evaluate(
        reranker,
        classifier,
        eval_encoded,
        eval_labels,
        eval_shapes,
        candidates,
        args.top_k,
        args.batch_size,
        device,
    )
    print(
        f"eval_dataset={args.eval_dataset} "
        f"eval_classifier_exact={eval_metrics['classifier_argmax_exact_acc']:.4f} "
        f"eval_reranker_exact={eval_metrics['reranker_exact_acc']:.4f} "
        f"eval_topk_oracle={eval_metrics['topk_oracle_acc']:.4f} "
        f"eval_row_acc={eval_metrics['row_acc']:.4f} "
        f"eval_row_mae={eval_metrics['row_mae']:.4f} "
        f"eval_total_box_mae={eval_metrics['total_box_mae']:.4f}",
        flush=True,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else reranker.state_dict(),
            "n": n,
            "partitions": candidates_np,
            "args": vars(args),
            "classifier_checkpoint": args.classifier_checkpoint,
            "best_valid_exact_acc": best_score[0],
            "best_valid_row_acc": best_score[1],
            "eval_metrics": eval_metrics,
        },
        args.model_out,
    )
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "best_valid_exact_acc": best_score[0],
        "best_valid_row_acc": best_score[1],
        "history": history,
        "eval_metrics": eval_metrics,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.model_out}")
    print(f"wrote {args.results_out}")


if __name__ == "__main__":
    main()
