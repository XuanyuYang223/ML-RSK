#!/usr/bin/env python3
"""Generate permutations and their RSK tableaux with Sage.

Run with Sage's Python:
    /home/yangx/miniforge3/envs/sage/bin/sage generate_rsk_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from sage.combinat.rsk import RSK


def tableau_to_lists(tableau) -> list[list[int]]:
    return [[int(x) for x in row] for row in tableau]


def make_example(n: int, rng: random.Random) -> dict:
    permutation = list(range(1, n + 1))
    rng.shuffle(permutation)
    p_tableau, q_tableau = RSK(permutation)
    p_rows = tableau_to_lists(p_tableau)
    q_rows = tableau_to_lists(q_tableau)
    shape = [len(row) for row in p_rows]
    return {
        "permutation": permutation,
        "P": p_rows,
        "Q": q_rows,
        "shape": shape,
    }


def pad_shape(shape: list[int], n: int) -> list[int]:
    return shape + [0] * (n - len(shape))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(os.getenv("RSK_N", "30")), help="Permutation size.")
    parser.add_argument(
        "--samples",
        type=int,
        default=int(os.getenv("RSK_SAMPLES", "10000")),
        help="Number of examples.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("RSK_SEED", "0")))
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("RSK_OUT_DIR", "data")))
    parser.add_argument("--prefix", type=str, default=os.getenv("RSK_PREFIX"))
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"rsk_n{args.n}_m{args.samples}_seed{args.seed}"
    jsonl_path = args.out_dir / f"{prefix}.jsonl"
    npz_path = args.out_dir / f"{prefix}.npz"

    rng = random.Random(args.seed)
    x = np.zeros((args.samples, args.n), dtype=np.int64)
    y_shape = np.zeros((args.samples, args.n), dtype=np.int64)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(args.samples):
            example = make_example(args.n, rng)
            x[i] = np.array(example["permutation"], dtype=np.int64)
            y_shape[i] = np.array(pad_shape(example["shape"], args.n), dtype=np.int64)
            f.write(json.dumps(example, separators=(",", ":")) + "\n")
            if (i + 1) % max(1, args.samples // 10) == 0:
                print(f"generated {i + 1}/{args.samples}")

    np.savez_compressed(
        npz_path,
        permutations=x,
        shapes=y_shape,
        n=np.array(args.n, dtype=np.int64),
        samples=np.array(args.samples, dtype=np.int64),
        seed=np.array(args.seed, dtype=np.int64),
    )
    print(f"wrote {jsonl_path}")
    print(f"wrote {npz_path}")


if __name__ in {"__main__", "sage.all"}:
    main()
