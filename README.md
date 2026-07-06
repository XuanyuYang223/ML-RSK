# ML-RSK

Small Sage + PyTorch/NumPy experiments for generating random permutations,
computing RSK data, and training neural networks on combinatorial targets.

## Environment

Sage is installed in the conda environment at:

```bash
/home/yangx/miniforge3/envs/sage/bin/sage --version
```

In the current shell, `sage` is not on `PATH`, so use the full path or activate
the environment first:

```bash
source /home/yangx/miniforge3/bin/activate sage
```

## Generate Data

```bash
/home/yangx/miniforge3/envs/sage/bin/sage rsk_shape/generate_rsk_data.py
```

The default is `n=30`, `samples=10000`, `seed=0`. To override those values:

```bash
RSK_N=30 RSK_SAMPLES=10000 RSK_SEED=0 \
  /home/yangx/miniforge3/envs/sage/bin/sage rsk_shape/generate_rsk_data.py
```

This writes:

- `data/rsk_shape/rsk_n30_m10000_seed0.jsonl`: full records with `permutation`, `P`, `Q`,
  and `shape`
- `data/rsk_shape/rsk_n30_m10000_seed0.npz`: NumPy arrays for model training

## Train Transformer Shape Model

This is the first main model. It reads the permutation as discrete tokens and
predicts the padded RSK shape as `n` categorical outputs, each in `0..n`.

```bash
/home/yangx/miniforge3/envs/sage/bin/python rsk_shape/train_shape_transformer.py \
  data/rsk_shape/rsk_n30_m10000_seed0.npz \
  --epochs 50 \
  --batch-size 128 \
  --d-model 128 \
  --num-layers 3 \
  --num-heads 4
```

Metrics:

- `row_acc`: accuracy over padded shape entries
- `exact_acc`: full padded shape vector exactly correct
- `row_mae`: mean absolute error per shape entry
- `total_box_mae`: absolute error of the predicted total number of boxes

## Evaluate Transformer Shape Model

```bash
/home/yangx/miniforge3/envs/sage/bin/python rsk_shape/eval_shape_transformer.py \
  data/rsk_shape/rsk_n30_m2000_seed1_test.npz \
  --checkpoint models/rsk_shape/shape_transformer.pt
```

## Train Permutation Invariant Models

These models use the same permutation dataset and train a Transformer classifier
for Coxeter length or permutation sign.

```bash
/home/yangx/miniforge3/envs/sage/bin/python perm_invariants/train_perm_invariants_transformer.py \
  data/rsk_shape/rsk_n30_m10000_seed0.npz \
  --task length \
  --epochs 20
```

```bash
/home/yangx/miniforge3/envs/sage/bin/python perm_invariants/train_perm_invariants_transformer.py \
  data/rsk_shape/rsk_n30_m10000_seed0.npz \
  --task sign \
  --epochs 20
```

The first short CPU run is recorded in:

- `results/perm_invariants_quick.md`

## Train Shape Count Models

Predict `log #SYT(lambda)` or `log #SSYT(lambda, m)` from a partition shape.

```bash
/home/yangx/miniforge3/envs/sage/bin/python shape_counts/train_shape_counts_mlp.py \
  --task syt \
  --n 30 \
  --epochs 300
```

```bash
/home/yangx/miniforge3/envs/sage/bin/python shape_counts/train_shape_counts_mlp.py \
  --task ssyt \
  --n 30 \
  --alphabet-size 30 \
  --epochs 300
```

## Train NumPy MLP Baseline

This is a lightweight sanity-check baseline:

```bash
/home/yangx/miniforge3/envs/sage/bin/python rsk_shape/train_shape_mlp_numpy.py \
  data/rsk_shape/rsk_n30_m10000_seed0.npz \
  --epochs 200 \
  --hidden 128
```

The model predicts the padded RSK shape vector. Full tableau prediction is a
harder structured-output problem and needs a more deliberate encoding.
