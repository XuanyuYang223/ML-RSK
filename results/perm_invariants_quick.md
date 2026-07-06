# Permutation Invariants Quick Run

Dataset:

- `data/rsk_shape/rsk_n30_m10000_seed0.npz`
- `n = 30`
- `train = 8000`
- `test = 2000`
- device: CPU

Model:

- Transformer encoder with CLS pooling
- `d_model = 64`
- `num_layers = 2`
- `num_heads = 4`
- `dim_feedforward = 256`
- `batch_size = 256`
- `epochs = 8`

## Coxeter Length

Command:

```bash
/home/yangx/miniforge3/envs/sage/bin/python perm_invariants/train_perm_invariants_transformer.py \
  data/rsk_shape/rsk_n30_m10000_seed0.npz \
  --task length \
  --epochs 8 \
  --batch-size 256 \
  --d-model 64 \
  --num-heads 4 \
  --num-layers 2 \
  --dim-feedforward 256 \
  --model-out models/perm_invariants/length_transformer_small.pt
```

Final epoch:

```text
epoch=0008 train_loss=4.7771 test_loss=4.7945 train_acc=0.0151 test_acc=0.0145 test_mae=23.226
```

Checkpoint:

- `models/perm_invariants/length_transformer_small.pt`

Interpretation:

The model is learning the broad length distribution, but exact 436-class
classification is still weak after this short CPU run. The test MAE of about 23
means it is not yet a precise Coxeter length predictor.

## Sign

Command:

```bash
/home/yangx/miniforge3/envs/sage/bin/python perm_invariants/train_perm_invariants_transformer.py \
  data/rsk_shape/rsk_n30_m10000_seed0.npz \
  --task sign \
  --epochs 8 \
  --batch-size 256 \
  --d-model 64 \
  --num-heads 4 \
  --num-layers 2 \
  --dim-feedforward 256 \
  --model-out models/perm_invariants/sign_transformer_small.pt
```

Final epoch:

```text
epoch=0008 train_loss=0.6930 test_loss=0.6936 train_acc=0.5174 test_acc=0.5025
```

Checkpoint:

- `models/perm_invariants/sign_transformer_small.pt`

Interpretation:

The sign task is basically at random chance. This is expected for a small,
short-trained vanilla Transformer because permutation parity is an exact mod-2
global property, not a smooth target.
