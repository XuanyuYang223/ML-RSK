# RSK Shape 100k Scale Check

Dataset:

- train/eval split source: `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- `n = 30`
- samples: `100000`
- split: `80000 / 20000`
- device: CPU

Model:

- architecture: encoder-decoder Transformer
- output: padded row lengths
- inference: argmax
- `d_model = 64`
- `num_layers = 2`
- `num_heads = 4`
- `dim_feedforward = 256`
- `batch_size = 512`
- `epochs = 5`

## Internal Split

| representation | test row acc | exact acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: |
| one-line | 0.869 | 0.009 | 0.141 | 0.695 |
| Lehmer | 0.872 | 0.010 | 0.137 | 0.632 |

## Independent Test Set

Evaluated on `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`.

| representation | row acc | exact acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: |
| one-line | 0.8699 | 0.0105 | 0.1399 | 0.6970 |
| Lehmer | 0.8733 | 0.0110 | 0.1350 | 0.6255 |

## Takeaway

Lehmer encoding is a small but consistent improvement over one-line encoding in
this run. Exact full-shape accuracy remains low, so the next useful work is
probably constraint-aware decoding or a better structured output, not only more
data.
