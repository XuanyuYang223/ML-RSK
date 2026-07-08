# RSK Shape Autoregressive Baseline

Dataset:

- train/eval split source: `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- independent eval: `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- `max_samples = 10000`
- split: `8000 / 2000`
- device: CPU

Model:

- autoregressive Transformer encoder-decoder
- permutation representation: Lehmer
- teacher-forced row prediction during training
- greedy row generation during evaluation
- `d_model = 32`
- `num_layers = 1`
- `num_heads = 4`
- `dim_feedforward = 128`
- `batch_size = 512`
- `epochs = 5`

## Results

| inference | internal row acc | internal exact acc | internal row MAE | internal total box MAE | independent row acc | independent exact acc | independent row MAE | independent total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained greedy | 0.242 | 0.000 | 1.106 | 0.000 | 0.2435 | 0.0000 | 1.1027 | 0.0000 |
| unconstrained greedy | 0.720 | 0.000 | 1.000 | 30.000 | 0.7180 | 0.0000 | 1.0000 | 30.0000 |
| score all partitions | n/a | n/a | n/a | n/a | 0.7183 | 0.0000 | 1.3248 | 0.0000 |

## Takeaway

This naive autoregressive baseline is not better than the non-autoregressive
Transformer plus partition decoding. Unconstrained greedy decoding drifts toward
the padded-zero majority pattern and loses all 30 boxes. Constrained greedy
decoding enforces a legal partition with 30 boxes, but it collapses early to a
poor legal shape. The useful next variant is likely beam search or partition
scoring under the autoregressive model, not plain greedy decoding.

Follow-up:

- `experiments/rsk_shape_ar_partition_scoring/` evaluates exactly that
  score-all-partitions decoder. It fixes the greedy collapse but still does not
  improve exact shape accuracy.
