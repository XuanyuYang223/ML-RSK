# RSK Shape AR Score-All-Partitions Eval

Dataset:

- `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- examples: `2000`
- device: CPU

Checkpoint:

- `experiments/rsk_shape_autoregressive_10k_e5/lehmer_constrained/model.pt`

Method:

- Enumerate all integer partitions of 30.
- For each permutation, compute the teacher-forced autoregressive log-probability
  of every candidate partition.
- Predict the partition with the highest summed row log-probability.

## Fair Comparison On The Independent 2k Test Set

| model | inference | row acc | exact acc | row MAE | total box MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| encoder-decoder rows | argmax | 0.8733 | 0.0110 | 0.1350 | 0.6255 |
| encoder-decoder rows | monotone | 0.8733 | 0.0110 | 0.1350 | 0.6255 |
| encoder-decoder rows | score all partitions | 0.8722 | 0.0230 | 0.1361 | 0.0000 |
| AR rows | constrained greedy | 0.2435 | 0.0000 | 1.1027 | 0.0000 |
| AR rows | score all partitions | 0.7183 | 0.0000 | 1.3248 | 0.0000 |

## Takeaway

Exhaustive partition scoring fixes the AR greedy-search failure: row accuracy
rises from `0.2435` to `0.7183`, while exact validity and total box count remain
guaranteed. However, exact full-shape accuracy is still `0.0000`, and row-wise
quality is much worse than the existing non-autoregressive encoder-decoder.

This suggests the trained AR model did learn some candidate-ranking signal, but
the current AR training objective/model is not strong enough. The next RSK shape
step should stop this AR branch for now and try either:

- non-AR model plus structured partition candidate scoring/loss, or
- direct `permutation -> partition class` prediction over the 5604 partitions of
  30.
