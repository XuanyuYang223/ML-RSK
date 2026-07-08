# RSK Shape Partition Top-K Reranking

Dataset:

- `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- examples: `2000`
- device: GPU, `cuda`

Models:

- candidate generator: `experiments/rsk_shape_partition_classifier_100k_e20/lehmer/model.pt`
- row scorer: `experiments/rsk_shape_scale_100k_e5/lehmer_encdec_rows_argmax/model.pt`

Method:

- Use the 5604-way Lehmer partition classifier to produce top-k candidate
  partitions.
- Rerank those candidates with the encoder-decoder row model's summed row
  log-probability.
- Also evaluate simple score mixing:

```text
score = classifier_log_prob + alpha * row_log_prob
```

## Independent Test Set

| top-k | method | exact acc | row acc | row MAE | total box MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | classifier argmax | 0.0245 | 0.8743 | 0.1322 | 0.0000 |
| 10 | row rerank | 0.0265 | 0.8737 | 0.1336 | 0.0000 |
| 10 | mix alpha=0.25 | 0.0265 | 0.8751 | 0.1317 | 0.0000 |
| 10 | mix alpha=1.0 | 0.0285 | 0.8748 | 0.1322 | 0.0000 |
| 10 | oracle | 0.1500 | 0.8843 | 0.1222 | 0.0000 |
| 50 | classifier argmax | 0.0245 | 0.8743 | 0.1322 | 0.0000 |
| 50 | row rerank | 0.0225 | 0.8724 | 0.1359 | 0.0000 |
| 50 | mix alpha=1.0 | 0.0285 | 0.8747 | 0.1323 | 0.0000 |
| 50 | oracle | 0.4000 | 0.9100 | 0.0963 | 0.0000 |
| 100 | classifier argmax | 0.0245 | 0.8743 | 0.1322 | 0.0000 |
| 100 | mix alpha=1.0 | 0.0285 | 0.8747 | 0.1323 | 0.0000 |
| 100 | oracle | 0.5685 | 0.9314 | 0.0746 | 0.0000 |

## Takeaway

Simple reranking gives a real but small improvement: the best mixed score reaches
`0.0285` exact accuracy, up from the classifier argmax at `0.0245` and the
encoder-decoder partition scorer at `0.0230`.

The oracle numbers are the important result. The correct shape is already in the
classifier's top 10 for `15.00%`, top 50 for `40.00%`, and top 100 for `56.85%`
of independent test examples. Candidate generation is therefore much stronger
than argmax accuracy suggests.

The old row model is not a strong enough reranker. The next useful experiment is
to train a learned top-k reranker or structured contrastive objective over
classifier candidates.
