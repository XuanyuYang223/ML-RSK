# RSK Shape Learned Top-K Reranker

Dataset:

- train/eval split source: `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- independent eval: `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- candidate generator: `experiments/rsk_shape_partition_classifier_100k_e20/lehmer/model.pt`
- device: GPU, `cuda`

Method:

- Freeze the trained 5604-way Lehmer partition classifier.
- For each permutation, generate classifier top-k candidate partitions.
- Train a pair scorer on `(permutation, candidate_partition)` with cross-entropy
  over candidate sets where the true partition is present.
- Keep the classifier log-probability as a learnable weighted feature in the
  reranker score.

Model:

- permutation encoder: Transformer encoder with CLS pooling
- candidate encoder: row-value and row-position embeddings plus normalized rows
- pair scorer: MLP over permutation, candidate, interaction, distance, and
  classifier log-probability features
- `d_model = 128`
- `num_layers = 2`
- `num_heads = 4`
- `dim_feedforward = 512`
- `epochs = 8`

## Independent Test Set

| top-k | method | exact acc | top-k oracle | row acc | row MAE | total box MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | classifier argmax | 0.0245 | 0.4000 | 0.8743 | 0.1322 | 0.0000 |
| 50 | old simple mix rerank | 0.0285 | 0.4000 | 0.8747 | 0.1323 | 0.0000 |
| 50 | learned reranker | 0.0295 | 0.4000 | 0.8746 | 0.1319 | 0.0000 |
| 100 | classifier argmax | 0.0245 | 0.5685 | 0.8743 | 0.1322 | 0.0000 |
| 100 | old simple mix rerank | 0.0285 | 0.5685 | 0.8747 | 0.1323 | 0.0000 |
| 100 | learned reranker | 0.0280 | 0.5685 | 0.8742 | 0.1323 | 0.0000 |

## Takeaway

The learned reranker improves the best exact-shape result slightly:
`0.0245 -> 0.0295` for top-50 candidates, beating the old row-model mix rerank
at `0.0285`.

However, the improvement is still tiny compared with the oracle headroom:
top-50 oracle remains `0.4000`, and top-100 oracle remains `0.5685`. Increasing
the candidate set to 100 did not help this reranker; it made the ranking problem
harder and reached only `0.0280` exact accuracy.

This suggests the current pair scorer is mostly learning a small correction to
the classifier ordering, not extracting enough new signal to exploit the large
candidate recall. A stronger next variant should either fine-tune the permutation
encoder jointly from the classifier checkpoint, add richer candidate-pair/listwise
features, or train the original classifier with a top-k/listwise objective rather
than using a separate shallow reranker.
