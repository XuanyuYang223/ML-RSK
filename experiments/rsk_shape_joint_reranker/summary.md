# RSK Shape Joint/Listwise Reranker

Dataset:

- train/eval split source: `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- independent eval: `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- candidate generator: `experiments/rsk_shape_partition_classifier_100k_e20/lehmer/model.pt`
- top-k: `50`
- device: GPU, `cuda`

Method:

- Load the trained 5604-way Lehmer partition classifier.
- Use it as a frozen candidate generator.
- Initialize the reranker permutation encoder from the classifier encoder:
  `cls_token`, token embedding, positional embedding, and Transformer encoder.
- Encode all top-50 candidate partitions, add classifier log-probability and
  rank features, then run self-attention over the candidate list.
- Train with cross-entropy over examples where the true partition is present in
  the top-50 list.

## Independent Test Set

| variant | exact acc | top-50 oracle | row acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| classifier argmax | 0.0245 | 0.4000 | 0.8743 | 0.1322 | 0.0000 |
| old simple mix rerank | 0.0285 | 0.4000 | 0.8747 | 0.1323 | 0.0000 |
| learned pair reranker | 0.0295 | 0.4000 | 0.8746 | 0.1319 | 0.0000 |
| joint/listwise reranker, fine-tuned encoder | 0.0290 | 0.4000 | 0.8742 | 0.1322 | 0.0000 |
| joint/listwise reranker, frozen encoder | 0.0295 | 0.4000 | 0.8744 | 0.1320 | 0.0000 |

## Takeaway

The joint/listwise reranker did not beat the previous best. The frozen-encoder
variant matches the learned pair reranker at `0.0295` exact accuracy, while the
fine-tuned encoder variant reaches `0.0290`.

This is still above the original classifier argmax (`0.0245`) and the old simple
mix rerank (`0.0285`), but it does not close the large top-50 oracle gap
(`0.4000`). Reusing the classifier encoder and adding candidate self-attention
is not enough by itself.

The next useful direction is probably to change the training objective or
candidate construction, for example by fine-tuning the full 5604-way classifier
with an auxiliary top-k/listwise loss, adding hard negatives from nearby shapes,
or training on more candidate-hit examples rather than only attaching a
post-hoc reranker.
