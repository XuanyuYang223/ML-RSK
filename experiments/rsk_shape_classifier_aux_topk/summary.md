# RSK Shape Classifier Auxiliary Top-K Fine-Tuning

Dataset:

- train/eval split source: `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- independent eval: `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- initialization: `experiments/rsk_shape_partition_classifier_100k_e20/lehmer/model.pt`
- device: GPU, `cuda`

Method:

- Continue fine-tuning the trained 5604-way Lehmer partition classifier.
- Keep the original 5604-way cross-entropy.
- Optionally add an auxiliary hard-negative loss:
  - take the current top-50 wrong labels for each example
  - compare the true partition against those hard negatives
  - apply cross-entropy over `[true, hard_negatives]`

Variants:

- `ce_only`: continue with only the original 5604-way cross-entropy
- `hn50_w01`: add top-50 hard-negative auxiliary loss with weight `0.1`
- `hn50_w05`: add top-50 hard-negative auxiliary loss with weight `0.5`

## Independent Test Set

| variant | exact acc | top-5 | top-10 | top-50 | top-100 | row acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original classifier | 0.0245 | 0.0890 | 0.1500 | 0.4000 | 0.5685 | 0.8743 | 0.1322 | 0.0000 |
| learned pair reranker | 0.0295 | n/a | n/a | 0.4000 | n/a | 0.8746 | 0.1319 | 0.0000 |
| joint/listwise reranker, frozen encoder | 0.0295 | n/a | n/a | 0.4000 | n/a | 0.8744 | 0.1320 | 0.0000 |
| `hn50_w05` | 0.0290 | 0.0855 | 0.1440 | 0.3930 | 0.5600 | 0.8732 | 0.1334 | 0.0000 |
| `hn50_w01` | 0.0255 | 0.0870 | 0.1530 | 0.4020 | 0.5760 | 0.8746 | 0.1318 | 0.0000 |
| `ce_only` | 0.0265 | 0.0870 | 0.1530 | 0.3980 | 0.5770 | 0.8748 | 0.1316 | 0.0000 |

## Takeaway

Auxiliary top-k fine-tuning did not improve the best exact accuracy. The best
exact result in this experiment is `0.0290` from the stronger hard-negative
weight, still below or equal to the post-hoc rerankers at `0.0295`.

The more interesting effect is candidate recall. Light fine-tuning improves
top-100 recall on the independent test set from `0.5685` to `0.5770`, and
`hn50_w01` slightly improves top-50 recall to `0.4020`. CE-only fine-tuning is
competitive with the hard-negative objective, so the recall gain may come mostly
from continued classifier training rather than the auxiliary loss itself.

The bottleneck remains top-1 ranking. The current classifier can put the true
shape in a large candidate set, but neither post-hoc reranking nor hard-negative
fine-tuning has converted much of that candidate recall into exact accuracy.
