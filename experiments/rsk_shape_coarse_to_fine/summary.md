# RSK Shape Coarse-To-Fine Predictor

Dataset:

- train/eval split source: `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- independent eval: `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- initialization: `experiments/rsk_shape_partition_classifier_100k_e20/lehmer/model.pt`
- device: GPU, `cuda`

Method:

- Start from the trained 5604-way Lehmer partition classifier.
- Add coarse shape heads on the shared classifier encoder:
  - first row length
  - number of nonzero rows
  - first two row lengths as a 226-way class
- Continue training with:

```text
loss = partition_CE + coarse_weight * (first_row_CE + num_rows_CE + first2_CE)
```

- At inference, compare:
  - ordinary partition argmax
  - masked partition argmax, where masks come from predicted coarse labels

## Independent Test Set

| variant | decode | exact acc | top-50 | top-100 | row acc | row MAE | first-row acc | num-rows acc | first2 acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original classifier | argmax | 0.0245 | 0.4000 | 0.5685 | 0.8743 | 0.1322 | n/a | n/a | n/a |
| learned pair reranker | top-50 rerank | 0.0295 | 0.4000 | n/a | 0.8746 | 0.1319 | n/a | n/a | n/a |
| joint/listwise frozen reranker | top-50 rerank | 0.0295 | 0.4000 | n/a | 0.8744 | 0.1320 | n/a | n/a | n/a |
| `coarse_weight=0.1` | first-row top-1 mask | 0.0305 | 0.4070 | 0.5825 | 0.8748 | 0.1313 | 0.4715 | 0.4340 | 0.2210 |
| `coarse_weight=0.2` | first-row top-1 mask | 0.0310 | 0.4090 | 0.5830 | 0.8748 | 0.1315 | 0.4745 | 0.4370 | 0.2230 |
| `coarse_weight=0.5` | first2 top-3 mask | 0.0305 | 0.4095 | 0.5820 | 0.8753 | 0.1309 | 0.4725 | 0.4425 | 0.2255 |

## Takeaway

Coarse-to-fine is the first variant in this branch to beat the previous exact
accuracy ceiling. The best run is `coarse_weight=0.2` with first-row top-1
masking, reaching `0.0310` exact accuracy on the independent test set.

The gain is still small, but the direction is different from the reranker and
hard-negative attempts: the model learns useful coarse shape information, and a
simple coarse mask can change the top-1 partition. The coarse heads themselves
are not highly accurate yet: first-row accuracy is only about `47%`, number of
rows about `44%`, and first-two-rows about `22%`.

The next useful step is to strengthen the coarse stage rather than add another
reranker. Good candidates are:

- train coarse heads longer or with larger weight while protecting partition CE
- use coarse labels that are easier and more stable than exact first rows
- build shape clusters over all 5604 partitions and train a cluster head
- evaluate oracle coarse masks to estimate how much head accuracy limits this
  approach
