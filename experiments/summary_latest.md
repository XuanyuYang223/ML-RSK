# Latest Experiment Summary

This records the current non-RSK-focused experiment pass.

## 1. Shape Count Feature Ablation

Path:

- `experiments/shape_count_features_n30_e120_v3/`

Tasks:

- `shape -> log #SYT`
- `shape -> log #SSYT`

Main result:

| task | rows | rows + cols | rows + cols + hooks |
| --- | ---: | ---: | ---: |
| log #SYT MAE | 0.183157 | 0.133851 | 0.100804 |
| log #SSYT MAE | 0.203174 | 0.155012 | 0.088561 |

Takeaway:

Adding conjugate partition columns helps. Hook-length summary features help the
most, which is expected because the exact formulas depend on hook lengths.

## 2. Coxeter Length Ablation

Path:

- `experiments/perm_invariants_20k_e8_v1/`

Task:

- `permutation -> Coxeter length`

Main result:

| setup | test MAE |
| --- | ---: |
| one-line classification | 22.854 |
| Lehmer classification | 16.653 |
| one-line regression | 15.418 |
| Lehmer regression | 6.532 |

Takeaway:

Regression is better than 436-way classification. Lehmer input is much better
than one-line input for length because Coxeter length is the sum of the Lehmer
code.

## 3. Sign / Parity Ablation

Paths:

- `experiments/perm_invariants_20k_e8_v1/`
- `experiments/perm_invariants_multitask_20k_e8_v1/`

Main result:

| setup | test sign acc |
| --- | ---: |
| one-line sign-only | 0.5097 |
| Lehmer sign-only | 0.5070 |
| one-line length+sign multi-task | 0.4933 |
| Lehmer length+sign multi-task | 0.4855 |

Takeaway:

Sign remains near random. Even when the model learns an approximate Coxeter
length, it does not learn length modulo 2.

## Code Fix Found During This Pass

The checkpoint best-state logic used `detach().cpu()` without `clone()`. On CPU,
that can keep references to mutable tensors, so the saved "best" state can drift
toward the final epoch. This was fixed in:

- `shape_counts/train_shape_counts_mlp.py`
- `rsk_shape/train_shape_transformer.py`
- `perm_invariants/train_perm_invariants_transformer.py`

## 4. RSK Shape Partition Decoding Check

Path:

- `experiments/rsk_shape_partition_decode/`

Task:

- `permutation -> RSK shape`

Main result on the independent 2k test set:

| representation | inference | row acc | exact acc | row MAE | total box MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| one-line | argmax | 0.8699 | 0.0105 | 0.1399 | 0.6970 |
| one-line | partition | 0.8693 | 0.0235 | 0.1402 | 0.0000 |
| Lehmer | argmax | 0.8733 | 0.0110 | 0.1350 | 0.6255 |
| Lehmer | partition | 0.8722 | 0.0230 | 0.1361 | 0.0000 |

Takeaway:

Constrained decoding over all partitions of 30 roughly doubles exact shape
accuracy and fixes total box count, but it barely changes row-wise error. This
suggests the model has some useful global shape signal, but post-hoc legality
constraints alone are not enough.

## 5. RSK Shape Autoregressive Baseline

Path:

- `experiments/rsk_shape_autoregressive_10k_e5/`

Task:

- `permutation -> RSK shape`

Main result on the independent 2k test set:

| inference | row acc | exact acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: |
| constrained greedy | 0.2435 | 0.0000 | 1.1027 | 0.0000 |
| unconstrained greedy | 0.7180 | 0.0000 | 1.0000 | 30.0000 |

Takeaway:

A naive autoregressive Transformer with greedy decoding is not enough.
Unconstrained greedy decoding drifts toward padded-zero rows, while constrained
greedy decoding creates legal but poor partitions. The next structured-output
variant should use beam search or partition scoring under the autoregressive
model.

## 6. AR Score-All-Partitions Eval

Path:

- `experiments/rsk_shape_ar_partition_scoring/`

Task:

- `permutation -> RSK shape`

Main result on the independent 2k test set:

| model | inference | row acc | exact acc | row MAE | total box MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| encoder-decoder rows | argmax | 0.8733 | 0.0110 | 0.1350 | 0.6255 |
| encoder-decoder rows | monotone | 0.8733 | 0.0110 | 0.1350 | 0.6255 |
| encoder-decoder rows | score all partitions | 0.8722 | 0.0230 | 0.1361 | 0.0000 |
| AR rows | constrained greedy | 0.2435 | 0.0000 | 1.1027 | 0.0000 |
| AR rows | score all partitions | 0.7183 | 0.0000 | 1.3248 | 0.0000 |

Takeaway:

Exhaustive partition scoring confirms that greedy search was bad, but the AR
model still does not solve exact shape prediction. This is a useful stopping
point for the current AR branch. A better next experiment is direct 5604-way
partition classification or non-AR structured partition candidate scoring.

## 7. RSK Shape 5604-Way Partition Classifier

Path:

- `experiments/rsk_shape_partition_classifier_100k_e20/`

Task:

- `permutation -> distribution over all partitions of 30`

Main result on the independent 2k test set:

| representation | exact/class acc | top-5 acc | top-10 acc | row acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one-line | 0.0175 | 0.0750 | 0.1185 | 0.8666 | 0.1427 | 0.0000 |
| Lehmer | 0.0245 | 0.0890 | 0.1500 | 0.8743 | 0.1322 | 0.0000 |

Takeaway:

The direct partition classifier gives the best exact-shape result so far, but
the improvement is small: `0.0245` exact accuracy versus `0.0230` from the
encoder-decoder plus partition scoring. The more useful signal is top-k:
with Lehmer input, the true shape appears in the top 10 for `15.00%` of examples.
This suggests a reranking or structured candidate objective may be a better next
step than simply increasing classifier size.

## 8. RSK Shape Partition Top-K Reranking

Path:

- `experiments/rsk_shape_partition_rerank/`

Task:

- rerank top-k partitions from the 5604-way classifier

Main result on the independent 2k test set:

| top-k | method | exact acc | row acc | row MAE | total box MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | classifier argmax | 0.0245 | 0.8743 | 0.1322 | 0.0000 |
| 10 | mix alpha=1.0 | 0.0285 | 0.8748 | 0.1322 | 0.0000 |
| 10 | oracle | 0.1500 | 0.8843 | 0.1222 | 0.0000 |
| 50 | mix alpha=1.0 | 0.0285 | 0.8747 | 0.1323 | 0.0000 |
| 50 | oracle | 0.4000 | 0.9100 | 0.0963 | 0.0000 |
| 100 | mix alpha=1.0 | 0.0285 | 0.8747 | 0.1323 | 0.0000 |
| 100 | oracle | 0.5685 | 0.9314 | 0.0746 | 0.0000 |

Takeaway:

Simple reranking with the old row model gives a small exact-accuracy gain,
`0.0245 -> 0.0285`. The much bigger signal is candidate recall: the correct
shape is in the classifier's top 100 for `56.85%` of examples. The next useful
step is to train a dedicated top-k reranker or structured contrastive objective
over classifier candidates.

## 9. RSK Shape Learned Top-K Reranker

Path:

- `experiments/rsk_shape_learned_reranker/`

Task:

- train a learned pair scorer for `(permutation, candidate_partition)` over
  top-k candidates from the 5604-way Lehmer partition classifier

Main result on the independent 2k test set:

| top-k | method | exact acc | top-k oracle | row acc | row MAE | total box MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | classifier argmax | 0.0245 | 0.4000 | 0.8743 | 0.1322 | 0.0000 |
| 50 | old simple mix rerank | 0.0285 | 0.4000 | 0.8747 | 0.1323 | 0.0000 |
| 50 | learned reranker | 0.0295 | 0.4000 | 0.8746 | 0.1319 | 0.0000 |
| 100 | learned reranker | 0.0280 | 0.5685 | 0.8742 | 0.1323 | 0.0000 |

Takeaway:

The learned top-50 reranker is the best exact-shape result so far, but the gain
is modest: `0.0245 -> 0.0295`, only slightly above the old simple mix rerank at
`0.0285`. Top-100 has much larger oracle recall but did not improve the learned
reranker, suggesting the current separate pair scorer is too weak to exploit the
candidate set. A better next step is likely joint/listwise training from the
classifier checkpoint or a richer reranker that compares candidates against each
other, not only independent `(permutation, partition)` pairs.

## 10. RSK Shape Joint/Listwise Reranker

Path:

- `experiments/rsk_shape_joint_reranker/`

Task:

- initialize a top-50 reranker from the trained 5604-way classifier encoder and
  score candidates jointly with candidate self-attention

Main result on the independent 2k test set:

| variant | exact acc | top-50 oracle | row acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| classifier argmax | 0.0245 | 0.4000 | 0.8743 | 0.1322 | 0.0000 |
| old simple mix rerank | 0.0285 | 0.4000 | 0.8747 | 0.1323 | 0.0000 |
| learned pair reranker | 0.0295 | 0.4000 | 0.8746 | 0.1319 | 0.0000 |
| joint/listwise reranker, fine-tuned encoder | 0.0290 | 0.4000 | 0.8742 | 0.1322 | 0.0000 |
| joint/listwise reranker, frozen encoder | 0.0295 | 0.4000 | 0.8744 | 0.1320 | 0.0000 |

Takeaway:

The joint/listwise reranker does not beat the previous best. Freezing the
classifier-initialized permutation encoder matches the learned pair reranker at
`0.0295`; fine-tuning the encoder reaches `0.0290`. This suggests the bottleneck
is not just the lack of candidate-to-candidate attention in the post-hoc
reranker. The next useful direction is to fine-tune the 5604-way classifier
itself with an auxiliary top-k/listwise or hard-negative objective, rather than
adding another frozen-candidate reranker.

## 11. RSK Shape Classifier Auxiliary Top-K Fine-Tuning

Path:

- `experiments/rsk_shape_classifier_aux_topk/`

Task:

- continue training the 5604-way Lehmer partition classifier with the original
  cross-entropy plus optional top-50 hard-negative auxiliary loss

Main result on the independent 2k test set:

| variant | exact acc | top-5 | top-10 | top-50 | top-100 | row acc | row MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original classifier | 0.0245 | 0.0890 | 0.1500 | 0.4000 | 0.5685 | 0.8743 | 0.1322 |
| learned pair reranker | 0.0295 | n/a | n/a | 0.4000 | n/a | 0.8746 | 0.1319 |
| joint/listwise frozen-encoder reranker | 0.0295 | n/a | n/a | 0.4000 | n/a | 0.8744 | 0.1320 |
| hard negatives, weight 0.5 | 0.0290 | 0.0855 | 0.1440 | 0.3930 | 0.5600 | 0.8732 | 0.1334 |
| hard negatives, weight 0.1 | 0.0255 | 0.0870 | 0.1530 | 0.4020 | 0.5760 | 0.8746 | 0.1318 |
| CE-only fine-tune | 0.0265 | 0.0870 | 0.1530 | 0.3980 | 0.5770 | 0.8748 | 0.1316 |

Takeaway:

Auxiliary hard-negative fine-tuning does not improve the best exact accuracy.
The strongest hard-negative run reaches `0.0290`, slightly below the post-hoc
rerankers at `0.0295`. Light fine-tuning improves top-100 candidate recall a
little (`0.5685 -> 0.5770`), but that recall gain still does not become top-1
accuracy. The next useful direction should change the problem decomposition,
for example a coarse-to-fine shape predictor with auxiliary coarse shape labels,
rather than adding another similar reranker or hard-negative loss.

## 12. RSK Shape Coarse-To-Fine Predictor

Path:

- `experiments/rsk_shape_coarse_to_fine/`

Task:

- train a multitask classifier with shared encoder, original 5604-way partition
  head, and auxiliary coarse shape heads; at inference, restrict partitions with
  predicted coarse labels

Main result on the independent 2k test set:

| variant | decode | exact acc | top-50 | top-100 | row acc | row MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original classifier | argmax | 0.0245 | 0.4000 | 0.5685 | 0.8743 | 0.1322 |
| learned pair reranker | top-50 rerank | 0.0295 | 0.4000 | n/a | 0.8746 | 0.1319 |
| joint/listwise frozen reranker | top-50 rerank | 0.0295 | 0.4000 | n/a | 0.8744 | 0.1320 |
| `coarse_weight=0.1` | first-row top-1 mask | 0.0305 | 0.4070 | 0.5825 | 0.8748 | 0.1313 |
| `coarse_weight=0.2` | first-row top-1 mask | 0.0310 | 0.4090 | 0.5830 | 0.8748 | 0.1315 |
| `coarse_weight=0.5` | first2 top-3 mask | 0.0305 | 0.4095 | 0.5820 | 0.8753 | 0.1309 |

Takeaway:

Coarse-to-fine gives the first small improvement beyond the `0.0295` ceiling
from reranking. The best result is `0.0310` exact accuracy with
`coarse_weight=0.2` and first-row top-1 masking. The coarse heads are still weak
in absolute terms, with first-row accuracy around `47%`, num-rows accuracy
around `44%`, and first-two-rows accuracy around `22%`. The next useful step is
to evaluate oracle coarse masks and easier bucket/cluster coarse labels to see
whether better coarse prediction can open more exact-accuracy headroom.
