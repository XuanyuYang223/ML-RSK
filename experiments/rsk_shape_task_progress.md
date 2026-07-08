# RSK Shape Task Progress

This file is the handoff note for the next chatbox. It records the initial
baseline description, what has changed since then, and the recommended next
step.

## Initial Snapshot

The earliest baseline was:

- permutations stored in one-line notation as length-`n` integer sequences with
  values `1..n`
- no one-hot, no Lehmer code, no Coxeter-generator encoding
- encoder-decoder Transformer for `permutation -> RSK shape`
- tokenizer: `nn.Embedding(n + 1, d_model)` plus learned positional embeddings
- encoder reads the full permutation
- decoder uses learned shape-query embeddings, not autoregressive previous
  outputs
- output is a padded length-`n` row sequence, each row a class over `0..n`
- inference is one forward pass with row-wise argmax
- no beam search, no sampling, no partition-constraint enforcement
- main dataset at that point was `10k` samples for `n=30`
- held-out RSK shape result was approximately:
  - `row_acc = 0.867`
  - `exact_acc = 0.0095`
  - `total_box_mae = 0.67`
- sign was basically random in the quick run
- Coxeter length learned a rough distribution but not exact values
- shape-count MLP tasks converged better because the targets are smoother

## What Is Outdated In The Initial Snapshot

The initial snapshot is useful as baseline history, but it is no longer the
current state.

- Representation is no longer only one-line. We systematically compared
  one-line, inverse, and Lehmer inputs; Lehmer is currently best for the main
  RSK shape classifier and Coxeter length.
- Partition constraints have been tested. Score-all-partitions decoding over
  all partitions of 30 roughly doubled exact RSK shape accuracy for the row
  model and fixed total box count.
- The main RSK shape experiments now use `100k` training samples at `n=30`, plus
  the independent `2k` test set.
- The current strongest RSK shape setup is not the original row-wise
  encoder-decoder. It is a 5604-way partition classifier plus reranking
  experiments.
- The autoregressive branch was tried and should be paused for now; constrained
  greedy and score-all-partitions under that AR model did not solve exact shape
  prediction.

## Current Artifacts

Important files to read first:

- `experiments/summary_latest.md`
- `experiments/rsk_shape_partition_classifier_100k_e20/summary.md`
- `experiments/rsk_shape_partition_rerank/summary.md`
- `experiments/rsk_shape_learned_reranker/summary.md`
- `experiments/rsk_shape_joint_reranker/summary.md`
- `experiments/rsk_shape_classifier_aux_topk/summary.md`
- `experiments/rsk_shape_coarse_to_fine/summary.md`
- `rsk_shape/train_shape_partition_classifier.py`
- `rsk_shape/eval_shape_partition_rerank.py`
- `rsk_shape/train_shape_partition_learned_reranker.py`
- `rsk_shape/train_shape_partition_joint_reranker.py`
- `rsk_shape/train_shape_partition_classifier_aux_topk.py`
- `rsk_shape/train_shape_coarse_to_fine.py`

Important checkpoints:

- `experiments/rsk_shape_partition_classifier_100k_e20/lehmer/model.pt`
- `experiments/rsk_shape_learned_reranker/top50_model.pt`
- `experiments/rsk_shape_learned_reranker/top100_model.pt`
- `experiments/rsk_shape_joint_reranker/top50_model.pt`
- `experiments/rsk_shape_joint_reranker/top50_frozen_encoder_model.pt`
- `experiments/rsk_shape_classifier_aux_topk/ce_only_model.pt`
- `experiments/rsk_shape_classifier_aux_topk/hn50_w01_model.pt`
- `experiments/rsk_shape_classifier_aux_topk/hn50_w05_model.pt`
- `experiments/rsk_shape_coarse_to_fine/w01_model.pt`
- `experiments/rsk_shape_coarse_to_fine/w02_model.pt`
- `experiments/rsk_shape_coarse_to_fine/w05_model.pt`

Important datasets:

- `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`

## Current Results

### Encoder-Decoder Row Model

On the independent 2k test set:

| representation | inference | exact acc | row acc | row MAE | total box MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| one-line | argmax | 0.0105 | 0.8699 | 0.1399 | 0.6970 |
| one-line | partition scoring | 0.0235 | 0.8693 | 0.1402 | 0.0000 |
| Lehmer | argmax | 0.0110 | 0.8733 | 0.1350 | 0.6255 |
| Lehmer | partition scoring | 0.0230 | 0.8722 | 0.1361 | 0.0000 |

### 5604-Way Partition Classifier

Task: `permutation -> distribution over all partitions of 30`.

On the independent 2k test set:

| representation | exact/class acc | top-5 acc | top-10 acc | row acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one-line | 0.0175 | 0.0750 | 0.1185 | 0.8666 | 0.1427 | 0.0000 |
| Lehmer | 0.0245 | 0.0890 | 0.1500 | 0.8743 | 0.1322 | 0.0000 |

Key point: classifier argmax exact accuracy is low, but the correct shape is
often near the top.

### Old Row-Model Reranking

Using the 5604-way Lehmer classifier as candidate generator and the old row
model as a scorer:

| top-k | method | exact acc | row acc | row MAE | total box MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10 | classifier argmax | 0.0245 | 0.8743 | 0.1322 | 0.0000 |
| 10 | mix alpha=1.0 | 0.0285 | 0.8748 | 0.1322 | 0.0000 |
| 10 | oracle | 0.1500 | 0.8843 | 0.1222 | 0.0000 |
| 50 | oracle | 0.4000 | 0.9100 | 0.0963 | 0.0000 |
| 100 | oracle | 0.5685 | 0.9314 | 0.0746 | 0.0000 |

### Learned Top-K Reranker

Script:

- `rsk_shape/train_shape_partition_learned_reranker.py`

Method:

- freeze the trained 5604-way Lehmer partition classifier
- generate top-k candidate partitions
- train a pair scorer on `(permutation, candidate_partition)`
- train with cross-entropy over candidate sets where the true partition is
  present
- include classifier log-probability as a learnable score feature

On the independent 2k test set:

| top-k | method | exact acc | top-k oracle | row acc | row MAE | total box MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | classifier argmax | 0.0245 | 0.4000 | 0.8743 | 0.1322 | 0.0000 |
| 50 | old simple mix rerank | 0.0285 | 0.4000 | 0.8747 | 0.1323 | 0.0000 |
| 50 | learned reranker | 0.0295 | 0.4000 | 0.8746 | 0.1319 | 0.0000 |
| 100 | learned reranker | 0.0280 | 0.5685 | 0.8742 | 0.1323 | 0.0000 |

Current best exact RSK shape result is the learned top-50 reranker at `0.0295`.
The gain is real but small compared with the oracle headroom.

### Joint/Listwise Reranker

Script:

- `rsk_shape/train_shape_partition_joint_reranker.py`

Method:

- freeze the 5604-way Lehmer classifier as the top-k candidate generator
- initialize the reranker permutation encoder from the classifier checkpoint
- score the top-50 candidate list jointly with candidate self-attention
- train with cross-entropy over candidate-hit examples
- test both fine-tuned and frozen permutation encoder variants

On the independent 2k test set:

| variant | exact acc | top-50 oracle | row acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| classifier argmax | 0.0245 | 0.4000 | 0.8743 | 0.1322 | 0.0000 |
| old simple mix rerank | 0.0285 | 0.4000 | 0.8747 | 0.1323 | 0.0000 |
| learned pair reranker | 0.0295 | 0.4000 | 0.8746 | 0.1319 | 0.0000 |
| joint/listwise reranker, fine-tuned encoder | 0.0290 | 0.4000 | 0.8742 | 0.1322 | 0.0000 |
| joint/listwise reranker, frozen encoder | 0.0295 | 0.4000 | 0.8744 | 0.1320 | 0.0000 |

The joint/listwise reranker did not improve over the previous best. The
frozen-encoder variant matches `0.0295`; the fine-tuned encoder variant reaches
`0.0290`.

### Classifier Auxiliary Top-K Fine-Tuning

Script:

- `rsk_shape/train_shape_partition_classifier_aux_topk.py`

Method:

- initialize from the trained 5604-way Lehmer partition classifier
- continue training the classifier itself
- keep the original 5604-way cross-entropy
- optionally add hard-negative auxiliary loss comparing the true partition
  against the current top-50 wrong labels

On the independent 2k test set:

| variant | exact acc | top-5 | top-10 | top-50 | top-100 | row acc | row MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original classifier | 0.0245 | 0.0890 | 0.1500 | 0.4000 | 0.5685 | 0.8743 | 0.1322 |
| learned pair reranker | 0.0295 | n/a | n/a | 0.4000 | n/a | 0.8746 | 0.1319 |
| joint/listwise frozen-encoder reranker | 0.0295 | n/a | n/a | 0.4000 | n/a | 0.8744 | 0.1320 |
| `hn50_w05` | 0.0290 | 0.0855 | 0.1440 | 0.3930 | 0.5600 | 0.8732 | 0.1334 |
| `hn50_w01` | 0.0255 | 0.0870 | 0.1530 | 0.4020 | 0.5760 | 0.8746 | 0.1318 |
| `ce_only` | 0.0265 | 0.0870 | 0.1530 | 0.3980 | 0.5770 | 0.8748 | 0.1316 |

The auxiliary top-k objective did not improve exact accuracy. Light fine-tuning
does improve top-100 candidate recall slightly (`0.5685 -> 0.5770`), but this
still does not translate into top-1 exact accuracy.

### Coarse-To-Fine Predictor

Script:

- `rsk_shape/train_shape_coarse_to_fine.py`

Method:

- initialize from the trained 5604-way Lehmer partition classifier
- add coarse heads on the shared encoder:
  - first row length
  - number of nonzero rows
  - first two rows as a 226-way class
- train with original 5604-way CE plus weighted coarse CE losses
- at inference, mask candidate partitions using predicted coarse labels

On the independent 2k test set:

| variant | decode | exact acc | top-50 | top-100 | row acc | row MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original classifier | argmax | 0.0245 | 0.4000 | 0.5685 | 0.8743 | 0.1322 |
| learned pair reranker | top-50 rerank | 0.0295 | 0.4000 | n/a | 0.8746 | 0.1319 |
| joint/listwise frozen reranker | top-50 rerank | 0.0295 | 0.4000 | n/a | 0.8744 | 0.1320 |
| `coarse_weight=0.1` | first-row top-1 mask | 0.0305 | 0.4070 | 0.5825 | 0.8748 | 0.1313 |
| `coarse_weight=0.2` | first-row top-1 mask | 0.0310 | 0.4090 | 0.5830 | 0.8748 | 0.1315 |
| `coarse_weight=0.5` | first2 top-3 mask | 0.0305 | 0.4095 | 0.5820 | 0.8753 | 0.1309 |

This is the first direction that beats the previous exact-accuracy ceiling.
The best result is `0.0310` from `coarse_weight=0.2` with first-row top-1
masking.

## Current Interpretation

The main problem has shifted:

- It is no longer only "the model cannot find the correct RSK shape."
- The classifier often puts the correct partition in the candidate set.
- The hard part is ranking the correct candidate first.
- A separate shallow pair scorer helps only slightly.
- A first joint/listwise reranker with classifier-encoder initialization also
  helps only slightly and does not close the oracle gap.
- Auxiliary hard-negative fine-tuning improves candidate recall slightly but
  still does not solve top-1 ranking.
- Coarse-to-fine masking gives the first small but real exact-accuracy
  improvement beyond `0.0295`, reaching `0.0310`.

The top-k oracle gap remains large:

- top-10 oracle: `0.1500`
- top-50 oracle: `0.4000`
- top-100 oracle: `0.5685`

So the useful next work is not more AR greedy decoding. It is a better
candidate-ranking or listwise objective.

## Completed Steps

Do not continue the original AR greedy branch.

The previous recommended task was completed:

```text
Train a joint/listwise reranker from the 5604-way classifier checkpoint.
Reuse or initialize from the trained classifier encoder, score top-k candidate
partitions jointly, and optimize a listwise candidate objective rather than a
separate shallow pair scorer.
```

Result:

- output directory: `experiments/rsk_shape_joint_reranker/`
- fine-tuned encoder exact acc: `0.0290`
- frozen encoder exact acc: `0.0295`
- no improvement over the previous learned pair reranker
- still far below top-50 oracle `0.4000`

The second recommended task was also completed:

```text
Fine-tune the 5604-way partition classifier with an auxiliary top-k/listwise
loss or hard-negative objective, instead of training a separate reranker after
the classifier is frozen.
```

Result:

- output directory: `experiments/rsk_shape_classifier_aux_topk/`
- `hn50_w05` exact acc: `0.0290`
- `hn50_w01` exact acc: `0.0255`
- `ce_only` exact acc: `0.0265`
- best top-100 candidate recall increased slightly to `0.5770`
- no improvement over the previous best exact acc `0.0295`

The third recommended task was completed:

```text
Train a coarse-to-fine RSK shape predictor: first predict a small set of coarse
shape statistics or a shape cluster, then predict/rerank partitions only within
that restricted family.
```

Result:

- output directory: `experiments/rsk_shape_coarse_to_fine/`
- best exact acc: `0.0310`
- best variant: `coarse_weight=0.2` with first-row top-1 mask
- top-50 recall: `0.4090`
- top-100 recall: `0.5830`
- first-row coarse accuracy: `0.4745`
- num-rows coarse accuracy: `0.4370`
- first-two-rows coarse accuracy: `0.2230`

## Recommended Next Step

At this point, repeated reranking variants have a consistent story:

- classifier top-k candidate generation is useful
- exact top-1 ranking barely improves
- post-hoc reranking, joint candidate attention, and hard-negative fine-tuning
  all remain near `0.03` exact accuracy
- coarse-to-fine is more promising, but the coarse heads are still weak

The next best task should strengthen the coarse stage and measure the available
headroom from coarse restrictions.

Recommended task:

```text
Run a coarse-stage ablation for RSK shape prediction: evaluate oracle coarse
masks and train easier cluster/bucket coarse labels to see how much exact
accuracy can improve if the coarse stage is more reliable.
```

Concrete implementation direction:

- add an eval script or extend `train_shape_coarse_to_fine.py` to report oracle
  coarse-mask exact accuracy:
  - true first row mask
  - true num-rows mask
  - true first2 mask
  - combinations of true coarse labels
- add easier coarse labels:
  - bucketed first row length
  - bucketed number of rows
  - shape clusters over all 5604 partitions by row-vector distance
- train coarse-to-fine variants with these labels
- compare predicted coarse masks against oracle coarse masks
- evaluate on `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- report exact acc, top-k oracle acc, row acc, row MAE, and total box MAE
- compare against:
  - classifier argmax exact acc: `0.0245`
  - old simple mix rerank exact acc: `0.0285`
  - learned top-50 reranker exact acc: `0.0295`
  - joint/listwise frozen-encoder reranker exact acc: `0.0295`
  - auxiliary top-k `hn50_w05` exact acc: `0.0290`
  - CE-only fine-tuned classifier top-100 recall: `0.5770`
  - coarse-to-fine exact acc: `0.0310`
  - top-50 oracle: `0.4000`
  - top-100 oracle: `0.5685`

Suggested output directory:

- `experiments/rsk_shape_coarse_oracle_ablation/`
