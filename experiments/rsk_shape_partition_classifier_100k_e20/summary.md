# RSK Shape 5604-Way Partition Classifier

Dataset:

- train/eval split source: `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- independent eval: `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- classes: all `5604` partitions of 30
- split: `80000 / 20000`
- device: GPU, `cuda`

Model:

- Transformer encoder with CLS pooling
- output: distribution over all partitions of 30
- `d_model = 128`
- `num_layers = 2`
- `num_heads = 4`
- `dim_feedforward = 512`
- `batch_size = 1024`
- `epochs = 20`

## Independent Test Set

| representation | exact/class acc | top-5 acc | top-10 acc | row acc | row MAE | total box MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one-line | 0.0175 | 0.0750 | 0.1185 | 0.8666 | 0.1427 | 0.0000 |
| Lehmer | 0.0245 | 0.0890 | 0.1500 | 0.8743 | 0.1322 | 0.0000 |

## Comparison To Previous RSK Shape Results

| model | inference | exact acc | row acc | row MAE | total box MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| encoder-decoder rows, Lehmer | argmax | 0.0110 | 0.8733 | 0.1350 | 0.6255 |
| encoder-decoder rows, Lehmer | score all partitions | 0.0230 | 0.8722 | 0.1361 | 0.0000 |
| AR rows, Lehmer | score all partitions | 0.0000 | 0.7183 | 1.3248 | 0.0000 |
| partition classifier, one-line | class argmax | 0.0175 | 0.8666 | 0.1427 | 0.0000 |
| partition classifier, Lehmer | class argmax | 0.0245 | 0.8743 | 0.1322 | 0.0000 |

## Takeaway

Direct 5604-way partition classification is the best exact-shape result so far,
but only by a small margin: Lehmer reaches `0.0245` exact accuracy, compared
with `0.0230` for the non-autoregressive encoder-decoder plus partition scoring.

The top-k numbers are more encouraging. The correct shape is in the top 10 for
`15.00%` of independent test examples with Lehmer input, suggesting that the
model often ranks the right partition nearby even when argmax is wrong.

The next useful step is likely a reranking or structured candidate objective:
use the 5604-way classifier to produce top-k candidates, then train/evaluate a
secondary score or loss that separates nearby partitions.
