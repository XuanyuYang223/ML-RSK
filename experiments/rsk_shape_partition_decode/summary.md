# RSK Shape Partition Decoding Check

Dataset:

- `data/rsk_shape/rsk_n30_m2000_seed1_test.npz`
- `n = 30`
- examples: `2000`
- device: CPU

Checkpoints:

- `experiments/rsk_shape_scale_100k_e5/baseline_one_line_encdec_rows_argmax/model.pt`
- `experiments/rsk_shape_scale_100k_e5/lehmer_encdec_rows_argmax/model.pt`

Method:

- Reuse the trained row-output Transformer logits.
- Compare plain per-row argmax with constrained `partition` inference.
- `partition` inference scores every integer partition of 30 and selects the legal shape with the largest summed row logit score.

| representation | inference | row acc | exact acc | row MAE | total box MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| one-line | argmax | 0.8699 | 0.0105 | 0.1399 | 0.6970 |
| one-line | monotone | 0.8699 | 0.0105 | 0.1399 | 0.6970 |
| one-line | partition | 0.8693 | 0.0235 | 0.1402 | 0.0000 |
| Lehmer | argmax | 0.8733 | 0.0110 | 0.1350 | 0.6255 |
| Lehmer | monotone | 0.8733 | 0.0110 | 0.1350 | 0.6255 |
| Lehmer | partition | 0.8722 | 0.0230 | 0.1361 | 0.0000 |

Takeaway:

Partition-constrained decoding roughly doubles exact shape accuracy and forces
the total number of boxes to be correct. Row-wise accuracy and MAE barely move,
so legality constraints alone do not solve the RSK shape task. The next useful
RSK direction is probably a structured training objective or autoregressive
shape model, not just post-hoc projection.
