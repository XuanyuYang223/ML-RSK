# Length + Sign Multi-Task Ablation

Dataset:

- `data/rsk_shape/rsk_n30_m100000_seed2.npz`
- `max_samples = 20000`
- split: `16000 / 4000`
- device: CPU

Model:

- Transformer encoder with CLS pooling
- shared trunk
- length regression head
- sign classification head
- `d_model = 64`
- `num_layers = 2`
- `num_heads = 4`
- `epochs = 8`

| representation | final test length MAE | final test sign acc |
| --- | ---: | ---: |
| one-line | 20.242 | 0.4933 |
| Lehmer | 10.220 | 0.4855 |

Takeaway:

Multi-task training improves the length signal more with Lehmer input, but it
does not improve sign/parity. The model can approximate length without learning
length modulo 2.
