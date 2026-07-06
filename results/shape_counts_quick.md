# Shape Count Quick Run

Dataset:

- all integer partitions of `n = 30`
- total partitions: `5604`
- train/test split: `4483 / 1121`
- device: CPU

Model:

- MLP on padded normalized partition vector
- `hidden = 256`
- `layers = 3`
- `batch_size = 256`
- `epochs = 120`
- target uses log scale

## SYT

Target:

```text
log #SYT(lambda)
```

Formula:

```text
log(n!) - sum_cells log(hook(cell))
```

Final result:

```text
test_log_mae=0.191207
test_log_rmse=0.253992
median_multiplicative_error=1.162439x
mean_relative_error=0.231461
```

Checkpoint:

- `models/shape_counts/syt_n30_mlp_quick.pt`

## SSYT

Target:

```text
log #SSYT(lambda, m)
```

with alphabet size:

```text
m = 30
```

Formula:

```text
sum_cells log(m + column - row) - sum_cells log(hook(cell))
```

Final result:

```text
test_log_mae=0.203174
test_log_rmse=0.282594
median_multiplicative_error=1.168824x
mean_relative_error=0.255880
```

Checkpoint:

- `models/shape_counts/ssyt_n30_m30_mlp_quick.pt`

## Takeaway

These tasks are much smoother than `permutation -> sign`. A small MLP already
gets useful approximate predictions on log counts, but it is not exact. The
largest errors are on unusual rectangular or tall-thin shapes, where hook
structure changes sharply.
