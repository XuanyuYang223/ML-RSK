# Shape Count Feature Ablation

Feature sets:

- `rows`: padded normalized row lengths.
- `rows_cols`: rows plus conjugate partition column lengths.
- `rows_cols_hooks`: rows, columns, and hook-length summary statistics. This is formula-inspired.

| task | feature set | test log MAE | test log RMSE | median mult err | mean rel err |
| --- | --- | ---: | ---: | ---: | ---: |
| ssyt | rows | 0.203174 | 0.282594 | 1.168824x | 0.255880 |
| ssyt | rows_cols | 0.155012 | 0.207518 | 1.130679x | 0.181050 |
| ssyt | rows_cols_hooks | 0.088561 | 0.117814 | 1.070252x | 0.096051 |
| syt | rows | 0.183157 | 0.255475 | 1.149938x | 0.227053 |
| syt | rows_cols | 0.133851 | 0.177284 | 1.116251x | 0.152391 |
| syt | rows_cols_hooks | 0.100804 | 0.137071 | 1.084223x | 0.111316 |

Takeaway:

Hook summaries are expected to be very strong because the exact formulas depend on hook lengths.
