# Shape Count Feature Ablation

Feature sets:

- `rows`: padded normalized row lengths.
- `rows_cols`: rows plus conjugate partition column lengths.
- `rows_cols_hooks`: rows, columns, and hook-length summary statistics. This is formula-inspired.

| task | feature set | test log MAE | test log RMSE | median mult err | mean rel err |
| --- | --- | ---: | ---: | ---: | ---: |
| syt | rows | 2.142600 | 0.253992 | 1.162439x | 0.231461 |
| syt | rows_cols | 1.557900 | 0.218909 | 1.129941x | 0.188450 |
| syt | rows_cols_hooks | 2.594800 | 0.269272 | 1.279032x | 0.290168 |
| ssyt | rows | 2.248500 | 0.282594 | 1.168824x | 0.255880 |
| ssyt | rows_cols | 2.743400 | 0.269989 | 1.188848x | 0.251482 |
| ssyt | rows_cols_hooks | 3.168600 | 0.416039 | 1.506231x | 0.506494 |

Takeaway:

Hook summaries are expected to be very strong because the exact formulas depend on hook lengths.
