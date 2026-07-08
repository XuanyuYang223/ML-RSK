# Permutation Invariant Ablation

| name | task | mode | representation | test acc | test MAE | test RMSE | rounded acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| length_classification_one_line | length | classification | one_line | 0.0158 | 22.854 | nan | nan |
| length_classification_lehmer | length | classification | lehmer | 0.0230 | 16.653 | nan | nan |
| length_regression_one_line | length | regression | one_line | nan | 15.418 | 19.246 | 0.0180 |
| length_regression_lehmer | length | regression | lehmer | nan | 6.532 | 8.131 | 0.0523 |
| sign_classification_one_line | sign | classification | one_line | 0.5097 | nan | nan | nan |
| sign_classification_lehmer | sign | classification | lehmer | 0.5070 | nan | nan | nan |

Notes:

- Lehmer code is a deterministic representation of the permutation.
- For Coxeter length, `sum(Lehmer code)` equals the target, so this is a strong inductive bias.
