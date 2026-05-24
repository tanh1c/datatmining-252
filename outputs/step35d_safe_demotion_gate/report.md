# Step35D safe demotion gate

This step restricts the Step35C rubric signal to one-step demotions in safe over-promotion contexts, while protecting ICLP/KR/LICS/LPNMR/core logic rows unless they have explicit meta/CAV/formal cues.

## Top safe candidates

|   c |   margin_cap |   conf_gap |   max_rows | require_meta_or_cav   | protect_all_logic   |   oof_qwk |   oof_lift_vs_step25 |   test_l1 |   changed |   improved |   worsened |   protected_demotions |   public_changed |   private_changed |
|----:|-------------:|-----------:|-----------:|:----------------------|:--------------------|----------:|---------------------:|----------:|----------:|-----------:|-----------:|----------------------:|-----------------:|------------------:|
|   1 |        0.025 |        0.1 |          3 | False                 | True                |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                1 |                 0 |
|   1 |        0.025 |        0.1 |          3 | False                 | False               |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                1 |                 0 |
|   1 |        0.025 |        0.1 |          3 | True                  | True                |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                1 |                 0 |
|   1 |        0.025 |        0.1 |          3 | True                  | False               |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                1 |                 0 |
|   2 |        0.025 |        0.1 |          3 | False                 | True                |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                2 |                 0 |
|   2 |        0.025 |        0.1 |          3 | False                 | False               |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                2 |                 0 |
|   2 |        0.025 |        0.1 |          3 | True                  | True                |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                2 |                 0 |
|   2 |        0.025 |        0.1 |          3 | True                  | False               |  0.661699 |          0.000438494 |  0.145552 |         3 |          3 |          0 |                     0 |                2 |                 0 |
|   1 |        0.015 |        0.1 |          3 | False                 | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          3 | True                  | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          3 | True                  | False               |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          4 | False                 | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          4 | True                  | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          4 | True                  | False               |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          5 | False                 | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          5 | True                  | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          5 | True                  | False               |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          6 | False                 | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          6 | True                  | True                |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |
|   1 |        0.015 |        0.1 |          6 | True                  | False               |  0.661531 |          0.000270845 |  0.145552 |         2 |          2 |          0 |                     0 |                1 |                 0 |

## Picks

```json
[
  {
    "label": "best_safe",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step35d_best_safe_submission.csv",
    "oof_qwk": 0.6616990523245885,
    "oof_lift_vs_step25": 0.00043849394535555586,
    "test_l1": 0.14555200938628546,
    "changed": 1
  },
  {
    "label": "second_safe",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step35d_second_safe_submission.csv",
    "oof_qwk": 0.6616990523245885,
    "oof_lift_vs_step25": 0.00043849394535555586,
    "test_l1": 0.14555200938628546,
    "changed": 1
  }
]
```
