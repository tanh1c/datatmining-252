# Step35A existing diagnostic gate

This step reuses existing OOF/public/private diagnostic score artifacts as lightweight rubric/pairwise-style judges over Step25. It applies one generalized blend rule uniformly to train/public/private and only writes submissions for candidates passing safety gates.

## Loaded diagnostics

```json
[
  "deberta_v3",
  "llm_zeroshot",
  "llm_zeroshot_v2",
  "modernbert_best",
  "modernbert_second",
  "openalex_anchor",
  "pairwise_e5",
  "pairwise_specter2",
  "scibert_finetune",
  "specter2_finetune"
]
```

## Top candidates — no safe candidate passed filters

| name                                        | kind           |   weight |   threshold_lambda |   oof_qwk |   oof_lift_vs_step25 |   test_l1 |   changed |   improved |   worsened |   public_changed |   private_changed |   risky_promotions |   core_demotions |
|:--------------------------------------------|:---------------|---------:|-------------------:|----------:|---------------------:|----------:|----------:|-----------:|-----------:|-----------------:|------------------:|-------------------:|-----------------:|
| step25_pairwise_specter2_blend_w0p0100_l0p5 | label_blend    |   0.01   |                0.5 |  0.661437 |          0.000176199 |  0.169042 |        32 |         15 |         17 |                3 |                 6 |                  0 |                6 |
| step25_pairwise_specter2_resid_w0p0100_l0p5 | residual_blend |   0.01   |                0.5 |  0.661437 |          0.000176199 |  0.169042 |        32 |         15 |         17 |                3 |                 6 |                  0 |                6 |
| step25_openalex_anchor_blend_w0p0150_l0p5   | label_blend    |   0.015  |                0.5 |  0.661316 |          5.58642e-05 |  0.18582  |        81 |         42 |         39 |               14 |                20 |                  0 |               16 |
| step25_openalex_anchor_resid_w0p0150_l0p5   | residual_blend |   0.015  |                0.5 |  0.661316 |          5.58642e-05 |  0.18582  |        81 |         42 |         39 |               14 |                20 |                  0 |               16 |
| step25_deberta_v3_blend_w0p0200_l1p0        | label_blend    |   0.02   |                1   |  0.661146 |         -0.000114791 |  0.162331 |        37 |         17 |         20 |                6 |                 8 |                  0 |                4 |
| step25_deberta_v3_resid_w0p0200_l1p0        | residual_blend |   0.02   |                1   |  0.661146 |         -0.000114791 |  0.162331 |        37 |         17 |         20 |                6 |                 8 |                  0 |                4 |
| step25_pairwise_e5_blend_w0p0200_l0p5       | label_blend    |   0.02   |                0.5 |  0.661065 |         -0.000195896 |  0.172398 |        75 |         40 |         35 |                9 |                18 |                  1 |               23 |
| step25_pairwise_e5_resid_w0p0200_l0p5       | residual_blend |   0.02   |                0.5 |  0.661065 |         -0.000195896 |  0.172398 |        75 |         40 |         35 |                9 |                18 |                  1 |               23 |
| step25_openalex_anchor_blend_w0p0300_l2p0   | label_blend    |   0.03   |                2   |  0.661054 |         -0.000207018 |  0.182465 |        62 |         31 |         31 |               13 |                14 |                  0 |                1 |
| step25_pairwise_e5_blend_w0p0150_l0p5       | label_blend    |   0.015  |                0.5 |  0.660977 |         -0.000283595 |  0.192532 |        95 |         50 |         45 |               13 |                18 |                  0 |               25 |
| step25_pairwise_e5_resid_w0p0150_l0p5       | residual_blend |   0.015  |                0.5 |  0.660977 |         -0.000283595 |  0.192532 |        95 |         50 |         45 |               13 |                18 |                  0 |               25 |
| step25_deberta_v3_blend_w0p0100_l0p5        | label_blend    |   0.01   |                0.5 |  0.66092  |         -0.000341048 |  0.161768 |        44 |         21 |         23 |                3 |                11 |                  0 |                7 |
| step25_deberta_v3_resid_w0p0100_l0p5        | residual_blend |   0.01   |                0.5 |  0.66092  |         -0.000341048 |  0.161768 |        44 |         21 |         23 |                3 |                11 |                  0 |                7 |
| step25_pairwise_e5_blend_w0p0300_l1p0       | label_blend    |   0.03   |                1   |  0.660906 |         -0.000354818 |  0.179109 |        63 |         30 |         33 |                8 |                14 |                  1 |               11 |
| step25_pairwise_e5_blend_w0p0300_l2p0       | label_blend    |   0.03   |                2   |  0.660855 |         -0.000405267 |  0.182465 |        59 |         29 |         30 |                8 |                15 |                  1 |                9 |
| step25_openalex_anchor_blend_w0p0075_l2p0   | label_blend    |   0.0075 |                2   |  0.660821 |         -0.000439302 |  0.179109 |        45 |         22 |         23 |                6 |                11 |                  0 |                6 |
| step25_openalex_anchor_resid_w0p0075_l2p0   | residual_blend |   0.0075 |                2   |  0.660821 |         -0.000439302 |  0.179109 |        45 |         22 |         23 |                6 |                11 |                  0 |                6 |
| step25_openalex_anchor_blend_w0p0200_l4p0   | label_blend    |   0.02   |                4   |  0.660784 |         -0.000476129 |  0.179109 |        59 |         29 |         30 |               10 |                13 |                  0 |                4 |
| step25_openalex_anchor_resid_w0p0200_l4p0   | residual_blend |   0.02   |                4   |  0.660784 |         -0.000476129 |  0.179109 |        59 |         29 |         30 |               10 |                13 |                  0 |                4 |
| step25_openalex_anchor_blend_w0p0300_l0p5   | label_blend    |   0.03   |                0.5 |  0.660752 |         -0.000508819 |  0.182465 |        62 |         31 |         31 |               13 |                15 |                  0 |                1 |

## Picks

```json
[]
```
