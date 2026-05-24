# Step35B rubric-conditioned reranker

This step trains a generalized rubric-conditioned binary reranker: each paper is paired with each of the five class rubrics, trained OOF on train labels only, converted into an expected label score, and blended lightly with Step25 for public/private uniformly.

## Top candidates — no safe candidate passed filters

| name                                      | kind        |    c |   weight |   threshold_lambda |   oof_qwk |   oof_lift_vs_step25 |   test_l1 |   changed |   improved |   worsened |   public_changed |   private_changed |   risky_promotions |   core_demotions |
|:------------------------------------------|:------------|-----:|---------:|-------------------:|----------:|---------------------:|----------:|----------:|-----------:|-----------:|-----------------:|------------------:|-------------------:|-----------------:|
| step25_rubric_lr_c2p0_blend_w0p0100_l2p0  | label_blend | 2    |   0.01   |                2   |  0.661299 |          3.80054e-05 |  0.145552 |        31 |         12 |         19 |                4 |                 4 |                  4 |                3 |
| step25_rubric_lr_c2p0_blend_w0p0100_l0p5  | label_blend | 2    |   0.01   |                0.5 |  0.660964 |         -0.000296651 |  0.189176 |       107 |         55 |         52 |               16 |                20 |                  0 |               30 |
| step25_rubric_lr_c0p5_blend_w0p0100_l2p0  | label_blend | 0.5  |   0.01   |                2   |  0.660528 |         -0.000732746 |  0.145552 |        37 |         13 |         24 |                5 |                 2 |                  5 |                3 |
| step25_rubric_lr_c2p0_blend_w0p0050_l1p0  | label_blend | 2    |   0.005  |                1   |  0.660121 |         -0.00113944  |  0.189176 |        80 |         38 |         42 |               12 |                16 |                  0 |               15 |
| step25_rubric_lr_c2p0_blend_w0p0300_l0p5  | label_blend | 2    |   0.03   |                0.5 |  0.660091 |         -0.00116969  |  0.169042 |        77 |         34 |         43 |               10 |                15 |                  2 |               14 |
| step25_rubric_lr_c2p0_blend_w0p0200_l1p0  | label_blend | 2    |   0.02   |                1   |  0.659952 |         -0.00130885  |  0.182465 |        71 |         32 |         39 |                9 |                14 |                  1 |               13 |
| step25_rubric_lr_c0p25_blend_w0p0100_l0p5 | label_blend | 0.25 |   0.01   |                0.5 |  0.659831 |         -0.00142991  |  0.192532 |       101 |         48 |         53 |               13 |                20 |                  0 |               25 |
| step25_rubric_lr_c0p5_blend_w0p0300_l0p5  | label_blend | 0.5  |   0.03   |                0.5 |  0.659785 |         -0.00147521  |  0.161768 |        69 |         28 |         41 |               10 |                11 |                  5 |                9 |
| step25_rubric_lr_c1p0_blend_w0p0300_l1p0  | label_blend | 1    |   0.03   |                1   |  0.659684 |         -0.00157698  |  0.172398 |        81 |         34 |         47 |               11 |                12 |                  4 |               12 |
| step25_rubric_lr_c0p5_blend_w0p0100_l0p5  | label_blend | 0.5  |   0.01   |                0.5 |  0.659667 |         -0.0015939   |  0.18582  |        64 |         27 |         37 |                8 |                12 |                  2 |               11 |
| step25_rubric_lr_c2p0_blend_w0p0025_l0p5  | label_blend | 2    |   0.0025 |                0.5 |  0.659637 |         -0.00162331  |  0.172398 |        33 |         13 |         20 |                3 |                 8 |                  1 |                8 |
| step25_rubric_lr_c1p0_blend_w0p0050_l0p5  | label_blend | 1    |   0.005  |                0.5 |  0.659634 |         -0.00162696  |  0.189176 |        90 |         43 |         47 |               12 |                16 |                  0 |               21 |
| step25_rubric_lr_c2p0_blend_w0p0300_l1p0  | label_blend | 2    |   0.03   |                1   |  0.659627 |         -0.00163386  |  0.182465 |        80 |         37 |         43 |               11 |                18 |                  0 |               20 |
| step25_rubric_lr_c1p0_blend_w0p0100_l0p5  | label_blend | 1    |   0.01   |                0.5 |  0.659519 |         -0.00174109  |  0.192532 |       104 |         49 |         55 |               13 |                21 |                  0 |               23 |
| step25_rubric_lr_c2p0_blend_w0p0200_l2p0  | label_blend | 2    |   0.02   |                2   |  0.659487 |         -0.00177331  |  0.179109 |        73 |         33 |         40 |               10 |                15 |                  0 |               16 |
| step25_rubric_lr_c0p5_blend_w0p0200_l0p5  | label_blend | 0.5  |   0.02   |                0.5 |  0.659479 |         -0.00178139  |  0.189176 |        78 |         35 |         43 |               12 |                17 |                  0 |               19 |
| step25_rubric_lr_c2p0_blend_w0p0050_l0p5  | label_blend | 2    |   0.005  |                0.5 |  0.659441 |         -0.00181915  |  0.192532 |        92 |         43 |         49 |               13 |                16 |                  0 |               18 |
| step25_rubric_lr_c1p0_blend_w0p0025_l0p5  | label_blend | 1    |   0.0025 |                0.5 |  0.659425 |         -0.00183512  |  0.172398 |        47 |         21 |         26 |                5 |                10 |                  0 |               10 |
| step25_rubric_lr_c2p0_blend_w0p0075_l0p5  | label_blend | 2    |   0.0075 |                0.5 |  0.659378 |         -0.00188219  |  0.175753 |        50 |         22 |         28 |                6 |                11 |                  1 |               11 |
| step25_rubric_lr_c2p0_blend_w0p0050_l2p0  | label_blend | 2    |   0.005  |                2   |  0.659367 |         -0.00189331  |  0.18582  |        71 |         33 |         38 |               10 |                16 |                  0 |               12 |

## Picks

```json
[]
```
