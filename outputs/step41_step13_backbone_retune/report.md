# Step41 Step13 backbone retune

This step retunes the Step13 E5/SciNCL/SPECTER/Ridge backbone while keeping the Step39-style Ridge(alpha=10) calibrator share and fixed BGE-M3 diagnostic at `w=0.0275`.

## Top candidates

| name                                                                |   w_e5 |   w_scincl |   w_specter |   w_ridge |   threshold_lambda |   oof_qwk |   oof_lift_vs_step39 |   oof_lift_vs_step36 |   oof_lift_vs_step25 |   test_l1 |   diff_vs_step39_total |   diff_vs_step36_total |   diff_vs_step25_total |   risky_promotions |   core_demotions |   public_changed |   private_changed |   forbidden_vs_step25 |
|:--------------------------------------------------------------------|-------:|-----------:|------------:|----------:|-------------------:|----------:|---------------------:|---------------------:|---------------------:|----------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-----------------:|-----------------:|------------------:|----------------------:|
| step13retune_e50p50_scincl0p20_specter0p30_ridge0p00_bge0p0275_l4p0 |   0.5  |       0.2  |        0.3  |      0    |                  4 |  0.663141 |          0.000555161 |          0.00104328  |          0.00188071  |  0.155619 |                     16 |                     17 |                     16 |                 24 |               22 |               11 |                12 |                     0 |
| step13retune_e50p40_scincl0p30_specter0p20_ridge0p10_bge0p0275_l4p0 |   0.4  |       0.3  |        0.2  |      0.1  |                  4 |  0.663124 |          0.000537549 |          0.00102567  |          0.0018631   |  0.165686 |                     24 |                     25 |                     28 |                 18 |               22 |               15 |                20 |                     0 |
| step13retune_e50p40_scincl0p30_specter0p20_ridge0p10_bge0p0275_l6p0 |   0.4  |       0.3  |        0.2  |      0.1  |                  6 |  0.663124 |          0.000537549 |          0.00102567  |          0.0018631   |  0.169042 |                     25 |                     26 |                     29 |                 18 |               22 |               15 |                21 |                     0 |
| step13retune_e50p50_scincl0p30_specter0p20_ridge0p00_bge0p0275_l4p0 |   0.5  |       0.3  |        0.2  |      0    |                  4 |  0.662586 |          0           |          0.000488121 |          0.00132555  |  0.152263 |                      0 |                      5 |                      8 |                 13 |               13 |                8 |                 5 |                     0 |
| step13retune_e50p50_scincl0p30_specter0p20_ridge0p00_bge0p0275_l6p0 |   0.5  |       0.3  |        0.2  |      0    |                  6 |  0.662586 |          0           |          0.000488121 |          0.00132555  |  0.152263 |                      0 |                      5 |                      8 |                 13 |               13 |                8 |                 5 |                     0 |
| step13retune_e50p45_scincl0p35_specter0p13_ridge0p07_bge0p0275_l4p0 |   0.45 |       0.35 |        0.13 |      0.07 |                  4 |  0.662406 |         -0.000180499 |          0.000307622 |          0.00114505  |  0.169042 |                     19 |                     20 |                     21 |                 18 |               25 |               12 |                16 |                     0 |
| step13retune_e50p46_scincl0p36_specter0p13_ridge0p05_bge0p0275_l6p0 |   0.46 |       0.36 |        0.13 |      0.05 |                  6 |  0.662356 |         -0.000230037 |          0.000258084 |          0.00109551  |  0.165686 |                     17 |                     18 |                     19 |                 13 |               23 |               12 |                14 |                     0 |
| step13retune_e50p45_scincl0p35_specter0p13_ridge0p07_bge0p0275_l6p0 |   0.45 |       0.35 |        0.13 |      0.07 |                  6 |  0.662215 |         -0.00037119  |          0.000116931 |          0.000954358 |  0.165686 |                     18 |                     19 |                     20 |                 18 |               25 |               11 |                16 |                     0 |
| step13retune_e50p45_scincl0p30_specter0p20_ridge0p05_bge0p0275_l4p0 |   0.45 |       0.3  |        0.2  |      0.05 |                  4 |  0.66194  |         -0.000646195 |         -0.000158073 |          0.000679354 |  0.175753 |                     20 |                     23 |                     22 |                  7 |               24 |               13 |                16 |                     0 |
| step13retune_e50p45_scincl0p37_specter0p13_ridge0p05_bge0p0275_l6p0 |   0.45 |       0.37 |        0.13 |      0.05 |                  6 |  0.661829 |         -0.000756972 |         -0.000268851 |          0.000568576 |  0.165686 |                     17 |                     18 |                     19 |                 13 |               24 |               11 |                15 |                     0 |
| step13retune_e50p35_scincl0p35_specter0p20_ridge0p10_bge0p0275_l4p0 |   0.35 |       0.35 |        0.2  |      0.1  |                  4 |  0.661829 |         -0.000756972 |         -0.000268851 |          0.000568576 |  0.175753 |                     36 |                     39 |                     42 |                 30 |               29 |               23 |                22 |                     1 |
| step13retune_e50p45_scincl0p35_specter0p12_ridge0p08_bge0p0275_l6p0 |   0.45 |       0.35 |        0.12 |      0.08 |                  6 |  0.661768 |         -0.000818462 |         -0.000330341 |          0.000507086 |  0.165686 |                     18 |                     19 |                     20 |                 19 |               25 |               12 |                15 |                     0 |
| step13retune_e50p35_scincl0p35_specter0p20_ridge0p10_bge0p0275_l6p0 |   0.35 |       0.35 |        0.2  |      0.1  |                  6 |  0.661768 |         -0.000818462 |         -0.000330341 |          0.000507086 |  0.172398 |                     34 |                     37 |                     40 |                 30 |               27 |               22 |                21 |                     1 |
| step13retune_e50p46_scincl0p36_specter0p13_ridge0p05_bge0p0275_l4p0 |   0.46 |       0.36 |        0.13 |      0.05 |                  4 |  0.661708 |         -0.000878325 |         -0.000390204 |          0.000447223 |  0.169042 |                     18 |                     19 |                     20 |                 12 |               23 |               13 |                14 |                     0 |
| step13retune_e50p45_scincl0p35_specter0p11_ridge0p09_bge0p0275_l4p0 |   0.45 |       0.35 |        0.11 |      0.09 |                  4 |  0.661699 |         -0.000887514 |         -0.000399393 |          0.000438034 |  0.169042 |                     22 |                     23 |                     26 |                 18 |               25 |               16 |                17 |                     0 |
| step13retune_e50p45_scincl0p35_specter0p11_ridge0p09_bge0p0275_l6p0 |   0.45 |       0.35 |        0.11 |      0.09 |                  6 |  0.661699 |         -0.000887514 |         -0.000399393 |          0.000438034 |  0.169042 |                     22 |                     23 |                     26 |                 18 |               25 |               16 |                17 |                     0 |
| step13retune_e50p45_scincl0p35_specter0p12_ridge0p08_bge0p0275_l4p0 |   0.45 |       0.35 |        0.12 |      0.08 |                  4 |  0.661688 |         -0.000898347 |         -0.000410226 |          0.000427202 |  0.169042 |                     19 |                     20 |                     21 |                 19 |               25 |               13 |                15 |                     0 |
| step13retune_e50p40_scincl0p30_specter0p30_ridge0p00_bge0p0275_l6p0 |   0.4  |       0.3  |        0.3  |      0    |                  6 |  0.661559 |         -0.00102744  |         -0.00053932  |          0.000298108 |  0.175753 |                     26 |                     29 |                     30 |                 20 |               29 |               15 |                16 |                     1 |
| step13retune_e50p50_scincl0p20_specter0p30_ridge0p00_bge0p0275_l6p0 |   0.5  |       0.2  |        0.3  |      0    |                  6 |  0.661549 |         -0.00103663  |         -0.000548508 |          0.000288919 |  0.179109 |                     22 |                     27 |                     28 |                 14 |               31 |               17 |                16 |                     0 |
| step13retune_e50p45_scincl0p35_specter0p10_ridge0p10_bge0p0275_l6p0 |   0.45 |       0.35 |        0.1  |      0.1  |                  6 |  0.661466 |         -0.00111983  |         -0.000631708 |          0.000205719 |  0.165686 |                     25 |                     26 |                     29 |                 20 |               23 |               18 |                18 |                     0 |

## Probe candidates

No probe candidates passed filters.

## Picks

```json
[]
```

## Forced exports

```json
[
  {
    "label": "high_e5_scincl_ridge0",
    "name": "step13retune_e50p55_scincl0p35_specter0p10_ridge0p00_bge0p0275_l4p0",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step41_high_e5_scincl_ridge0_submission.csv",
    "w_e5": 0.55,
    "w_scincl": 0.35,
    "w_specter": 0.1,
    "w_ridge": 0.0,
    "threshold_lambda": 4.0,
    "oof_qwk": 0.6590130599701706,
    "oof_lift_vs_step39": -0.003573046639384736,
    "oof_lift_vs_step36": -0.003084925459252208,
    "oof_lift_vs_step25": -0.002247498409062354,
    "test_l1": 0.16233053287621835,
    "diff_vs_step39_total": 12,
    "diff_vs_step36_total": 13,
    "diff_vs_step25_total": 14
  },
  {
    "label": "higher_e5_ridge0",
    "name": "step13retune_e50p60_scincl0p30_specter0p10_ridge0p00_bge0p0275_l4p0",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step41_higher_e5_ridge0_submission.csv",
    "w_e5": 0.6,
    "w_scincl": 0.3,
    "w_specter": 0.1,
    "w_ridge": 0.0,
    "threshold_lambda": 4.0,
    "oof_qwk": 0.6591275462276434,
    "oof_lift_vs_step39": -0.003458560381911946,
    "oof_lift_vs_step36": -0.002970439201779418,
    "oof_lift_vs_step25": -0.002133012151589564,
    "test_l1": 0.1690419422721915,
    "diff_vs_step39_total": 20,
    "diff_vs_step36_total": 21,
    "diff_vs_step25_total": 18
  },
  {
    "label": "balanced_high_e5_ridge0",
    "name": "step13retune_e50p55_scincl0p30_specter0p15_ridge0p00_bge0p0275_l4p0",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step41_balanced_high_e5_ridge0_submission.csv",
    "w_e5": 0.55,
    "w_scincl": 0.3,
    "w_specter": 0.15,
    "w_ridge": 0.0,
    "threshold_lambda": 4.0,
    "oof_qwk": 0.6590082595716396,
    "oof_lift_vs_step39": -0.003577847037915749,
    "oof_lift_vs_step36": -0.003089725857783221,
    "oof_lift_vs_step25": -0.002252298807593367,
    "test_l1": 0.18246476106413784,
    "diff_vs_step39_total": 18,
    "diff_vs_step36_total": 21,
    "diff_vs_step25_total": 20
  },
  {
    "label": "specter25_from_e5_ridge0",
    "name": "step13retune_e50p45_scincl0p30_specter0p25_ridge0p00_bge0p0275_l4p0",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step41_specter25_from_e5_ridge0_submission.csv",
    "w_e5": 0.45,
    "w_scincl": 0.3,
    "w_specter": 0.25,
    "w_ridge": 0.0,
    "threshold_lambda": 4.0,
    "oof_qwk": 0.6594745825300481,
    "oof_lift_vs_step39": -0.0031115240795072108,
    "oof_lift_vs_step36": -0.0026234028993746827,
    "oof_lift_vs_step25": -0.0017859758491848288,
    "test_l1": 0.17239764697017804,
    "diff_vs_step39_total": 19,
    "diff_vs_step36_total": 20,
    "diff_vs_step25_total": 21
  },
  {
    "label": "specter25_from_scincl_ridge0",
    "name": "step13retune_e50p50_scincl0p25_specter0p25_ridge0p00_bge0p0275_l4p0",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step41_specter25_from_scincl_ridge0_submission.csv",
    "w_e5": 0.5,
    "w_scincl": 0.25,
    "w_specter": 0.25,
    "w_ridge": 0.0,
    "threshold_lambda": 4.0,
    "oof_qwk": 0.6591988160042306,
    "oof_lift_vs_step39": -0.003387290605324722,
    "oof_lift_vs_step36": -0.002899169425192194,
    "oof_lift_vs_step25": -0.00206174237500234,
    "test_l1": 0.18246476106413786,
    "diff_vs_step39_total": 22,
    "diff_vs_step36_total": 25,
    "diff_vs_step25_total": 26
  },
  {
    "label": "specter30_balanced_ridge0",
    "name": "step13retune_e50p40_scincl0p30_specter0p30_ridge0p00_bge0p0275_l4p0",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step41_specter30_balanced_ridge0_submission.csv",
    "w_e5": 0.4,
    "w_scincl": 0.3,
    "w_specter": 0.3,
    "w_ridge": 0.0,
    "threshold_lambda": 4.0,
    "oof_qwk": 0.6610840419866013,
    "oof_lift_vs_step39": -0.0015020646229539913,
    "oof_lift_vs_step36": -0.0010139434428214633,
    "oof_lift_vs_step25": -0.00017651639263160934,
    "test_l1": 0.17575335166816464,
    "diff_vs_step39_total": 26,
    "diff_vs_step36_total": 29,
    "diff_vs_step25_total": 30
  },
  {
    "label": "specter30_from_scincl_ridge0",
    "name": "step13retune_e50p50_scincl0p20_specter0p30_ridge0p00_bge0p0275_l4p0",
    "submission": "C:\\Users\\LG\\Desktop\\Study Material\\DataMining\\Assignment\\outputs\\submissions\\next_step41_specter30_from_scincl_ridge0_submission.csv",
    "w_e5": 0.5,
    "w_scincl": 0.2,
    "w_specter": 0.3,
    "w_ridge": 0.0,
    "threshold_lambda": 4.0,
    "oof_qwk": 0.6631412678277752,
    "oof_lift_vs_step39": 0.0005551612182198573,
    "oof_lift_vs_step36": 0.0010432823983523853,
    "oof_lift_vs_step25": 0.0018807094485422393,
    "test_l1": 0.1556191234802452,
    "diff_vs_step39_total": 16,
    "diff_vs_step36_total": 17,
    "diff_vs_step25_total": 16
  }
]
```
