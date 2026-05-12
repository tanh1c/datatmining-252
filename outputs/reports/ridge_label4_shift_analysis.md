# Ridge Label-4 Shift Analysis

Purpose: compare the public-proven `0.62463` threshold variant against the two tied `0.62820` variants and decide whether further label-4 tuning is evidence-based or likely public overfit.

## Key Findings

- The public-score jump from `0.62463` to `0.62820` is supported by only one visible public-test change: id `1486`, `3->4`.
- Most changed rows are in `private_test`, so the public leaderboard cannot tell whether `label4_51` or `label4_55` is better for final scoring.
- `label4_51` and `label4_55` differ on only four rows, all in `private_test` (`2156`, `1545`, `1955`, `878`), each as `3->4`.
- Both `0.62820` submissions make no `4->5` promotions. They mainly widen label `4` by moving borderline `3`s up and one borderline `5` down.
- The changed rows are concentrated in a very narrow score band near thresholds, so this looks like threshold calibration rather than model instability.

## Compared Submissions

| Candidate | Public LB | Threshold effect | Distribution |
| --- | ---: | --- | --- |
| `next_ridge_label4_more_mid_submission.csv` | 0.62463 | baseline for this analysis | `{1:237, 2:183, 3:79, 4:46, 5:51}` |
| `next_ridge_label4_51_submission.csv` | 0.62820 | lower 3/4 threshold slightly, raise 4/5 threshold slightly | `{1:237, 2:183, 3:75, 4:51, 5:50}` |
| `next_ridge_label4_55_submission.csv` | 0.62820 | lower 3/4 threshold more, raise 4/5 threshold slightly | `{1:237, 2:183, 3:71, 4:55, 5:50}` |

## Shift Summary: `label4_51_0.62820`

- Changed rows versus `0.62463`: `5`
- Split counts: `{'private_test': 4, 'public_test': 1}`
- Transition counts: `{'3->4': 4, '5->4': 1}`
- Score range among changed rows: `2.8025` to `3.1935`
- Median score among changed rows: `2.8051`

### `3->4` Rows

| Split | ID | Score | Venue | Year | First author | Venue mean/count | Author mean/count | Nearest train labels | Title |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |
| private_test | 1133 | 2.8025 | kr | 2018 | Patrick Koopmann | 2.73/661 | 4.00/1 | L5@0.31, L2@0.26, L1@0.25 | Query Answering for Rough EL Ontologies. |
| private_test | 2844 | 2.8051 | lics | 2018 | Raphaëlle Crubillé | 1.94/604 |  | L1@0.19, L4@0.18, L2@0.16 | Probabilistic Stable Functions on Discrete Cones are Power Series. |
| private_test | 1255 | 2.8062 | kr | 2016 | John Dinsmore | 2.73/661 |  | L4@0.33, L2@0.24, L2@0.22 | Cognitive Affordance Representations in Uncertain Logic. |
| public_test | 1486 | 2.8037 | cav | 2025 | Mate Soos | 2.12/657 | 2.00/2 | L3@0.53, L1@0.36, L4@0.33 | Engineering an Efficient Probabilistic Exact Model Counter. |

### `4->5` Rows

- None.

### `5->4` Rows

| Split | ID | Score | Venue | Year | First author | Venue mean/count | Author mean/count | Nearest train labels | Title |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |
| private_test | 903 | 3.1935 | kr | 2020 | Hong Liu | 2.73/661 | 2.92/12 | L4@0.63, L4@0.27, L5@0.26 | An Evolutionary Algorithm for Rule Learning over Knowledge Graphs. |

## Shift Summary: `label4_55_0.62820`

- Changed rows versus `0.62463`: `9`
- Split counts: `{'private_test': 8, 'public_test': 1}`
- Transition counts: `{'3->4': 8, '5->4': 1}`
- Score range among changed rows: `2.7906` to `3.1935`
- Median score among changed rows: `2.8025`

### `3->4` Rows

| Split | ID | Score | Venue | Year | First author | Venue mean/count | Author mean/count | Nearest train labels | Title |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |
| private_test | 1955 | 2.7906 | cav | 2020 | Qiao Wang | 2.12/657 | 2.50/16 | L1@0.23, L5@0.21, L3@0.19 | Sequence Prediction-based Proactive Caching in Vehicular Content Networks. |
| private_test | 878 | 2.7944 | kr | 2021 | Zeynep G. Saribatur | 2.73/661 | 4.50/2 | L5@0.53, L4@0.48, L5@0.24 | Existential Abstraction on Argumentation Frameworks via Clustering. |
| private_test | 2156 | 2.7970 | cav | 2017 |  | 2.12/657 | 2.29/213 | L2@0.21, L4@0.18, L1@0.18 | Learning a Static Analyzer from Data. |
| private_test | 1545 | 2.7997 | cav | 2024 | Yi Lin | 2.12/657 | 3.00/2 | L2@0.32, L1@0.22, L3@0.20 | Dynamic Programming for Symbolic Boolean Realizability and Synthesis. |
| private_test | 1133 | 2.8025 | kr | 2018 | Patrick Koopmann | 2.73/661 | 4.00/1 | L5@0.31, L2@0.26, L1@0.25 | Query Answering for Rough EL Ontologies. |
| private_test | 2844 | 2.8051 | lics | 2018 | Raphaëlle Crubillé | 1.94/604 |  | L1@0.19, L4@0.18, L2@0.16 | Probabilistic Stable Functions on Discrete Cones are Power Series. |
| private_test | 1255 | 2.8062 | kr | 2016 | John Dinsmore | 2.73/661 |  | L4@0.33, L2@0.24, L2@0.22 | Cognitive Affordance Representations in Uncertain Logic. |
| public_test | 1486 | 2.8037 | cav | 2025 | Mate Soos | 2.12/657 | 2.00/2 | L3@0.53, L1@0.36, L4@0.33 | Engineering an Efficient Probabilistic Exact Model Counter. |

### `4->5` Rows

- None.

### `5->4` Rows

| Split | ID | Score | Venue | Year | First author | Venue mean/count | Author mean/count | Nearest train labels | Title |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |
| private_test | 903 | 3.1935 | kr | 2020 | Hong Liu | 2.73/661 | 2.92/12 | L4@0.63, L4@0.27, L5@0.26 | An Evolutionary Algorithm for Rule Learning over Knowledge Graphs. |

## Pattern Read: `label4_51_0.62820`

- `3->4`: 4 rows. `0` have venue prior >= 3.0; `1` have a repeated first-author surname; `4` have at least one nearest title/author neighbor labeled 4 or 5.
- `5->4`: 1 row(s). `0` have venue prior >= 3.0; `1` have at least one nearest neighbor labeled 4 or 5. This is a mild safety move, not an aggressive downgrade.

## Pattern Read: `label4_55_0.62820`

- `3->4`: 8 rows. `0` have venue prior >= 3.0; `5` have a repeated first-author surname; `7` have at least one nearest title/author neighbor labeled 4 or 5.
- `5->4`: 1 row(s). `0` have venue prior >= 3.0; `1` have at least one nearest neighbor labeled 4 or 5. This is a mild safety move, not an aggressive downgrade.

## Boundary Scan

- Around the `3/4` boundary (`2.76..2.84`) there are `17` rows: `3` public and `14` private.
- The confirmed public-impact row is id `1486` at score `2.8037`. The next higher public rows are id `2149` at `2.8312` and id `1321` at `2.8355`, so small downward moves from `2.8002` mostly affect private rows first.
- Around the `4/5` boundary (`3.17..3.23`) there are `4` rows: `1` public and `3` private. The public row id `1579` is at `3.2251`, much higher than the current `4/5` threshold `3.2022`, so small upward moves of the `4/5` threshold mostly affect private rows first.
- This explains why `label4_51` and `label4_55` tie on public LB: their extra differences are private-only.

## Conclusion

- The `0.62820` gain is not caused by broad random reshuffling: it changes only a small band of borderline continuous scores near the `3/4` and `4/5` thresholds.
- There are no `4->5` promotions versus `0.62463`; the improvement came from moving borderline `3`s into `4` while pulling one borderline `5` down to `4`.
- This supports continuing narrow threshold experiments around label 4, but the search should stay small because public LB only validates the public half.
- Recommended next tests: label-4 counts around `49`, `50`, `56`, and `57`; avoid changing thresholds `1/2` and `2/3` unless OOF evidence improves too.

Detail CSV: `outputs/reports/ridge_label4_shift_details.csv`
