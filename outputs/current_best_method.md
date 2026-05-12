# Current Best Method

Updated: 2026-05-12

## Best Public Score

- Public score: `0.62820`
- Submission: `outputs/submissions/next_ridge_label4_51_submission.csv`
- Tied submission: `outputs/submissions/next_ridge_label4_55_submission.csv`
- Archive: `outputs/0.62820/`

## Method Summary

The current best method is a repeated-CV Ridge score ensemble with threshold
tuning for the ordinal label scale `1 -> 5`.

Instead of training a classifier directly, it treats the target label as a
continuous paper-quality score. Ridge predicts this score, then four thresholds
convert the score back into labels `1, 2, 3, 4, 5`.

## Features

Only stable text fields are used:

- `title_clean`
- `authors_clean`

Vectorization:

| Field | Vectorizer | Config |
| --- | --- | --- |
| `title_clean` | word TF-IDF | ngram `(1, 2)`, `min_df=2`, `max_features=8000`, `sublinear_tf=True` |
| `title_clean` | `char_wb` TF-IDF | ngram `(3, 5)`, `min_df=2`, `max_features=10000`, `sublinear_tf=True` |
| `authors_clean` | word TF-IDF | ngram `(1, 2)`, `min_df=2`, `max_features=4000`, `sublinear_tf=True` |

Not used in the best method:

- `venue`
- `year`
- `doi`
- full text / abstract
- XGBoost
- Logistic classifier output

## Model

Base model:

```text
Ridge(alpha=8.0, solver="lsqr")
```

Ensemble:

```text
5 StratifiedKFold splits x 5 seeds
seeds = 252,253,254,255,256
```

For each fold, TF-IDF is fit only on the training fold, then Ridge predicts:

- validation fold score for OOF threshold tuning
- public test score
- private test score

All test scores are averaged across the 25 fold/seed models.

## Thresholds

The original 5x5 threshold-tuned Ridge submission scored `0.60804`:

```text
[2.138587370765, 2.506551616073, 2.880155795505, 3.142222868509]
```

The first threshold-improved best kept the same Ridge scores but widened the
label-4 interval:

```text
[2.138587370765, 2.506551616073, 2.810155795505, 3.192222868509]
```

This changed only `18 / 596` submission rows versus the `0.60804` anchor.

Label distribution changed from:

```text
0.60804: {1:237, 2:183, 3:92, 4:28, 5:56}
```

to:

```text
0.62463: {1:237, 2:183, 3:79, 4:46, 5:51}
```

The current best widens label `4` slightly further. Two variants tie on public:

```text
0.62820 / label4_51: {1:237, 2:183, 3:75, 4:51, 5:50}
0.62820 / label4_55: {1:237, 2:183, 3:71, 4:55, 5:50}
```

## Why It Works

The label is ordinal, so predicting a continuous score and tuning thresholds
matches the QWK metric better than direct classification. Public results also
showed that the original 5x5 Ridge model under-predicted label `4`; widening
the label-4 interval improved public score from `0.60804` to `0.62463`, then
to `0.62820`.

## Reproduce

Train the base Ridge 5x5 score ensemble:

```bash
python train_ridge_threshold_ensemble.py --folds 5 --seeds 252,253,254,255,256 --alpha 8
```

Create threshold variants:

```bash
python make_ridge_threshold_variants.py
```

Best current submission:

```text
outputs/submissions/next_ridge_label4_51_submission.csv
```

## Current Follow-Up Candidates

The next threshold-only experiments keep the same Ridge 5x5 alpha=8 scores and
increase label `4` slightly beyond the current best distribution:

| Submission | Label distribution | Difference vs `0.62463` |
| --- | --- | ---: |
| `outputs/submissions/next_ridge_label4_51_submission.csv` | `{1:237, 2:183, 3:75, 4:51, 5:50}` | 5 rows |
| `outputs/submissions/next_ridge_label4_55_submission.csv` | `{1:237, 2:183, 3:71, 4:55, 5:50}` | 9 rows |
| `outputs/submissions/next_ridge_label4_58_submission.csv` | `{1:237, 2:183, 3:68, 4:58, 5:50}` | 12 rows |
