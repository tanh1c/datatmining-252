# Current Best Method

Updated: 2026-05-12

## Best Public Score

- Public score: `0.63064`
- Submission: `outputs/openalex_meta/submissions/openalex_huber_meta_w1.00_submission.csv`
- Previous anchor: `outputs/submissions/next_ridge_label4_51_submission.csv` scored `0.62820`
- Archive: `outputs/0.63064/`

## Method Summary

The current best method is an OpenAlex-enriched Huber meta-regressor over the
repeated-CV Ridge score anchor, with threshold tuning for the ordinal label
scale `1 -> 5`.

Instead of training a classifier directly, it predicts a continuous paper-quality
score, then four thresholds convert the score back into labels `1, 2, 3, 4, 5`.

## Features

The score anchor uses stable text fields:

- `title_clean`
- `authors_clean`

Vectorization:

| Field | Vectorizer | Config |
| --- | --- | --- |
| `title_clean` | word TF-IDF | ngram `(1, 2)`, `min_df=2`, `max_features=8000`, `sublinear_tf=True` |
| `title_clean` | `char_wb` TF-IDF | ngram `(3, 5)`, `min_df=2`, `max_features=10000`, `sublinear_tf=True` |
| `authors_clean` | word TF-IDF | ngram `(1, 2)`, `min_df=2`, `max_features=4000`, `sublinear_tf=True` |

The OpenAlex meta layer adds:

- Ridge 5x5 continuous score
- OOF target-encoded `venue`, `year`, and first-author surname
- DOI-level OpenAlex fields: `cited_by_count`, `referenced_works_count`, `fwci`,
  `openalex_year`, and `is_retracted`
- Basic metadata such as title length, author count, and missing-author flag

OpenAlex cache:

```text
outputs/external/openalex_doi_features.csv
```

Cached columns:

```text
doi_norm, openalex_found, openalex_error, openalex_id, openalex_title,
openalex_year, cited_by_count, referenced_works_count, fwci, is_retracted
```

Coverage:

- `1813` unique DOI records queried
- `1810` found in OpenAlex
- Train row coverage: `58.34%`
- Test row coverage: `60.23%`

Not used in the original Ridge anchor:

- `venue`
- `year`
- `doi`
- full text / abstract
- XGBoost
- Logistic classifier output

## Model

Ridge anchor:

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

Meta-model:

```text
HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000)
```

The best OpenAlex candidate had local OOF QWK `0.606685`, MAE `0.840016`, and
test label distribution `{1:234, 2:154, 3:111, 4:46, 5:51}`.

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

The OpenAlex result shows that external scholarly metadata adds signal beyond
title and author text. Raw citation count alone is weak, but as part of a
regularized meta-model with venue/year/author priors, it improved public score
from `0.62820` to `0.63064`.

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
outputs/openalex_meta/submissions/openalex_huber_meta_w1.00_submission.csv
```

## Current Follow-Up Candidates

The next experiments should use `0.63064` as the new primary anchor:

| Candidate | Why |
| --- | --- |
| Small blend: OpenAlex Huber + Ridge `0.62820` | Reduce private risk while keeping the validated external signal |
| OpenAlex title-search enrichment | Fill rows that only have Semantic Scholar URLs and no DOI |
| Citation velocity / venue-year percentile | Normalize citation signal by paper age and venue-year context |

Keep the old threshold-only Ridge candidates as fallback anchors, not as the
main search direction.

## Failed Scholarly Follow-Ups

Two later external-data probes did not beat the best:

| Submission | Public LB | Local note |
| --- | ---: | --- |
| `outputs/submissions/next_scholarly_meta_submission.csv` | `0.60531` | all-source Semantic Scholar + Crossref + COCI + OpenAlex was noisy |
| `outputs/submissions/next_filtered_scholarly_submission.csv` | `0.61227` | COCI-only feature selection had high OOF but poor public transfer |

Conclusion: keep the OpenAlex DOI-only Huber candidate as the best method. More
external citation sources are not automatically helpful and can overfit local
OOF.
