# Data Mining Assignment - Stage 2

This project trains a Random Forest baseline for the ASP research-paper
classification task. The goal is to predict the ordinal `Label` value from 1
to 5 for papers in the public and private test sets.

## Dataset

The competition data contains research-paper metadata:

- `id`
- `title`
- `venue`
- `year`
- `authors`
- `doi`
- `Label` in `train.csv` only

Stage 2 files:

- `data/raw/train.csv`: 2494 labeled training samples
- `data/raw/public_test.csv`: 298 unlabeled public-test samples
- `data/raw/private_test.csv`: 298 unlabeled private-test samples
- `data/raw/Test_Submission.csv`: sample submission ids for both test sets

Submissions are evaluated with Quadratic Weighted Kappa (QWK).

## Project Structure

```text
.
├── data/
│   └── raw/
│       ├── train.csv
│       ├── public_test.csv
│       ├── private_test.csv
│       └── Test_Submission.csv
├── outputs/
│   ├── reports/
│   └── submissions/
├── src/
│   └── train_randomforest_best.py
├── train_randomforest_best.py
├── requirements.txt
└── README.md
```

`train_randomforest_best.py` at the project root is kept as a compatibility
wrapper. The maintained training code lives in `src/train_randomforest_best.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train_randomforest_best.py
```

For a quicker smoke test:

```bash
python train_randomforest_best.py --n-estimators 20 --skip-cv
```

To run the TF-IDF linear/Ridge experiments and CV-selected blends:

```bash
python train_tfidf_linear_baselines.py
```

To run EDA and EDA-driven Logistic candidates:

```bash
python run_eda.py
python train_eda_candidates.py
python train_author_venue_candidates.py
python train_best_tuning.py
python train_xgboost_candidates.py
python train_ridge_threshold_ensemble.py
python make_ridge_threshold_variants.py
```

To run the OpenAlex external-metadata pipeline and private-safe blends:

```bash
python train_openalex_title_meta.py
```

This resumes `outputs/external/openalex_title_features.csv`, writes enriched
OpenAlex reports to `outputs/openalex_title_meta/`, and writes blend candidates
to `outputs/private_safe_blends/`.

## Outputs

The script writes:

- `outputs/submissions/randomforest_public_submission.csv`
- `outputs/submissions/randomforest_private_submission.csv`
- `outputs/submissions/randomforest_submission.csv`
- `outputs/reports/randomforest_report.txt`

The linear-baseline script writes:

- `outputs/linear_baselines/submissions/*_submission.csv`
- `outputs/linear_baselines/reports/linear_baselines_report.csv`
- `outputs/submissions/best_cv_submission.csv`

The EDA scripts write:

- `outputs/eda/eda_report.md`
- `outputs/eda/*.csv`
- `outputs/eda_candidates/reports/eda_candidates_report.csv`
- `outputs/submissions/next_eda_best_submission.csv`
- `outputs/author_venue_candidates/reports/author_venue_candidates_report.csv`
- `outputs/submissions/next_author_venue_best_submission.csv`
- `outputs/best_tuning/reports/best_tuning_report.csv`
- `outputs/best_tuning/reports/first_author_bias_audit.md`
- `outputs/submissions/next_best_tuned_submission.csv`
- `outputs/xgboost_candidates/reports/xgboost_candidates_report.csv`
- `outputs/submissions/next_xgboost_best_submission.csv`
- `outputs/ridge_threshold/reports/*_metrics.csv`
- `outputs/submissions/next_ridge_threshold_submission.csv`
- `outputs/ridge_threshold_variants/reports/ridge_threshold_variants_report.csv`
- `outputs/submissions/next_ridge_threshold_variant_submission.csv`
- `outputs/external/openalex_title_features.csv`
- `outputs/openalex_title_meta/reports/openalex_title_meta_candidates.csv`
- `outputs/private_safe_blends/reports/private_safe_blend_candidates.csv`
- `outputs/submissions/next_openalex_title_meta_submission.csv`
- `outputs/submissions/next_private_safe_blend_submission.csv`

The combined submission is reordered to match `data/raw/Test_Submission.csv`.

The submitted Random Forest result with leaderboard score `0.46096` is archived
in `outputs/0.46096/`.

## Current Baseline

The baseline builds a compact text feature from:

- title repeated twice
- venue token repeated twice
- authors
- year token

It then trains a TF-IDF plus Random Forest pipeline and reports cross-validation
accuracy and QWK on the training data before fitting on all labeled rows.
