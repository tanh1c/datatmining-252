# Repository Guidelines

## Project Structure & Module Organization

This repository contains a compact Kaggle-style data mining workflow for ASP paper classification.

- `data/raw/`: immutable input CSV files (`train.csv`, `public_test.csv`, `private_test.csv`, `Test_Submission.csv`).
- `src/`: maintained Python source code.
  - `train_randomforest_best.py`: RandomForest baseline.
  - `train_tfidf_linear_baselines.py`: TF-IDF linear, Ridge, SVM, and blend experiments.
- Root scripts (`train_randomforest_best.py`, `train_tfidf_linear_baselines.py`) are thin entrypoints for convenience.
- `outputs/`: generated reports, submissions, and score checkpoints such as `outputs/0.48078/`.
- `README.md`: project usage summary.
- `requirements.txt`: Python dependencies.

Keep raw data unchanged. Put generated files under `outputs/`, not beside source files.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the RandomForest baseline:

```bash
python train_randomforest_best.py
```

Run a fast RandomForest smoke test:

```bash
python train_randomforest_best.py --n-estimators 20 --skip-cv
```

Run linear TF-IDF experiments:

```bash
python train_tfidf_linear_baselines.py
```

Outputs are written to `outputs/submissions/`, `outputs/reports/`, and `outputs/linear_baselines/`.

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, type hints where practical, and small functions with explicit inputs. Prefer `Path` over raw string paths. Keep constants uppercase (`RANDOM_STATE`, `LABEL_COLUMN`) and functions in `snake_case`.

Submission filenames should describe the model, for example `logreg_word_char_submission.csv` or `blend_top3_submission.csv`.

## Testing Guidelines

There is no formal test suite yet. Validate changes by running at least one smoke command and checking output shape/order:

```bash
python train_randomforest_best.py --n-estimators 20 --skip-cv
python train_tfidf_linear_baselines.py
```

Combined submissions must contain 596 rows, columns `id,Label`, labels in `1..5`, and ids ordered like `data/raw/Test_Submission.csv`.

## Commit & Pull Request Guidelines

Use clear conventional-style commit messages, such as `feat: add logistic baseline` or `docs: track leaderboard results`.

Pull requests should include the changed scripts, command outputs or CV scores, generated submission path, and any Kaggle public score if submitted.

## Leaderboard Tracking

Record public leaderboard results in `outputs/leaderboard_tracking.md` and archive important submissions under score-named folders such as `outputs/0.48078/`. Public leaderboard covers only about 50% of test data, so avoid choosing models from public score alone.
