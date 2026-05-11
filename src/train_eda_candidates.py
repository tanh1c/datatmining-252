from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.eda_analysis import add_eda_columns, make_logreg_pipeline, text_variant
from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths
from src.train_tfidf_linear_baselines import write_submission


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EDA-driven Logistic TF-IDF candidates.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/eda_candidates/submissions")
    parser.add_argument("--report-dir", default="outputs/eda_candidates/reports")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train, public, private, sample = load_data(paths)
    train = add_eda_columns(train).reset_index(drop=True)
    public = add_eda_columns(public).reset_index(drop=True)
    private = add_eda_columns(private).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    if not output_dir.is_absolute():
        output_dir = paths.project_root / output_dir
    if not report_dir.is_absolute():
        report_dir = paths.project_root / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("title_first_author_word_char", "title_first_author", True),
        ("title_venue_year_word_char", "title_venue_year", True),
        ("title_x2_word_char", "title_x2", True),
        ("title_only_word_char", "title_only", True),
    ]

    y = train[LABEL_COLUMN].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, variant, word_char in candidates:
        model = make_logreg_pipeline(word_char=word_char)
        x_train = text_variant(train, variant)
        x_public = text_variant(public, variant)
        x_private = text_variant(private, variant)

        oof_pred = cross_val_predict(model, x_train, y, cv=cv, n_jobs=None)
        cv_accuracy = accuracy_score(y, oof_pred)
        cv_qwk = cohen_kappa_score(y, oof_pred, weights="quadratic")

        model.fit(x_train, y)
        public_pred = model.predict(x_public).astype(int)
        private_pred = model.predict(x_private).astype(int)
        public_path, private_path, combined_path = write_submission(
            name,
            public,
            private,
            public_pred,
            private_pred,
            sample,
            output_dir,
        )
        rows.append(
            {
                "name": name,
                "variant": variant,
                "vectorizer": "word_char" if word_char else "word",
                "cv_accuracy": cv_accuracy,
                "cv_qwk": cv_qwk,
                "public_submission": public_path,
                "private_submission": private_path,
                "combined_submission": combined_path,
            }
        )
        print(f"{name}: cv_qwk={cv_qwk:.6f}, submission={combined_path}")

    report = pd.DataFrame(rows).sort_values("cv_qwk", ascending=False)
    report_path = report_dir / "eda_candidates_report.csv"
    report.to_csv(report_path, index=False)

    best = report.iloc[0]
    best_target = paths.project_root / "outputs" / "submissions" / "next_eda_best_submission.csv"
    best_target.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(best["combined_submission"]).to_csv(best_target, index=False)
    print(f"Best candidate: {best['name']} cv_qwk={best['cv_qwk']:.6f}")
    print(f"Best copy: {best_target}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
