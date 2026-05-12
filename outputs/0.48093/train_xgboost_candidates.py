from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.eda_analysis import add_eda_columns
from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths
from src.train_tfidf_linear_baselines import write_submission


RANDOM_STATE = 42


@dataclass(frozen=True)
class XGBCandidate:
    name: str
    variant: str
    max_features: int = 10000
    max_depth: int = 2
    learning_rate: float = 0.05
    n_estimators: int = 350
    subsample: float = 0.85
    colsample_bytree: float = 0.75
    reg_lambda: float = 6.0
    reg_alpha: float = 0.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = add_eda_columns(df)
    out["title_text"] = out["title"].fillna("").astype(str)
    return out


def make_text(df: pd.DataFrame, variant: str) -> pd.Series:
    title = df["title_text"]
    year = df["year_token"]
    venue = df["venue_token"]
    first = df["first_author_token"]

    if variant == "title_year":
        return title + " " + year
    if variant == "title_x2_year":
        return title + " " + title + " " + year
    if variant == "title_first_surname":
        return title + " " + title + " " + first
    if variant == "title_first_surname_year":
        return title + " " + title + " " + first + " " + year
    if variant == "title_first_surname_venue":
        return title + " " + title + " " + first + " " + venue
    if variant == "title_first_surname_venue_year":
        return title + " " + title + " " + first + " " + venue + " " + year
    if variant == "title_venue_year":
        return title + " " + title + " " + venue + " " + year
    raise ValueError(f"Unknown variant: {variant}")


def make_model(candidate: XGBCandidate) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=candidate.max_features,
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=5,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    max_depth=candidate.max_depth,
                    learning_rate=candidate.learning_rate,
                    n_estimators=candidate.n_estimators,
                    subsample=candidate.subsample,
                    colsample_bytree=candidate.colsample_bytree,
                    reg_lambda=candidate.reg_lambda,
                    reg_alpha=candidate.reg_alpha,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def candidates() -> list[XGBCandidate]:
    return [
        XGBCandidate("xgb_title_year_d2_5k", "title_year", max_features=5000, max_depth=2),
        XGBCandidate("xgb_title_year_d3_10k", "title_year", max_features=10000, max_depth=3, reg_lambda=8.0),
        XGBCandidate("xgb_title_x2_year_d2_10k", "title_x2_year", max_features=10000, max_depth=2),
        XGBCandidate("xgb_title_first_surname_d2_10k", "title_first_surname", max_features=10000, max_depth=2),
        XGBCandidate("xgb_title_first_surname_year_d2_10k", "title_first_surname_year", max_features=10000, max_depth=2),
        XGBCandidate("xgb_title_first_surname_venue_d2_10k", "title_first_surname_venue", max_features=10000, max_depth=2),
        XGBCandidate("xgb_title_first_surname_venue_year_d2_10k", "title_first_surname_venue_year", max_features=10000, max_depth=2),
        XGBCandidate("xgb_title_venue_year_d2_10k", "title_venue_year", max_features=10000, max_depth=2),
        XGBCandidate("xgb_title_year_d2_10k_lr03", "title_year", max_features=10000, max_depth=2, learning_rate=0.03, n_estimators=550),
        XGBCandidate("xgb_title_first_surname_year_d2_10k_lr03", "title_first_surname_year", max_features=10000, max_depth=2, learning_rate=0.03, n_estimators=550),
    ]


def cross_val_predict_with_weights(
    model: Pipeline,
    x: pd.Series,
    y: np.ndarray,
    folds: int,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    predictions = np.zeros(len(y), dtype=int)

    for train_idx, valid_idx in cv.split(x, y):
        fold_model = clone(model)
        sample_weight = compute_sample_weight("balanced", y[train_idx])
        fold_model.fit(
            x.iloc[train_idx],
            y[train_idx],
            clf__sample_weight=sample_weight,
        )
        predictions[valid_idx] = fold_model.predict(x.iloc[valid_idx]).astype(int)

    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost TF-IDF candidates.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/xgboost_candidates/submissions")
    parser.add_argument("--report-dir", default="outputs/xgboost_candidates/reports")
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
    train = prepare(train).reset_index(drop=True)
    public = prepare(public).reset_index(drop=True)
    private = prepare(private).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    if not output_dir.is_absolute():
        output_dir = paths.project_root / output_dir
    if not report_dir.is_absolute():
        report_dir = paths.project_root / report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    y = train[LABEL_COLUMN].astype(int).to_numpy() - 1
    rows = []

    for candidate in candidates():
        model = make_model(candidate)
        x_train = make_text(train, candidate.variant)
        x_public = make_text(public, candidate.variant)
        x_private = make_text(private, candidate.variant)

        oof_pred = cross_val_predict_with_weights(model, x_train, y, args.cv_folds)
        cv_accuracy = accuracy_score(y, oof_pred)
        cv_qwk = cohen_kappa_score(y, oof_pred, weights="quadratic")

        final_weight = compute_sample_weight("balanced", y)
        model.fit(x_train, y, clf__sample_weight=final_weight)
        public_pred = model.predict(x_public).astype(int) + 1
        private_pred = model.predict(x_private).astype(int) + 1

        public_path, private_path, combined_path = write_submission(
            candidate.name,
            public,
            private,
            public_pred,
            private_pred,
            sample,
            output_dir,
        )

        rows.append(
            {
                "name": candidate.name,
                "variant": candidate.variant,
                "max_features": candidate.max_features,
                "max_depth": candidate.max_depth,
                "learning_rate": candidate.learning_rate,
                "n_estimators": candidate.n_estimators,
                "subsample": candidate.subsample,
                "colsample_bytree": candidate.colsample_bytree,
                "reg_lambda": candidate.reg_lambda,
                "reg_alpha": candidate.reg_alpha,
                "cv_accuracy": cv_accuracy,
                "cv_qwk": cv_qwk,
                "public_submission": public_path,
                "private_submission": private_path,
                "combined_submission": combined_path,
            }
        )
        print(f"{candidate.name}: cv_qwk={cv_qwk:.6f}, cv_accuracy={cv_accuracy:.6f}")

    report = pd.DataFrame(rows).sort_values("cv_qwk", ascending=False)
    report_path = report_dir / "xgboost_candidates_report.csv"
    report.to_csv(report_path, index=False)

    best = report.iloc[0]
    best_target = paths.project_root / "outputs" / "submissions" / "next_xgboost_best_submission.csv"
    best_target.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(best["combined_submission"]).to_csv(best_target, index=False)

    print("")
    print(f"Best XGBoost candidate: {best['name']}")
    print(f"cv_qwk={best['cv_qwk']:.6f}")
    print(f"Best copy: {best_target}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
