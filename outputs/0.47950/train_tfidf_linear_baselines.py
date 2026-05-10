from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.train_randomforest_best import (
    LABEL_COLUMN,
    load_data,
    make_text,
    reorder_like_sample,
    resolve_paths,
)


RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    cv_accuracy: float
    cv_qwk: float
    public_path: Path
    private_path: Path
    combined_path: Path


def word_vectorizer(max_features: int = 20000) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def word_char_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=30000,
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "logreg_word": Pipeline(
            [
                ("tfidf", word_vectorizer()),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=4.0,
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=RANDOM_STATE,
                            solver="liblinear",
                        )
                    ),
                ),
            ]
        ),
        "ridge_word": Pipeline(
            [
                ("tfidf", word_vectorizer()),
                ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
            ]
        ),
        "linearsvc_word": Pipeline(
            [
                ("tfidf", word_vectorizer()),
                (
                    "clf",
                    LinearSVC(
                        C=0.7,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        dual="auto",
                    ),
                ),
            ]
        ),
        "sgd_log_word": Pipeline(
            [
                ("tfidf", word_vectorizer()),
                (
                    "clf",
                    SGDClassifier(
                        loss="modified_huber",
                        alpha=1e-5,
                        class_weight="balanced",
                        max_iter=2000,
                        tol=1e-4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "logreg_word_char": Pipeline(
            [
                ("tfidf", word_char_vectorizer()),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=3.0,
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=RANDOM_STATE,
                            solver="liblinear",
                        )
                    ),
                ),
            ]
        ),
        "ridge_word_char": Pipeline(
            [
                ("tfidf", word_char_vectorizer()),
                ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
            ]
        ),
        "linearsvc_word_char": Pipeline(
            [
                ("tfidf", word_char_vectorizer()),
                (
                    "clf",
                    LinearSVC(
                        C=0.5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        dual="auto",
                    ),
                ),
            ]
        ),
    }


def predict_scores(model: Pipeline, x: pd.Series) -> np.ndarray:
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
    else:
        predictions = model.predict(x)
        classes = np.asarray(model.classes_)
        scores = np.full((len(predictions), len(classes)), -1.0)
        for row, prediction in enumerate(predictions):
            scores[row, int(np.where(classes == prediction)[0][0])] = 1.0

    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return np.asarray(scores, dtype=float)


def scores_to_labels(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmax(scores, axis=1)].astype(int)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return centered / scale


def cross_val_scores(
    model: Pipeline,
    x_train: pd.Series,
    y_train: np.ndarray,
    folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.sort(np.unique(y_train))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = np.zeros((len(y_train), len(classes)), dtype=float)
    oof_pred = np.zeros(len(y_train), dtype=int)

    for train_idx, valid_idx in cv.split(x_train, y_train):
        fold_model = clone(model)
        fold_model.fit(x_train.iloc[train_idx], y_train[train_idx])
        fold_scores = predict_scores(fold_model, x_train.iloc[valid_idx])
        fold_classes = np.asarray(fold_model.classes_)

        aligned = np.zeros((len(valid_idx), len(classes)), dtype=float)
        for column, label in enumerate(fold_classes):
            target_column = int(np.where(classes == label)[0][0])
            aligned[:, target_column] = fold_scores[:, column]

        oof_scores[valid_idx] = aligned
        oof_pred[valid_idx] = scores_to_labels(aligned, classes)

    return oof_scores, oof_pred, classes


def write_submission(
    name: str,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    public_pred: np.ndarray,
    private_pred: np.ndarray,
    sample_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    public_submission = pd.DataFrame({"id": public_df["id"], LABEL_COLUMN: public_pred})
    private_submission = pd.DataFrame({"id": private_df["id"], LABEL_COLUMN: private_pred})
    combined_submission = pd.concat(
        [public_submission, private_submission],
        ignore_index=True,
    )
    combined_submission = reorder_like_sample(combined_submission, sample_df)

    public_path = output_dir / f"{name}_public_submission.csv"
    private_path = output_dir / f"{name}_private_submission.csv"
    combined_path = output_dir / f"{name}_submission.csv"
    public_submission.to_csv(public_path, index=False)
    private_submission.to_csv(private_path, index=False)
    combined_submission.to_csv(combined_path, index=False)
    return public_path, private_path, combined_path


def evaluate_and_write_model(
    name: str,
    model: Pipeline,
    x_train: pd.Series,
    y_train: np.ndarray,
    x_public: pd.Series,
    x_private: pd.Series,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    output_dir: Path,
    folds: int,
) -> tuple[ExperimentResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oof_scores, oof_pred, classes = cross_val_scores(model, x_train, y_train, folds)
    cv_accuracy = accuracy_score(y_train, oof_pred)
    cv_qwk = cohen_kappa_score(y_train, oof_pred, weights="quadratic")

    final_model = clone(model)
    final_model.fit(x_train, y_train)
    public_scores = predict_scores(final_model, x_public)
    private_scores = predict_scores(final_model, x_private)
    public_pred = scores_to_labels(public_scores, np.asarray(final_model.classes_))
    private_pred = scores_to_labels(private_scores, np.asarray(final_model.classes_))

    paths = write_submission(
        name,
        public_df,
        private_df,
        public_pred,
        private_pred,
        sample_df,
        output_dir,
    )
    result = ExperimentResult(name, cv_accuracy, cv_qwk, *paths)
    return result, oof_scores, public_scores, private_scores, classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try TF-IDF linear baselines and simple CV-selected blends."
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/linear_baselines/submissions")
    parser.add_argument("--report-dir", default="outputs/linear_baselines/reports")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train_df, public_df, private_df, sample_df = load_data(paths)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    if not output_dir.is_absolute():
        output_dir = paths.project_root / output_dir
    if not report_dir.is_absolute():
        report_dir = paths.project_root / report_dir

    x_train = make_text(train_df).reset_index(drop=True)
    y_train = train_df[LABEL_COLUMN].astype(int).to_numpy()
    x_public = make_text(public_df).reset_index(drop=True)
    x_private = make_text(private_df).reset_index(drop=True)

    models = make_models()
    results: list[ExperimentResult] = []
    score_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    print("Running linear TF-IDF experiments...")
    for name, model in models.items():
        result, oof_scores, public_scores, private_scores, classes = evaluate_and_write_model(
            name,
            model,
            x_train,
            y_train,
            x_public,
            x_private,
            public_df,
            private_df,
            sample_df,
            output_dir,
            args.cv_folds,
        )
        results.append(result)
        score_cache[name] = (oof_scores, public_scores, private_scores, classes)
        print(f"{name}: cv_qwk={result.cv_qwk:.6f}, cv_accuracy={result.cv_accuracy:.6f}")

    ranked = sorted(results, key=lambda item: item.cv_qwk, reverse=True)
    selected_names = [result.name for result in ranked[: args.top_k]]

    blend_results: list[ExperimentResult] = []
    if len(selected_names) >= 2:
        classes = score_cache[selected_names[0]][3]
        for k in range(2, len(selected_names) + 1):
            names = selected_names[:k]
            blend_name = "blend_top" + str(k)
            oof_blend = np.mean(
                [normalize_scores(score_cache[name][0]) for name in names],
                axis=0,
            )
            public_blend = np.mean(
                [normalize_scores(score_cache[name][1]) for name in names],
                axis=0,
            )
            private_blend = np.mean(
                [normalize_scores(score_cache[name][2]) for name in names],
                axis=0,
            )
            oof_pred = scores_to_labels(oof_blend, classes)
            public_pred = scores_to_labels(public_blend, classes)
            private_pred = scores_to_labels(private_blend, classes)
            paths_written = write_submission(
                blend_name,
                public_df,
                private_df,
                public_pred,
                private_pred,
                sample_df,
                output_dir,
            )
            blend_result = ExperimentResult(
                blend_name,
                accuracy_score(y_train, oof_pred),
                cohen_kappa_score(y_train, oof_pred, weights="quadratic"),
                *paths_written,
            )
            blend_results.append(blend_result)
            print(
                f"{blend_name} ({', '.join(names)}): "
                f"cv_qwk={blend_result.cv_qwk:.6f}, "
                f"cv_accuracy={blend_result.cv_accuracy:.6f}"
            )

    all_results = sorted(results + blend_results, key=lambda item: item.cv_qwk, reverse=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "linear_baselines_report.csv"
    pd.DataFrame(
        [
            {
                "rank": index + 1,
                "name": result.name,
                "cv_qwk": result.cv_qwk,
                "cv_accuracy": result.cv_accuracy,
                "combined_submission": result.combined_path,
            }
            for index, result in enumerate(all_results)
        ]
    ).to_csv(report_path, index=False)

    best = all_results[0]
    print("")
    print(f"Best by CV QWK: {best.name}")
    print(f"cv_qwk={best.cv_qwk:.6f}")
    print(f"submission={best.combined_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
