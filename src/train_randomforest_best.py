from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
LABEL_COLUMN = "Label"


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    output_dir: Path
    report_dir: Path
    train_file: Path
    public_test_file: Path
    private_test_file: Path
    sample_file: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = project_root()
    data_dir = (root / args.data_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    report_dir = (root / args.report_dir).resolve()

    return Paths(
        project_root=root,
        data_dir=data_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        train_file=(data_dir / args.train_file).resolve(),
        public_test_file=(data_dir / args.public_test_file).resolve(),
        private_test_file=(data_dir / args.private_test_file).resolve(),
        sample_file=(data_dir / args.sample_file).resolve(),
    )


def require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing columns: {missing}")


def load_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(paths.train_file)
    public_df = pd.read_csv(paths.public_test_file)
    private_df = pd.read_csv(paths.private_test_file)
    sample_df = pd.read_csv(paths.sample_file)

    feature_columns = ["id", "title", "venue", "year", "authors", "doi"]
    require_columns(train_df, feature_columns + [LABEL_COLUMN], paths.train_file.name)
    require_columns(public_df, feature_columns, paths.public_test_file.name)
    require_columns(private_df, feature_columns, paths.private_test_file.name)
    require_columns(sample_df, ["id", LABEL_COLUMN], paths.sample_file.name)

    return train_df, public_df, private_df, sample_df


def clean_token(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def make_text(df: pd.DataFrame) -> pd.Series:
    data = df.copy()

    title = data["title"].fillna("").astype(str)
    authors = data["authors"].fillna("").astype(str)
    venue = clean_token(data["venue"])
    year = (
        pd.to_numeric(data["year"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )

    venue_token = "venue_" + venue
    year_token = "year_" + year

    return (
        title
        + " "
        + title
        + " "
        + venue_token
        + " "
        + venue_token
        + " "
        + authors
        + " "
        + year_token
    )


def make_model(n_estimators: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_cv(
    x_train: pd.Series,
    y_train: np.ndarray,
    n_estimators: int,
    folds: int,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    predictions = cross_val_predict(
        make_model(n_estimators),
        x_train,
        y_train,
        cv=cv,
        n_jobs=None,
    )
    return {
        "cv_accuracy": accuracy_score(y_train, predictions),
        "cv_qwk": cohen_kappa_score(y_train, predictions, weights="quadratic"),
    }


def reorder_like_sample(submission: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    sample_ids = sample_df["id"].tolist()
    submission_ids = submission["id"].tolist()

    if set(sample_ids) != set(submission_ids):
        missing = sorted(set(sample_ids) - set(submission_ids))
        extra = sorted(set(submission_ids) - set(sample_ids))
        raise ValueError(
            "Submission ids do not match sample ids. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )

    return sample_df[["id"]].merge(submission, on="id", how="left")


def write_submission(
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    public_pred: np.ndarray,
    private_pred: np.ndarray,
    sample_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    public_submission = pd.DataFrame({"id": public_df["id"], LABEL_COLUMN: public_pred})
    private_submission = pd.DataFrame({"id": private_df["id"], LABEL_COLUMN: private_pred})
    combined_submission = pd.concat(
        [public_submission, private_submission],
        ignore_index=True,
    )
    combined_submission = reorder_like_sample(combined_submission, sample_df)

    paths = {
        "public": output_dir / "randomforest_public_submission.csv",
        "private": output_dir / "randomforest_private_submission.csv",
        "combined": output_dir / "randomforest_submission.csv",
    }

    public_submission.to_csv(paths["public"], index=False)
    private_submission.to_csv(paths["private"], index=False)
    combined_submission.to_csv(paths["combined"], index=False)
    return paths


def write_report(report_dir: Path, lines: list[str]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "randomforest_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Stage 2 RandomForest baseline and create submissions."
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/submissions")
    parser.add_argument("--report-dir", default="outputs/reports")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train_df, public_df, private_df, sample_df = load_data(paths)

    x_train = make_text(train_df)
    y_train = train_df[LABEL_COLUMN].astype(int).to_numpy()
    x_public = make_text(public_df)
    x_private = make_text(private_df)

    metrics: dict[str, float] = {}
    if not args.skip_cv:
        metrics = evaluate_cv(x_train, y_train, args.n_estimators, args.cv_folds)

    model = make_model(args.n_estimators)
    model.fit(x_train, y_train)
    public_pred = model.predict(x_public).astype(int)
    private_pred = model.predict(x_private).astype(int)

    submission_paths = write_submission(
        public_df,
        private_df,
        public_pred,
        private_pred,
        sample_df,
        paths.output_dir,
    )

    lines = [
        "RandomForest Stage 2 baseline",
        f"train_rows={len(train_df)}",
        f"public_rows={len(public_df)}",
        f"private_rows={len(private_df)}",
        "features=title_x2 + venue_x2 + authors + year",
        "vectorizer=tfidf_word_1_2_max5000",
        f"n_estimators={args.n_estimators}",
    ]
    if metrics:
        lines.extend(
            [
                f"cv_folds={args.cv_folds}",
                f"cv_accuracy={metrics['cv_accuracy']:.6f}",
                f"cv_qwk={metrics['cv_qwk']:.6f}",
            ]
        )
    else:
        lines.append("cv=skipped")

    lines.extend(
        [
            f"public_submission={submission_paths['public']}",
            f"private_submission={submission_paths['private']}",
            f"combined_submission={submission_paths['combined']}",
        ]
    )
    report_path = write_report(paths.report_dir, lines)
    lines.append(f"report={report_path}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
