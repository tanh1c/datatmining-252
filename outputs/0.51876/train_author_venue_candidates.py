from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.eda_analysis import add_eda_columns
from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths
from src.train_tfidf_linear_baselines import write_submission


RANDOM_STATE = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    variant: str
    c: float = 3.0
    char_min: int = 3
    char_max: int = 5
    word_max_features: int = 20000
    char_max_features: int = 30000


def normalize_words(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def first_author_full(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "first_author_missing"
    first = str(value).split(",")[0]
    normalized = normalize_words(first)
    if not normalized:
        return "first_author_missing"
    return "first_author_full_" + normalized.replace(" ", "_")


def full_authors_text(value: object) -> str:
    normalized = normalize_words(value)
    if not normalized:
        return "authors_missing"
    return "authors_full " + normalized


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = add_eda_columns(df)
    out["title_text"] = out["title"].fillna("").astype(str)
    out["full_authors_text"] = out["authors"].map(full_authors_text)
    out["first_author_full_token"] = out["authors"].map(first_author_full)
    return out


def make_text(df: pd.DataFrame, variant: str) -> pd.Series:
    title = df["title_text"]
    first_surname = df["first_author_token"]
    first_full = df["first_author_full_token"]
    authors_full = df["full_authors_text"]
    venue = df["venue_token"]
    year = df["year_token"]

    if variant == "title_first_surname":
        return title + " " + title + " " + first_surname
    if variant == "title_first_surname_venue":
        return title + " " + title + " " + first_surname + " " + venue
    if variant == "title_first_surname_venue_year":
        return title + " " + title + " " + first_surname + " " + venue + " " + year
    if variant == "title_first_full":
        return title + " " + title + " " + first_full
    if variant == "title_first_full_venue":
        return title + " " + title + " " + first_full + " " + venue
    if variant == "title_full_authors":
        return title + " " + title + " " + authors_full
    if variant == "title_full_authors_venue":
        return title + " " + title + " " + authors_full + " " + venue
    if variant == "title_full_authors_venue_year":
        return title + " " + title + " " + authors_full + " " + venue + " " + year
    if variant == "title_venue":
        return title + " " + title + " " + venue
    if variant == "title_venue_year":
        return title + " " + title + " " + venue + " " + year
    if variant == "author_venue_only":
        return authors_full + " " + first_surname + " " + venue
    raise ValueError(f"Unknown variant: {variant}")


def make_model(candidate: Candidate) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                max_features=candidate.word_max_features,
                                min_df=1,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(candidate.char_min, candidate.char_max),
                                max_features=candidate.char_max_features,
                                min_df=1,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=candidate.c,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )


def candidates() -> list[Candidate]:
    return [
        Candidate("anchor_title_first_surname_c3", "title_first_surname", c=3.0),
        Candidate("title_first_surname_venue_c3", "title_first_surname_venue", c=3.0),
        Candidate("title_first_surname_venue_year_c3", "title_first_surname_venue_year", c=3.0),
        Candidate("title_first_full_c3", "title_first_full", c=3.0),
        Candidate("title_first_full_venue_c3", "title_first_full_venue", c=3.0),
        Candidate("title_full_authors_c3", "title_full_authors", c=3.0),
        Candidate("title_full_authors_venue_c3", "title_full_authors_venue", c=3.0),
        Candidate("title_full_authors_venue_year_c3", "title_full_authors_venue_year", c=3.0),
        Candidate("title_venue_c3", "title_venue", c=3.0),
        Candidate("title_venue_year_c3", "title_venue_year", c=3.0),
        Candidate("author_venue_only_c3", "author_venue_only", c=3.0),
        Candidate("title_first_surname_c1_5", "title_first_surname", c=1.5),
        Candidate("title_first_surname_c6", "title_first_surname", c=6.0),
        Candidate("title_first_surname_char4_6_c3", "title_first_surname", c=3.0, char_min=4, char_max=6),
        Candidate("title_full_authors_venue_c1_5", "title_full_authors_venue", c=1.5),
        Candidate("title_full_authors_venue_c6", "title_full_authors_venue", c=6.0),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train author and venue focused candidates.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/author_venue_candidates/submissions")
    parser.add_argument("--report-dir", default="outputs/author_venue_candidates/reports")
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
    report_dir.mkdir(parents=True, exist_ok=True)

    y = train[LABEL_COLUMN].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for candidate in candidates():
        model = make_model(candidate)
        x_train = make_text(train, candidate.variant)
        x_public = make_text(public, candidate.variant)
        x_private = make_text(private, candidate.variant)

        oof_pred = cross_val_predict(model, x_train, y, cv=cv, n_jobs=None)
        cv_accuracy = accuracy_score(y, oof_pred)
        cv_qwk = cohen_kappa_score(y, oof_pred, weights="quadratic")

        model.fit(x_train, y)
        public_pred = model.predict(x_public).astype(int)
        private_pred = model.predict(x_private).astype(int)
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
                "c": candidate.c,
                "char_ngram": f"{candidate.char_min}-{candidate.char_max}",
                "word_max_features": candidate.word_max_features,
                "char_max_features": candidate.char_max_features,
                "cv_accuracy": cv_accuracy,
                "cv_qwk": cv_qwk,
                "public_submission": public_path,
                "private_submission": private_path,
                "combined_submission": combined_path,
            }
        )
        print(f"{candidate.name}: cv_qwk={cv_qwk:.6f}, cv_accuracy={cv_accuracy:.6f}")

    report = pd.DataFrame(rows).sort_values("cv_qwk", ascending=False)
    report_path = report_dir / "author_venue_candidates_report.csv"
    report.to_csv(report_path, index=False)

    best = report.iloc[0]
    best_target = paths.project_root / "outputs" / "submissions" / "next_author_venue_best_submission.csv"
    best_target.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(best["combined_submission"]).to_csv(best_target, index=False)

    print("")
    print(f"Best candidate: {best['name']}")
    print(f"cv_qwk={best['cv_qwk']:.6f}")
    print(f"Best copy: {best_target}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
