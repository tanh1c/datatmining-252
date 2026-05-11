from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths


RANDOM_STATE = 42


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def clean_token_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def first_author(authors: object) -> str:
    if pd.isna(authors) or not str(authors).strip():
        return "missing_author"
    first = str(authors).split(",")[0].strip().lower()
    parts = re.findall(r"[a-z0-9]+", first)
    if not parts:
        return "missing_author"
    return "first_author_" + parts[-1]


def doi_kind(doi: object) -> str:
    if pd.isna(doi) or not str(doi).strip():
        return "doi_missing"
    text = str(doi).lower()
    if "semanticscholar.org" in text:
        return "doi_semantic_scholar"
    if text.startswith("10."):
        return "doi_formal"
    if text.startswith("http"):
        return "doi_other_url"
    return "doi_other"


def add_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title_norm"] = out["title"].map(normalize_text)
    out["title_words"] = out["title"].fillna("").astype(str).str.split().str.len()
    out["title_chars"] = out["title"].fillna("").astype(str).str.len()
    out["authors_filled"] = out["authors"].fillna("").astype(str)
    out["author_count"] = out["authors_filled"].map(
        lambda text: 0 if not text.strip() else len([x for x in text.split(",") if x.strip()])
    )
    out["first_author_token"] = out["authors"].map(first_author)
    out["venue_token"] = "venue_" + clean_token_series(out["venue"])
    out["year_token"] = "year_" + pd.to_numeric(
        out["year"], errors="coerce"
    ).fillna(0).astype(int).astype(str)
    out["doi_kind"] = out["doi"].map(doi_kind)
    return out


def label_distribution(train: pd.DataFrame) -> pd.DataFrame:
    counts = train[LABEL_COLUMN].value_counts().sort_index()
    return pd.DataFrame(
        {
            LABEL_COLUMN: counts.index,
            "count": counts.values,
            "percent": (counts.values / len(train) * 100).round(2),
        }
    )


def missingness(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["title", "venue", "year", "authors", "doi", LABEL_COLUMN]:
        missing = int(train[column].isna().sum())
        blank = int((train[column].fillna("").astype(str).str.strip() == "").sum())
        rows.append(
            {
                "column": column,
                "missing": missing,
                "blank_or_missing": blank,
                "blank_or_missing_percent": round(blank / len(train) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def venue_summary(train: pd.DataFrame) -> pd.DataFrame:
    return (
        train.groupby("venue", dropna=False)[LABEL_COLUMN]
        .agg(["count", "mean", "median", "std"])
        .sort_values(["count", "mean"], ascending=[False, True])
        .reset_index()
        .round({"mean": 3, "median": 3, "std": 3})
    )


def year_summary(train: pd.DataFrame) -> pd.DataFrame:
    return (
        train.groupby("year", dropna=False)[LABEL_COLUMN]
        .agg(["count", "mean", "median", "std"])
        .sort_values("year")
        .reset_index()
        .round({"mean": 3, "median": 3, "std": 3})
    )


def duplicate_title_summary(train: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        train.groupby("title_norm")[LABEL_COLUMN]
        .agg(count="count", unique_labels="nunique", labels=lambda x: ",".join(map(str, sorted(x.unique()))))
        .reset_index()
    )
    return grouped[grouped["count"] > 1].sort_values(
        ["unique_labels", "count"], ascending=[False, False]
    )


def overlap_summary(train: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    train_titles = set(train["title_norm"])
    train_venues = set(train["venue"].fillna("").astype(str))
    train_years = set(train["year"].dropna().astype(int))
    train_first_authors = set(train["first_author_token"])

    rows = []
    for name, df in [("public", public), ("private", private)]:
        rows.append(
            {
                "split": name,
                "rows": len(df),
                "title_exact_matches_train": int(df["title_norm"].isin(train_titles).sum()),
                "venue_unseen_count": int((~df["venue"].fillna("").astype(str).isin(train_venues)).sum()),
                "year_unseen_count": int((~df["year"].dropna().astype(int).isin(train_years)).sum()),
                "first_author_matches_train": int(df["first_author_token"].isin(train_first_authors).sum()),
            }
        )
    return pd.DataFrame(rows)


def text_variant(df: pd.DataFrame, variant: str) -> pd.Series:
    title = df["title"].fillna("").astype(str)
    venue = df["venue_token"]
    year = df["year_token"]
    authors = df["authors"].fillna("").astype(str)
    first = df["first_author_token"]
    doi = df["doi_kind"]

    if variant == "title_only":
        return title
    if variant == "title_x2":
        return title + " " + title
    if variant == "title_first_author":
        return title + " " + title + " " + first
    if variant == "title_venue_year":
        return title + " " + title + " " + venue + " " + venue + " " + year
    if variant == "title_venue_year_authors":
        return title + " " + title + " " + venue + " " + venue + " " + year + " " + authors
    if variant == "title_venue_year_authors_doi_kind":
        return title + " " + title + " " + venue + " " + venue + " " + year + " " + authors + " " + doi
    if variant == "metadata_only":
        return venue + " " + venue + " " + year + " " + first + " " + doi
    raise ValueError(f"Unknown variant: {variant}")


def make_logreg_pipeline(word_char: bool) -> Pipeline:
    clf = OneVsRestClassifier(
        LogisticRegression(
            C=3.0 if word_char else 4.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
    )
    if not word_char:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=20000,
                        min_df=1,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                ("clf", clf),
            ]
        )

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
                ),
            ),
            ("clf", clf),
        ]
    )


def run_ablation(train: pd.DataFrame, folds: int) -> pd.DataFrame:
    variants = [
        "title_only",
        "title_x2",
        "title_first_author",
        "title_venue_year",
        "title_venue_year_authors",
        "title_venue_year_authors_doi_kind",
        "metadata_only",
    ]
    y = train[LABEL_COLUMN].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for variant in variants:
        x = text_variant(train, variant)
        for word_char in [False, True]:
            if variant == "metadata_only" and word_char:
                continue
            model = make_logreg_pipeline(word_char=word_char)
            pred = cross_val_predict(model, x, y, cv=cv, n_jobs=None)
            rows.append(
                {
                    "variant": variant,
                    "vectorizer": "word_char" if word_char else "word",
                    "cv_accuracy": accuracy_score(y, pred),
                    "cv_qwk": cohen_kappa_score(y, pred, weights="quadratic"),
                }
            )

    return pd.DataFrame(rows).sort_values("cv_qwk", ascending=False)


def write_report(
    out_dir: Path,
    train: pd.DataFrame,
    public: pd.DataFrame,
    private: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> Path:
    label_df = tables["label_distribution"]
    missing_df = tables["missingness"]
    venue_df = tables["venue_summary"]
    duplicate_df = tables["duplicate_titles"]
    overlap_df = tables["overlap_summary"]
    ablation_df = tables["feature_ablation_cv"]

    lines = [
        "# EDA Report",
        "",
        "## Dataset Shape",
        "",
        f"- Train rows: {len(train)}",
        f"- Public test rows: {len(public)}",
        f"- Private test rows: {len(private)}",
        f"- Label range: {sorted(train[LABEL_COLUMN].unique().tolist())}",
        "",
        "## Label Distribution",
        "",
        label_df.to_markdown(index=False),
        "",
        "## Missingness",
        "",
        missing_df.to_markdown(index=False),
        "",
        "## Strong Data Signals",
        "",
        f"- Venues are limited: {train['venue'].nunique()} unique train venues.",
        f"- Most common venue: {venue_df.iloc[0]['venue']} ({int(venue_df.iloc[0]['count'])} rows).",
        f"- Duplicate normalized titles in train: {len(duplicate_df)} groups.",
        f"- Duplicate title groups with conflicting labels: {int((duplicate_df['unique_labels'] > 1).sum())}.",
        "",
        "## Train/Test Coverage",
        "",
        overlap_df.to_markdown(index=False),
        "",
        "## Feature Ablation",
        "",
        ablation_df.head(12).to_markdown(index=False),
        "",
        "## Actionable Takeaways",
        "",
        "1. Logistic TF-IDF remains the right search area; it beats the RandomForest baseline in CV and public submissions.",
        "2. Title text is the core signal. Metadata-only is weaker, so metadata should be added as small tokens rather than dominating the text.",
        "3. Venue/year tokens help but can overfit; keep them prefixed and test against public/private drift.",
        "4. Author information is useful mainly as a compact cue. Try first-author surname rather than the full author list.",
        "5. `doi` should not be used as raw text. If used, keep only coarse `doi_kind` tokens because DOI strings are noisy identifiers.",
        "6. Since public leaderboard is only half of test data, favor simple Logistic variants with stable CV over aggressive blends.",
        "",
    ]

    report_path = out_dir / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for the Stage 2 paper classification data.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/eda")
    parser.add_argument("--report-dir", default="outputs/eda")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train, public, private, _ = load_data(paths)
    train = add_eda_columns(train)
    public = add_eda_columns(public)
    private = add_eda_columns(private)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = paths.project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "label_distribution": label_distribution(train),
        "missingness": missingness(train),
        "venue_summary": venue_summary(train),
        "year_summary": year_summary(train),
        "duplicate_titles": duplicate_title_summary(train),
        "overlap_summary": overlap_summary(train, public, private),
        "feature_ablation_cv": run_ablation(train, args.cv_folds),
    }

    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    report_path = write_report(out_dir, train, public, private, tables)
    print(f"EDA report: {report_path}")
    print(f"Best ablation: {tables['feature_ablation_cv'].iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
