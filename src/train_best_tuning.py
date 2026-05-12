from __future__ import annotations

import argparse
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
class TuneCandidate:
    name: str
    c: float
    char_min: int = 3
    char_max: int = 5
    word_max_features: int = 20000
    char_max_features: int = 30000
    class_weight: str | None = "balanced"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = add_eda_columns(df)
    out["title_text"] = out["title"].fillna("").astype(str)
    return out


def make_text(df: pd.DataFrame) -> pd.Series:
    return df["title_text"] + " " + df["title_text"] + " " + df["first_author_token"]


def make_model(candidate: TuneCandidate) -> Pipeline:
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
                        class_weight=candidate.class_weight,
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )


def candidates() -> list[TuneCandidate]:
    base = [
        TuneCandidate(f"title_first_surname_c{str(c).replace('.', '_')}", c=c)
        for c in [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    ]
    extras = [
        TuneCandidate("title_first_surname_c1_5_char3_6", c=1.5, char_min=3, char_max=6),
        TuneCandidate("title_first_surname_c1_5_char4_6", c=1.5, char_min=4, char_max=6),
        TuneCandidate("title_first_surname_c1_5_caps10k20k", c=1.5, word_max_features=10000, char_max_features=20000),
        TuneCandidate("title_first_surname_c1_5_caps30k50k", c=1.5, word_max_features=30000, char_max_features=50000),
        TuneCandidate("title_first_surname_c1_2_caps30k50k", c=1.2, word_max_features=30000, char_max_features=50000),
        TuneCandidate("title_first_surname_c1_8_caps30k50k", c=1.8, word_max_features=30000, char_max_features=50000),
        TuneCandidate("title_first_surname_c1_5_unweighted", c=1.5, class_weight=None),
    ]
    return base + extras


def author_bias_report(
    train: pd.DataFrame,
    public: pd.DataFrame,
    private: pd.DataFrame,
    combined_submission: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    test = pd.concat(
        [
            public.assign(split="public"),
            private.assign(split="private"),
        ],
        ignore_index=True,
    )
    test = test.merge(combined_submission, on="id", how="left")

    train_author = (
        train.groupby("first_author_token")[LABEL_COLUMN]
        .agg(
            train_count="count",
            unique_train_labels="nunique",
            train_label_min="min",
            train_label_max="max",
            train_label_mean="mean",
            train_labels=lambda values: ",".join(map(str, sorted(values.unique()))),
        )
        .reset_index()
    )
    test_author = (
        test.groupby("first_author_token")
        .agg(
            test_count=("id", "count"),
            public_count=("split", lambda values: int((values == "public").sum())),
            private_count=("split", lambda values: int((values == "private").sum())),
            predicted_labels=(LABEL_COLUMN, lambda values: ",".join(map(str, sorted(values.unique())))),
            prediction_mean=(LABEL_COLUMN, "mean"),
        )
        .reset_index()
    )
    audit = test_author.merge(train_author, on="first_author_token", how="left")
    audit["seen_in_train"] = audit["train_count"].notna()
    audit["train_count"] = audit["train_count"].fillna(0).astype(int)
    audit["unique_train_labels"] = audit["unique_train_labels"].fillna(0).astype(int)

    audit["single_train_label_repeated_test"] = (
        (audit["train_count"] == 1) & (audit["test_count"] >= 2)
    )
    audit["single_train_label5_repeated_test"] = (
        audit["single_train_label_repeated_test"] & (audit["train_label_min"] == 5)
    )
    audit["pure_author_repeated_test"] = (
        (audit["train_count"] >= 2)
        & (audit["unique_train_labels"] == 1)
        & (audit["test_count"] >= 2)
    )
    audit["all_predictions_match_only_train_label"] = (
        (audit["unique_train_labels"] == 1)
        & (audit["predicted_labels"] == audit["train_label_min"].fillna(-1).astype(int).astype(str))
    )

    audit = audit.sort_values(
        ["single_train_label5_repeated_test", "single_train_label_repeated_test", "test_count"],
        ascending=[False, False, False],
    )
    audit_path = output_dir / "first_author_bias_audit.csv"
    audit.to_csv(audit_path, index=False)

    risky = audit[
        audit["single_train_label_repeated_test"] | audit["pure_author_repeated_test"]
    ].copy()
    risky_path = output_dir / "first_author_bias_risky_cases.csv"
    risky.to_csv(risky_path, index=False)

    summary_lines = [
        "# First-Author Bias Audit",
        "",
        "This audit checks whether `first_author_surname` can over-bias predictions.",
        "",
        f"- Test first-author groups: {len(audit)}",
        f"- Seen in train: {int(audit['seen_in_train'].sum())}",
        f"- Unseen in train: {int((~audit['seen_in_train']).sum())}",
        f"- Train count = 1 and test count >= 2: {int(audit['single_train_label_repeated_test'].sum())}",
        f"- Train count = 1 with label 5 and test count >= 2: {int(audit['single_train_label5_repeated_test'].sum())}",
        f"- Pure-label train author with train count >= 2 and test count >= 2: {int(audit['pure_author_repeated_test'].sum())}",
        f"- Pure-label train authors where all test predictions match the only train label: {int(audit['all_predictions_match_only_train_label'].sum())}",
        "",
        "## Interpretation",
        "",
        "- The model can learn author tokens, so rare-author leakage-like bias is possible.",
        "- The audit separates risky author groups from normal title-driven predictions.",
        "- If public score drops for a higher-CV author-heavy model, prefer compact/regularized author features.",
    ]
    summary_path = output_dir / "first_author_bias_audit.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return summary_path, audit_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the current best title + first-author surname model.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/best_tuning/submissions")
    parser.add_argument("--report-dir", default="outputs/best_tuning/reports")
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

    x_train = make_text(train)
    x_public = make_text(public)
    x_private = make_text(private)
    y = train[LABEL_COLUMN].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for candidate in candidates():
        model = make_model(candidate)
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
                "c": candidate.c,
                "class_weight": candidate.class_weight or "none",
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
    report_path = report_dir / "best_tuning_report.csv"
    report.to_csv(report_path, index=False)

    best = report.iloc[0]
    best_target = paths.project_root / "outputs" / "submissions" / "next_best_tuned_submission.csv"
    best_target.parent.mkdir(parents=True, exist_ok=True)
    best_submission = pd.read_csv(best["combined_submission"])
    best_submission.to_csv(best_target, index=False)

    bias_summary_path, bias_audit_path = author_bias_report(
        train,
        public,
        private,
        best_submission,
        report_dir,
    )

    print("")
    print(f"Best tuned candidate: {best['name']}")
    print(f"cv_qwk={best['cv_qwk']:.6f}")
    print(f"Best copy: {best_target}")
    print(f"Report: {report_path}")
    print(f"Bias summary: {bias_summary_path}")
    print(f"Bias audit: {bias_audit_path}")


if __name__ == "__main__":
    main()
