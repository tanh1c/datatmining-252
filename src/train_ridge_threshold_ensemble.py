from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths
from src.train_tfidf_linear_baselines import write_submission


RANDOM_STATE = 42


@dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 8.0
    title_word_max_features: int = 8000
    title_char_max_features: int = 10000
    authors_word_max_features: int = 4000
    min_df: int = 2
    title_word_ngram: tuple[int, int] = (1, 2)
    title_char_ngram: tuple[int, int] = (3, 5)
    authors_word_ngram: tuple[int, int] = (1, 2)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title_clean"] = out["title"].map(clean_text)
    out["authors_clean"] = out["authors"].map(clean_text)
    return out


def make_vectorizers(config: RidgeConfig) -> tuple[TfidfVectorizer, TfidfVectorizer, TfidfVectorizer]:
    return (
        TfidfVectorizer(
            analyzer="word",
            ngram_range=config.title_word_ngram,
            min_df=config.min_df,
            max_features=config.title_word_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=config.title_char_ngram,
            min_df=config.min_df,
            max_features=config.title_char_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        TfidfVectorizer(
            analyzer="word",
            ngram_range=config.authors_word_ngram,
            min_df=config.min_df,
            max_features=config.authors_word_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
    )


def fit_transform_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    config: RidgeConfig,
):
    title_word, title_char, authors_word = make_vectorizers(config)

    x_train = hstack(
        [
            title_word.fit_transform(train_df["title_clean"]),
            title_char.fit_transform(train_df["title_clean"]),
            authors_word.fit_transform(train_df["authors_clean"]),
        ],
        format="csr",
    )
    x_valid = hstack(
        [
            title_word.transform(valid_df["title_clean"]),
            title_char.transform(valid_df["title_clean"]),
            authors_word.transform(valid_df["authors_clean"]),
        ],
        format="csr",
    )
    x_public = hstack(
        [
            title_word.transform(public_df["title_clean"]),
            title_char.transform(public_df["title_clean"]),
            authors_word.transform(public_df["authors_clean"]),
        ],
        format="csr",
    )
    x_private = hstack(
        [
            title_word.transform(private_df["title_clean"]),
            title_char.transform(private_df["title_clean"]),
            authors_word.transform(private_df["authors_clean"]),
        ],
        format="csr",
    )
    return x_train, x_valid, x_public, x_private


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.sort(np.asarray(thresholds, dtype=float))
    return np.digitize(scores, thresholds) + 1


def tune_thresholds(y_true: np.ndarray, oof_scores: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    def objective(raw_thresholds: np.ndarray) -> float:
        thresholds = np.sort(raw_thresholds)
        min_gap = np.min(np.diff(thresholds))
        penalty = 0.0 if min_gap >= 0.03 else (0.03 - min_gap) * 5.0
        labels = scores_to_labels(oof_scores, thresholds)
        return -cohen_kappa_score(y_true, labels, weights="quadratic") + penalty

    result = differential_evolution(
        objective,
        bounds=[(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed,
        maxiter=90,
        popsize=12,
        polish=True,
        updating="immediate",
        workers=1,
    )
    thresholds = np.sort(result.x)
    qwk = cohen_kappa_score(y_true, scores_to_labels(oof_scores, thresholds), weights="quadratic")
    return thresholds, qwk


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def experiment_id(args: argparse.Namespace, config: RidgeConfig) -> str:
    raw = (
        f"ridge_threshold|folds={args.folds}|seeds={','.join(map(str, args.seeds))}|"
        f"alpha={config.alpha}|min_df={config.min_df}|"
        f"tw={config.title_word_max_features}|tc={config.title_char_max_features}|"
        f"aw={config.authors_word_max_features}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def run_repeated_cv(
    train_df: pd.DataFrame,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    config: RidgeConfig,
    folds: int,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    y = train_df[LABEL_COLUMN].astype(float).to_numpy()
    y_class = train_df[LABEL_COLUMN].astype(int).to_numpy()
    oof_sum = np.zeros(len(train_df), dtype=float)
    oof_count = np.zeros(len(train_df), dtype=float)
    public_sum = np.zeros(len(public_df), dtype=float)
    private_sum = np.zeros(len(private_df), dtype=float)
    rows = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(train_df, y_class), start=1):
            fold_train = train_df.iloc[train_idx]
            fold_valid = train_df.iloc[valid_idx]
            x_train, x_valid, x_public, x_private = fit_transform_features(
                fold_train,
                fold_valid,
                public_df,
                private_df,
                config,
            )
            model = Ridge(alpha=config.alpha, solver="lsqr")
            model.fit(x_train, y[train_idx])

            valid_scores = np.clip(model.predict(x_valid), 1.0, 5.0)
            public_scores = np.clip(model.predict(x_public), 1.0, 5.0)
            private_scores = np.clip(model.predict(x_private), 1.0, 5.0)

            oof_sum[valid_idx] += valid_scores
            oof_count[valid_idx] += 1.0
            public_sum += public_scores
            private_sum += private_scores

            rows.append(
                {
                    "seed": seed,
                    "fold": fold_idx,
                    "valid_rows": len(valid_idx),
                    "valid_score_mean": float(np.mean(valid_scores)),
                    "valid_score_std": float(np.std(valid_scores)),
                }
            )
            print(f"seed={seed} fold={fold_idx}: valid_mean={np.mean(valid_scores):.4f}")

    oof_scores = oof_sum / oof_count
    public_scores = public_sum / (len(seeds) * folds)
    private_scores = private_sum / (len(seeds) * folds)
    return oof_scores, public_scores, private_scores, pd.DataFrame(rows)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated-CV Ridge score ensemble with QWK threshold tuning.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/ridge_threshold/submissions")
    parser.add_argument("--report-dir", default="outputs/ridge_threshold/reports")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("252,253,254,255,256"))
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--threshold-seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train_df, public_df, private_df, sample_df = load_data(paths)
    train_df = prepare(train_df).reset_index(drop=True)
    public_df = prepare(public_df).reset_index(drop=True)
    private_df = prepare(private_df).reset_index(drop=True)
    config = RidgeConfig(alpha=args.alpha)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    if not output_dir.is_absolute():
        output_dir = paths.project_root / output_dir
    if not report_dir.is_absolute():
        report_dir = paths.project_root / report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    exp_hash = experiment_id(args, config)
    exp_name = f"ridge_threshold_{args.folds}fold_{len(args.seeds)}seed_{exp_hash}"

    oof_scores, public_scores, private_scores, fold_report = run_repeated_cv(
        train_df,
        public_df,
        private_df,
        config,
        args.folds,
        args.seeds,
    )

    y_true = train_df[LABEL_COLUMN].astype(int).to_numpy()
    thresholds, cv_qwk = tune_thresholds(y_true, oof_scores, args.threshold_seed)
    oof_pred = scores_to_labels(oof_scores, thresholds)
    public_pred = scores_to_labels(public_scores, thresholds)
    private_pred = scores_to_labels(private_scores, thresholds)

    public_path, private_path, combined_path = write_submission(
        exp_name,
        public_df,
        private_df,
        public_pred,
        private_pred,
        sample_df,
        output_dir,
    )

    best_target = paths.project_root / "outputs" / "submissions" / "next_ridge_threshold_submission.csv"
    best_target.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(combined_path).to_csv(best_target, index=False)

    metrics = {
        "experiment": exp_name,
        "folds": args.folds,
        "seeds": ",".join(map(str, args.seeds)),
        "alpha": config.alpha,
        "thresholds": ",".join(f"{value:.12f}" for value in thresholds),
        "oof_qwk": cv_qwk,
        "mae": mean_absolute_error(y_true, oof_pred),
        "adjacent_accuracy": adjacent_accuracy(y_true, oof_pred),
        "macro_f1": f1_score(y_true, oof_pred, average="macro"),
        "label_distribution": dict(pd.Series(pd.concat([
            pd.Series(public_pred),
            pd.Series(private_pred),
        ])).value_counts().sort_index()),
        "public_submission": str(public_path),
        "private_submission": str(private_path),
        "combined_submission": str(combined_path),
        "best_copy": str(best_target),
    }
    report_path = report_dir / f"{exp_name}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(report_path, index=False)
    fold_report.to_csv(report_dir / f"{exp_name}_folds.csv", index=False)
    pd.DataFrame(
        {
            "id": train_df["id"],
            "Label": y_true,
            "oof_score": oof_scores,
            "oof_pred": oof_pred,
        }
    ).to_csv(report_dir / f"{exp_name}_oof.csv", index=False)
    pd.DataFrame({"id": public_df["id"], "score": public_scores}).to_csv(
        report_dir / f"{exp_name}_public_scores.csv",
        index=False,
    )
    pd.DataFrame({"id": private_df["id"], "score": private_scores}).to_csv(
        report_dir / f"{exp_name}_private_scores.csv",
        index=False,
    )

    print("")
    print(f"Experiment: {exp_name}")
    print(f"OOF QWK={cv_qwk:.6f}")
    print(f"MAE={metrics['mae']:.6f}")
    print(f"AdjAcc={metrics['adjacent_accuracy']:.6f}")
    print(f"MacroF1={metrics['macro_f1']:.6f}")
    print(f"Thresholds={thresholds.tolist()}")
    print(f"Submission={combined_path}")
    print(f"Best copy={best_target}")
    print(f"Report={report_path}")


if __name__ == "__main__":
    main()
