from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths


RANDOM_STATE = 42


@dataclass(frozen=True)
class EmbeddingCandidate:
    name: str
    text_variant: str
    model_name: str
    alpha: float = 8.0


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def make_split_frame(train_df: pd.DataFrame, public_df: pd.DataFrame, private_df: pd.DataFrame) -> pd.DataFrame:
    train = train_df.drop(columns=[LABEL_COLUMN]).copy()
    train["source_split"] = "train"
    public = public_df.copy()
    public["source_split"] = "public_test"
    private = private_df.copy()
    private["source_split"] = "private_test"
    return pd.concat([train, public, private], ignore_index=True)


def attach_abstracts(all_rows: pd.DataFrame, abstracts_path: Path) -> pd.DataFrame:
    out = all_rows.copy()
    if not abstracts_path.exists():
        out["abstract"] = ""
        out["has_abstract"] = 0
        out["abstract_source"] = ""
        return out

    abstracts = pd.read_csv(abstracts_path)
    keep = ["source_split", "id", "abstract", "has_abstract", "abstract_source"]
    abstracts = abstracts[[column for column in keep if column in abstracts.columns]].copy()
    out = out.merge(abstracts, on=["source_split", "id"], how="left")
    out["abstract"] = out["abstract"].fillna("")
    out["has_abstract"] = out["has_abstract"].fillna(False).astype(int)
    out["abstract_source"] = out["abstract_source"].fillna("")
    return out


def build_texts(df: pd.DataFrame, variant: str) -> list[str]:
    title = df["title"].map(clean_text)
    abstract = df["abstract"].map(clean_text).str.slice(0, 1800)
    venue = df["venue"].fillna("").astype(str)
    year = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int).astype(str)

    if variant == "title":
        return title.tolist()
    if variant == "title_abstract":
        return ("Title: " + title + "\nAbstract: " + abstract).tolist()
    if variant == "title_abstract_meta":
        return ("Venue: " + venue + "\nYear: " + year + "\nTitle: " + title + "\nAbstract: " + abstract).tolist()
    raise ValueError(f"Unknown text variant: {variant}")


def safe_model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace("\\", "__").replace(" ", "_")


def encode_texts(
    texts: list[str],
    model_name: str,
    cache_path: Path,
    batch_size: int,
    local_files_only: bool,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, seed: int = RANDOM_STATE) -> tuple[np.ndarray, float, float, dict[int, int]]:
    def objective(raw_thresholds: np.ndarray) -> float:
        thresholds = np.sort(raw_thresholds)
        gap = np.min(np.diff(thresholds))
        penalty = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(scores, thresholds)
        return -cohen_kappa_score(y_true, labels, weights="quadratic") + penalty

    result = differential_evolution(
        objective,
        bounds=[(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed,
        maxiter=70,
        popsize=10,
        polish=True,
        workers=1,
        updating="immediate",
    )
    thresholds = np.sort(result.x)
    labels = scores_to_labels(scores, thresholds)
    qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
    mae = mean_absolute_error(y_true, labels)
    distribution = pd.Series(labels).value_counts().sort_index().astype(int).to_dict()
    return thresholds, float(qwk), float(mae), distribution


def make_model(candidate: EmbeddingCandidate):
    if candidate.model_name == "huber":
        return HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000)
    if candidate.model_name == "ridge":
        return Ridge(alpha=candidate.alpha)
    raise ValueError(f"Unknown dense model: {candidate.model_name}")


def run_repeated_cv(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    candidate: EmbeddingCandidate,
    folds: int,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    oof_sum = np.zeros(len(x_train), dtype=float)
    oof_count = np.zeros(len(x_train), dtype=float)
    test_sum = np.zeros(len(x_test), dtype=float)
    rows = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train, y), start=1):
            scaler = StandardScaler()
            fold_x = scaler.fit_transform(x_train[train_idx])
            valid_x = scaler.transform(x_train[valid_idx])
            test_x = scaler.transform(x_test)
            model = make_model(candidate)
            model.fit(fold_x, y[train_idx].astype(float))
            valid_scores = np.clip(model.predict(valid_x), 1.0, 5.0)
            test_scores = np.clip(model.predict(test_x), 1.0, 5.0)
            oof_sum[valid_idx] += valid_scores
            oof_count[valid_idx] += 1.0
            test_sum += test_scores
            rows.append(
                {
                    "candidate": candidate.name,
                    "seed": seed,
                    "fold": fold_idx,
                    "valid_mean": float(np.mean(valid_scores)),
                    "valid_std": float(np.std(valid_scores)),
                }
            )
            print(f"{candidate.name} seed={seed} fold={fold_idx} mean={np.mean(valid_scores):.4f}")

    oof_scores = oof_sum / oof_count
    test_scores = test_sum / (len(seeds) * folds)
    return oof_scores, test_scores, pd.DataFrame(rows)


def experiment_hash(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrained embedding candidates for topic/semantic label prediction.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/pretrained_embeddings/submissions")
    parser.add_argument("--report-dir", default="outputs/pretrained_embeddings/reports")
    parser.add_argument("--embedding-dir", default="outputs/pretrained_embeddings/cache")
    parser.add_argument("--abstracts-file", default="outputs/external/abstracts_merged.csv")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--hf-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("252,253,254,255,256"))
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train_df, public_df, private_df, sample_df = load_data(paths)
    all_rows = make_split_frame(train_df, public_df, private_df)

    abstracts_path = paths.project_root / args.abstracts_file
    all_rows = attach_abstracts(all_rows, abstracts_path)
    output_dir = paths.project_root / args.output_dir
    report_dir = paths.project_root / args.report_dir
    embedding_dir = paths.project_root / args.embedding_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    n_train = len(train_df)
    n_test = len(public_df) + len(private_df)
    y = train_df[LABEL_COLUMN].astype(int).to_numpy()

    candidates = [
        EmbeddingCandidate("bge_title_ridge_a3", "title", "ridge", alpha=3.0),
        EmbeddingCandidate("bge_title_abs_ridge_a3", "title_abstract", "ridge", alpha=3.0),
        EmbeddingCandidate("bge_title_abs_meta_ridge_a3", "title_abstract_meta", "ridge", alpha=3.0),
        EmbeddingCandidate("bge_title_abs_huber", "title_abstract", "huber"),
    ]

    rows = []
    fold_rows = []
    for candidate in candidates:
        texts = build_texts(all_rows, candidate.text_variant)
        cache_name = f"{safe_model_slug(args.hf_model)}__{candidate.text_variant}.npy"
        embeddings = encode_texts(
            texts,
            args.hf_model,
            embedding_dir / cache_name,
            batch_size=args.batch_size,
            local_files_only=not args.allow_download,
        )
        if len(embeddings) != len(all_rows):
            raise ValueError(f"Embedding row count mismatch: {len(embeddings)} != {len(all_rows)}")
        x_train = embeddings[:n_train]
        x_test = embeddings[n_train : n_train + n_test]
        oof_scores, test_scores, folds = run_repeated_cv(x_train, x_test, y, candidate, args.folds, args.seeds)
        fold_rows.append(folds)
        thresholds, qwk, mae, train_distribution = tune_thresholds(y, oof_scores)
        test_labels = scores_to_labels(test_scores, thresholds)
        submission = pd.DataFrame({"id": sample_df["id"], LABEL_COLUMN: test_labels})
        exp_id = experiment_hash([candidate.name, args.hf_model, ",".join(map(str, args.seeds)), str(args.folds)])
        submission_path = output_dir / f"{candidate.name}_{exp_id}_submission.csv"
        submission.to_csv(submission_path, index=False)
        np.save(report_dir / f"{candidate.name}_{exp_id}_oof_scores.npy", oof_scores)
        np.save(report_dir / f"{candidate.name}_{exp_id}_test_scores.npy", test_scores)
        rows.append(
            {
                "candidate": candidate.name,
                "hf_model": args.hf_model,
                "text_variant": candidate.text_variant,
                "dense_model": candidate.model_name,
                "alpha": candidate.alpha,
                "folds": args.folds,
                "seeds": ",".join(map(str, args.seeds)),
                "oof_qwk": qwk,
                "mae": mae,
                "thresholds": ",".join(f"{value:.12f}" for value in thresholds),
                "train_distribution": train_distribution,
                "test_distribution": pd.Series(test_labels).value_counts().sort_index().astype(int).to_dict(),
                "submission": str(submission_path),
            }
        )

    report = pd.DataFrame(rows).sort_values("oof_qwk", ascending=False)
    report.to_csv(report_dir / "pretrained_embedding_candidates.csv", index=False)
    if fold_rows:
        pd.concat(fold_rows, ignore_index=True).to_csv(report_dir / "pretrained_embedding_folds.csv", index=False)

    best = report.iloc[0]
    best_copy = paths.project_root / "outputs" / "submissions" / "next_pretrained_embedding_submission.csv"
    pd.read_csv(best["submission"]).to_csv(best_copy, index=False)

    summary_lines = [
        "# Pretrained Embedding Candidates",
        "",
        f"Model: `{args.hf_model}`",
        "",
        "These candidates test whether pretrained semantic embeddings capture the topic axis behind labels better than hand-written lexicons.",
        "",
        "## Top Candidates",
        "",
        report.to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        "- Compare public LB against the current `0.63064` OpenAlex DOI-only Huber anchor.",
        "- If embeddings win, stop optimizing citation/impact features and focus on semantic text modeling.",
        "- If embeddings lose, use their OOF/test disagreements to identify topic-specific threshold fixes.",
    ]
    (report_dir / "pretrained_embedding_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(report.to_string(index=False))
    print(f"best copy: {best_copy}")


if __name__ == "__main__":
    main()
