from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.train_randomforest_best import LABEL_COLUMN
from src.train_scholarly_meta import add_oof_target_encoding


RIDGE_ANCHOR_EXPERIMENT = "ridge_threshold_5fold_5seed_891e6ee4"


def load_anchor_scores(project_root: Path) -> tuple[np.ndarray, np.ndarray]:
    report_dir = project_root / "outputs" / "ridge_threshold" / "reports"
    oof = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_oof.csv")["oof_score"].to_numpy()
    public_scores = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_public_scores.csv")["score"].to_numpy()
    private_scores = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_private_scores.csv")["score"].to_numpy()
    return oof, np.concatenate([public_scores, private_scores])


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, seed: int = 42) -> tuple[np.ndarray, float, float]:
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
        maxiter=55,
        popsize=9,
        polish=True,
        updating="immediate",
        workers=1,
    )
    thresholds = np.sort(result.x)
    labels = scores_to_labels(scores, thresholds)
    qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
    mae = mean_absolute_error(y_true, labels)
    return thresholds, float(qwk), float(mae)


def make_model(numeric_cols: list[str], categorical_cols: list[str], model_name: str = "huber") -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    if model_name == "ridge":
        model = Ridge(alpha=5.0)
    else:
        model = HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000)
    return Pipeline([("prep", preprocessor), ("model", model)])


def evaluate_feature_set(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    name: str,
    output_dir: Path,
    model_name: str = "huber",
    weights: tuple[float, ...] = (1.0, 0.85, 0.75, 0.65, 0.5, 0.35, 0.25),
) -> list[dict[str, object]]:
    y = train_df[LABEL_COLUMN].astype(int).to_numpy()
    y_float = y.astype(float)
    anchor_oof = train_df["ridge_score"].to_numpy()
    anchor_test = test_df["ridge_score"].to_numpy()
    x_train = train_df[numeric_cols + categorical_cols]
    x_test = test_df[numeric_cols + categorical_cols]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    model = make_model(numeric_cols, categorical_cols, model_name)
    oof = np.clip(cross_val_predict(model, x_train, y_float, cv=cv), 1.0, 5.0)
    model.fit(x_train, y_float)
    test_scores = np.clip(model.predict(x_test), 1.0, 5.0)

    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for weight in weights:
        blend_oof = oof if weight == 1.0 else (1 - weight) * anchor_oof + weight * oof
        blend_test = test_scores if weight == 1.0 else (1 - weight) * anchor_test + weight * test_scores
        thresholds, qwk, mae = tune_thresholds(y, blend_oof)
        labels = scores_to_labels(blend_test, thresholds)
        submission_path = output_dir / f"{name}_{model_name}_w{weight:.2f}_submission.csv"
        pd.DataFrame({"id": sample_df["id"], LABEL_COLUMN: labels}).to_csv(submission_path, index=False)
        rows.append(
            {
                "name": name,
                "model": model_name,
                "external_weight": weight,
                "ridge_weight": 1 - weight,
                "oof_qwk": qwk,
                "mae": mae,
                "feature_count": len(numeric_cols) + len(categorical_cols),
                "numeric_features": ",".join(numeric_cols),
                "categorical_features": ",".join(categorical_cols),
                "thresholds": ",".join(f"{value:.12f}" for value in thresholds),
                "test_distribution": pd.Series(labels).value_counts().sort_index().astype(int).to_dict(),
                "submission": str(submission_path),
            }
        )
    return rows


def feature_groups() -> dict[str, tuple[list[str], list[str]]]:
    return {
        "openalex_core": (
            [
                "oa_found",
                "oa_title_match_score",
                "log_oa_cited_by_count",
                "log_oa_referenced_works_count",
                "log_oa_fwci",
                "oa_fwci_venue_year_pct",
            ],
            ["oa_source"],
        ),
        "semantic_scholar_core": (
            [
                "s2_found",
                "log_s2_reference_count",
                "log_s2_citation_count",
                "log_s2_influential_citation_count",
                "influential_ratio",
                "s2_influential_citation_count_venue_year_pct",
            ],
            ["s2_source"],
        ),
        "crossref_core": (
            [
                "crossref_found",
                "log_crossref_reference_count",
                "log_crossref_is_referenced_by_count",
                "crossref_subject_count",
                "crossref_license_count",
                "crossref_has_abstract",
            ],
            ["crossref_type"],
        ),
        "opencitations_core": (
            ["coci_found", "log_coci_citation_count"],
            [],
        ),
        "source_agreement": (
            [
                "log_best_citation_count",
                "log_citation_velocity",
                "citation_count_mean",
                "citation_source_count",
                "citation_disagreement",
                "external_found_count",
                "best_citation_count_venue_year_pct",
                "citation_velocity_venue_year_pct",
                "best_reference_count_venue_year_pct",
            ],
            [],
        ),
        "impact_percentiles": (
            [
                "best_citation_count_venue_pct",
                "best_citation_count_year_pct",
                "best_citation_count_venue_year_pct",
                "citation_velocity_venue_pct",
                "citation_velocity_year_pct",
                "citation_velocity_venue_year_pct",
                "oa_fwci_venue_pct",
                "oa_fwci_year_pct",
                "oa_fwci_venue_year_pct",
            ],
            [],
        ),
    }


def available(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank scholarly features and create filtered candidates.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--enriched-file", default="outputs/scholarly_meta/reports/scholarly_enriched_rows.csv")
    parser.add_argument("--output-dir", default="outputs/feature_selection/submissions")
    parser.add_argument("--report-dir", default="outputs/feature_selection/reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train_raw = pd.read_csv(project_root / args.data_dir / "train.csv")
    public_raw = pd.read_csv(project_root / args.data_dir / "public_test.csv").assign(source_split="public_test")
    private_raw = pd.read_csv(project_root / args.data_dir / "private_test.csv").assign(source_split="private_test")
    sample_df = pd.read_csv(project_root / args.data_dir / "Test_Submission.csv")
    enriched = pd.read_csv(project_root / args.enriched_file)
    anchor_oof, anchor_test = load_anchor_scores(project_root)

    train_df = enriched[enriched["source_split"].eq("train")].copy()
    train_df = train_df.merge(train_raw[["id", LABEL_COLUMN]], on="id", how="left")
    test_df = enriched[enriched["source_split"].isin(["public_test", "private_test"])].copy()
    test_df = pd.concat(
        [
            public_raw[["id"]].merge(test_df, on="id", how="left"),
            private_raw[["id"]].merge(test_df, on="id", how="left"),
        ],
        ignore_index=True,
    )
    train_df["ridge_score"] = anchor_oof
    test_df["ridge_score"] = anchor_test
    for col, smooth in [("venue", 20.0), ("year", 20.0), ("first_surname", 8.0)]:
        train_df[f"{col}_te"], test_df[f"{col}_te"] = add_oof_target_encoding(train_df, test_df, col, smooth)

    base_numeric = ["ridge_score", "year", "title_len", "author_count", "has_authors", "venue_te", "year_te", "first_surname_te"]
    base_categorical = ["venue"]
    base_numeric = available(base_numeric, train_df)
    base_categorical = available(base_categorical, train_df)

    report_dir = project_root / args.report_dir
    output_dir = project_root / args.output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    y = train_df[LABEL_COLUMN].astype(int).to_numpy()
    univariate_rows = []
    all_group_features: list[str] = []
    for group_numeric, _ in feature_groups().values():
        all_group_features.extend(group_numeric)
    for col in sorted(set(available(all_group_features, train_df))):
        values = pd.to_numeric(train_df[col], errors="coerce")
        valid = values.notna()
        if valid.sum() < 20 or values[valid].nunique() <= 1:
            continue
        rho, p_value = spearmanr(values[valid], y[valid])
        univariate_rows.append(
            {
                "feature": col,
                "coverage": float(valid.mean()),
                "unique_values": int(values[valid].nunique()),
                "spearman": float(rho) if not np.isnan(rho) else 0.0,
                "abs_spearman": float(abs(rho)) if not np.isnan(rho) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            }
        )
    univariate = pd.DataFrame(univariate_rows).sort_values(["abs_spearman", "coverage"], ascending=[False, False])
    univariate.to_csv(report_dir / "scholarly_feature_univariate_rank.csv", index=False)

    evaluation_rows = []
    # Baseline meta without external scholarly features.
    evaluation_rows.extend(
        evaluate_feature_set(
            train_df,
            test_df,
            sample_df,
            base_numeric,
            base_categorical,
            "base_no_external",
            output_dir,
            model_name="huber",
            weights=(1.0, 0.65, 0.5, 0.35, 0.25),
        )
    )

    groups = feature_groups()
    for group_name, (numeric, categorical) in groups.items():
        numeric_cols = base_numeric + available(numeric, train_df)
        categorical_cols = base_categorical + available(categorical, train_df)
        evaluation_rows.extend(
            evaluate_feature_set(
                train_df,
                test_df,
                sample_df,
                numeric_cols,
                categorical_cols,
                group_name,
                output_dir,
                model_name="huber",
                weights=(1.0, 0.65, 0.5, 0.35, 0.25),
            )
        )

    # Filtered feature set: keep OpenAlex core plus the strongest non-OpenAlex signal family,
    # but avoid the noisy all-source dump.
    filtered_numeric = base_numeric + available(
        [
            "oa_found",
            "oa_title_match_score",
            "log_oa_cited_by_count",
            "log_oa_fwci",
            "oa_fwci_venue_year_pct",
            "log_s2_influential_citation_count",
            "s2_influential_citation_count_venue_year_pct",
            "log_citation_velocity",
            "citation_velocity_venue_year_pct",
            "citation_source_count",
        ],
        train_df,
    )
    filtered_categorical = base_categorical + available(["oa_source"], train_df)
    evaluation_rows.extend(
        evaluate_feature_set(
            train_df,
            test_df,
            sample_df,
            filtered_numeric,
            filtered_categorical,
            "filtered_low_noise",
            output_dir,
            model_name="huber",
            weights=(1.0, 0.85, 0.75, 0.65, 0.5, 0.35, 0.25),
        )
    )

    evaluations = pd.DataFrame(evaluation_rows).sort_values("oof_qwk", ascending=False)
    evaluations.to_csv(report_dir / "scholarly_feature_group_evaluation.csv", index=False)
    best = evaluations.iloc[0]
    best_copy = project_root / "outputs" / "submissions" / "next_filtered_scholarly_submission.csv"
    pd.read_csv(best["submission"]).to_csv(best_copy, index=False)

    summary = [
        "# Scholarly Feature Selection",
        "",
        "Goal: identify which external scholarly features help rating prediction and remove noisy feature families.",
        "",
        "## Best Group Evaluations",
        "",
        evaluations.head(15).to_markdown(index=False),
        "",
        "## Top Univariate Numeric Features",
        "",
        univariate.head(20).to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        "- Keep features that survive group-level OOF checks, especially OpenAlex core and source-normalized impact percentiles.",
        "- Treat raw all-source citation counts as noisy unless they improve OOF in a filtered group.",
        "- Use `outputs/submissions/next_filtered_scholarly_submission.csv` only if its OOF is competitive with the current `next_private_safe_blend_submission.csv`.",
    ]
    (report_dir / "scholarly_feature_selection.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(evaluations.head(15).to_string(index=False))
    print(f"best copy: {best_copy}")


if __name__ == "__main__":
    main()
