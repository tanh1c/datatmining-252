"""Build OpenAlex continuous OOF anchor — schema-compatible with SciNCL/SPECTER2/Ridge.

Outputs (under outputs/openalex_anchor/):
  - oof_scores.csv     : id, Label, oof_score, oof_pred
  - public_scores.csv  : id, score, pred
  - private_scores.csv : id, score, pred
  - metrics.json
  - openalex_anchor_submission.csv (single-anchor submission, for sanity check)

Differs from train_openalex_title_meta.py:
  - No ridge_score feature — we want a STANDALONE non-text anchor for stacking,
    not a meta blend. Stacking step combines anchors at submission time.
  - Repeated CV (5 folds x 3 seeds) so OOF rows match the encoder anchors.
  - Schema mirrors outputs/0.72103/scincl_oof_scores.csv, outputs/0.69972/{public,private}_scores.csv.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "outputs" / "external"
OUT_DIR = PROJECT_ROOT / "outputs" / "openalex_anchor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2026
SEEDS = [252, 253, 254]
FOLDS = 5
LABEL_COLUMN = "Label"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "semanticscholar.org" in text.lower():
        return ""
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    return text if text.lower().startswith("10.") else ""


def first_author_surname(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    first_author = str(value).split(",")[0]
    tokens = re.findall(r"[A-Za-zÀ-ỹÁ-ỹ'’-]+", first_author)
    return tokens[-1].lower() if tokens else ""


def add_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["doi_norm"] = out["doi"].map(normalize_doi)
    out["first_surname"] = out["authors"].map(first_author_surname)
    out["has_authors"] = out["authors"].notna().astype(int)
    out["title_len"] = out["title"].fillna("").astype(str).str.len()
    out["author_count"] = (
        out["authors"].fillna("").map(lambda v: 0 if not str(v).strip() else len(str(v).split(",")))
    )
    return out


def attach_openalex(rows: pd.DataFrame, doi_cache: pd.DataFrame, title_cache: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    doi_rename = {
        "openalex_found": "doi_openalex_found",
        "openalex_id": "doi_openalex_id",
        "openalex_title": "doi_openalex_title",
        "openalex_year": "doi_openalex_year",
        "cited_by_count": "doi_cited_by_count",
        "referenced_works_count": "doi_referenced_works_count",
        "fwci": "doi_fwci",
        "is_retracted": "doi_is_retracted",
    }
    rows = rows.merge(doi_cache.rename(columns=doi_rename), on="doi_norm", how="left")

    title_rename = {
        "title_search_found": "title_openalex_found",
        "openalex_id": "title_openalex_id",
        "openalex_title": "title_openalex_title",
        "openalex_year": "title_openalex_year",
        "cited_by_count": "title_cited_by_count",
        "referenced_works_count": "title_referenced_works_count",
        "fwci": "title_fwci",
        "is_retracted": "title_is_retracted",
    }
    if not title_cache.empty:
        rows = rows.merge(title_cache.rename(columns=title_rename), on=["source_split", "id"], how="left")
    else:
        for c in title_rename.values():
            rows[c] = np.nan
        rows["title_match_score"] = np.nan
        rows["title_year_delta"] = np.nan

    use_doi = rows["doi_openalex_found"].map(lambda v: False if pd.isna(v) else bool(v))
    title_found = rows["title_openalex_found"].map(lambda v: False if pd.isna(v) else bool(v))
    use_title = (~use_doi) & title_found
    rows["openalex_source"] = np.where(use_doi, "doi", np.where(use_title, "title", "none"))
    rows["openalex_found"] = (use_doi | use_title).astype(int)

    pairs = {
        "openalex_year": ("doi_openalex_year", "title_openalex_year"),
        "cited_by_count": ("doi_cited_by_count", "title_cited_by_count"),
        "referenced_works_count": ("doi_referenced_works_count", "title_referenced_works_count"),
        "fwci": ("doi_fwci", "title_fwci"),
        "is_retracted": ("doi_is_retracted", "title_is_retracted"),
    }
    for field, (doi_col, title_col) in pairs.items():
        rows[field] = np.where(use_doi, rows.get(doi_col), np.where(use_title, rows.get(title_col), np.nan))
    return rows


def add_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["cited_by_count", "referenced_works_count", "fwci", "openalex_year"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["title_match_score"] = pd.to_numeric(out.get("title_match_score", np.nan), errors="coerce").fillna(0.0)
    out["title_year_delta"] = pd.to_numeric(out.get("title_year_delta", np.nan), errors="coerce")
    out["citation_age"] = (CURRENT_YEAR - out["openalex_year"] + 1).clip(lower=1)
    out["citation_velocity"] = out["cited_by_count"] / out["citation_age"]

    for c in ["cited_by_count", "referenced_works_count", "fwci", "citation_velocity"]:
        out[f"log_{c}"] = np.log1p(out[c].clip(lower=0))
        out[f"{c}_global_pct"] = out[c].rank(pct=True)

    for name, keys in {"venue": ["venue"], "year": ["year"], "venue_year": ["venue", "year"]}.items():
        for c in ["cited_by_count", "citation_velocity", "fwci"]:
            out[f"{c}_{name}_pct"] = out.groupby(keys)[c].rank(pct=True)
            out[f"{c}_{name}_pct"] = out[f"{c}_{name}_pct"].fillna(out[f"{c}_global_pct"])
    return out


def oof_target_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str, smooth: float, seed: int):
    y = train_df[LABEL_COLUMN].astype(float).to_numpy()
    y_class = train_df[LABEL_COLUMN].astype(int).to_numpy()
    prior = float(np.mean(y))
    oof = np.zeros(len(train_df), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in cv.split(train_df, y_class):
        fold = pd.DataFrame({"key": train_df.iloc[tr][column].fillna("").astype(str), "target": y[tr]})
        stats = fold.groupby("key")["target"].agg(["count", "mean"])
        mapping = ((stats["mean"] * stats["count"] + prior * smooth) / (stats["count"] + smooth)).to_dict()
        oof[va] = [mapping.get(k, prior) for k in train_df.iloc[va][column].fillna("").astype(str)]
    full = pd.DataFrame({"key": train_df[column].fillna("").astype(str), "target": y})
    stats = full.groupby("key")["target"].agg(["count", "mean"])
    mapping = ((stats["mean"] * stats["count"] + prior * smooth) / (stats["count"] + smooth)).to_dict()
    test_vals = np.array([mapping.get(k, prior) for k in test_df[column].fillna("").astype(str)], dtype=float)
    return oof, test_vals


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    pre = ColumnTransformer(
        [
            ("num", numeric, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline([("prep", pre), ("model", HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=2000))])


def scores_to_labels(scores, thresholds):
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def tune_thresholds_constrained(y_true, oof, lambd=0.5, seed=42):
    train_dist = pd.Series(y_true).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    def objective(raw):
        thr = np.sort(raw)
        gap = np.min(np.diff(thr))
        gap_pen = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(oof, thr)
        qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
        pred_dist = pd.Series(labels).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()
        dist_pen = float(np.sum(np.abs(pred_dist - train_dist)))
        return -qwk + gap_pen + lambd * dist_pen

    res = differential_evolution(
        objective, [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed, maxiter=120, popsize=15, polish=True, updating="immediate", workers=1,
    )
    thr = np.sort(res.x)
    return thr, cohen_kappa_score(y_true, scores_to_labels(oof, thr), weights="quadratic")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    public = pd.read_csv(DATA_DIR / "public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test.csv")
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")

    train = add_basic_columns(train).assign(source_split="train").reset_index(drop=True)
    public = add_basic_columns(public).assign(source_split="public_test").reset_index(drop=True)
    private = add_basic_columns(private).assign(source_split="private_test").reset_index(drop=True)
    test = pd.concat([public, private], ignore_index=True)
    all_rows = pd.concat([train.drop(columns=[LABEL_COLUMN]), test], ignore_index=True, sort=False)

    doi_cache = pd.read_csv(EXTERNAL_DIR / "openalex_doi_features.csv")
    title_cache = pd.read_csv(EXTERNAL_DIR / "openalex_title_features.csv")

    enriched = attach_openalex(all_rows, doi_cache, title_cache)
    enriched = add_percentile_features(enriched)

    join_cols = ["source_split", "id", "title", "venue", "year", "authors", "doi",
                 "doi_norm", "first_surname", "has_authors", "title_len", "author_count"]
    train_e = train.merge(enriched.drop(columns=[LABEL_COLUMN], errors="ignore"), on=join_cols, how="left")
    test_e = test.merge(enriched, on=join_cols, how="left")

    numeric_cols = [
        "openalex_found", "title_match_score", "title_year_delta",
        "log_cited_by_count", "log_referenced_works_count", "log_fwci", "log_citation_velocity",
        "cited_by_count_global_pct", "cited_by_count_venue_pct", "cited_by_count_year_pct", "cited_by_count_venue_year_pct",
        "citation_velocity_global_pct", "citation_velocity_venue_pct", "citation_velocity_year_pct", "citation_velocity_venue_year_pct",
        "fwci_global_pct", "fwci_venue_pct", "fwci_year_pct", "fwci_venue_year_pct",
        "openalex_year", "citation_age", "title_len", "author_count", "has_authors",
        "venue_te", "year_te", "first_surname_te",
    ]
    categorical_cols = ["venue", "openalex_source"]

    y = train_e[LABEL_COLUMN].astype(int).to_numpy()
    y_float = y.astype(float)

    oof_sum = np.zeros(len(train_e))
    oof_count = np.zeros(len(train_e))
    public_sum = np.zeros(len(public))
    private_sum = np.zeros(len(private))
    n_models = 0

    for seed in SEEDS:
        # Re-fit target-encoded columns per seed (fold split changes by seed).
        for col, smooth in [("venue", 20.0), ("year", 20.0), ("first_surname", 8.0)]:
            tr_te, te_te = oof_target_encode(train_e, test_e, col, smooth, seed=seed)
            train_e[f"{col}_te"] = tr_te
            test_e[f"{col}_te"] = te_te

        x_train = train_e[numeric_cols + categorical_cols]
        x_test = test_e[numeric_cols + categorical_cols]

        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
            pipe = build_pipeline(numeric_cols, categorical_cols)
            pipe.fit(x_train.iloc[tr_idx], y_float[tr_idx])
            val_scores = np.clip(pipe.predict(x_train.iloc[va_idx]), 1.0, 5.0)
            test_scores = np.clip(pipe.predict(x_test), 1.0, 5.0)
            oof_sum[va_idx] += val_scores
            oof_count[va_idx] += 1.0
            public_sum += test_scores[: len(public)]
            private_sum += test_scores[len(public):]
            n_models += 1
            print(f"seed={seed} fold={fold} val_mae={mean_absolute_error(y_float[va_idx], val_scores):.4f}")

    oof_scores = oof_sum / np.clip(oof_count, 1.0, None)
    public_scores = public_sum / n_models
    private_scores = private_sum / n_models

    # Standalone metrics + threshold-tuned submission
    thresholds, oof_qwk = tune_thresholds_constrained(y, oof_scores)
    oof_labels = scores_to_labels(oof_scores, thresholds)
    public_labels = scores_to_labels(public_scores, thresholds)
    private_labels = scores_to_labels(private_scores, thresholds)

    print(f"\n=== OpenAlex anchor ===")
    print(f"  raw round-QWK  : {cohen_kappa_score(y, np.clip(np.round(oof_scores), 1, 5).astype(int), weights='quadratic'):.4f}")
    print(f"  constrained QWK: {oof_qwk:.4f}")
    print(f"  thresholds     : {thresholds.tolist()}")
    print(f"  OOF dist       : {dict(zip([1,2,3,4,5], pd.Series(oof_labels).value_counts().sort_index().reindex([1,2,3,4,5], fill_value=0).tolist()))}")

    # Save in same schema as SciNCL/SPECTER2
    pd.DataFrame({"id": train["id"], "Label": y, "oof_score": oof_scores, "oof_pred": oof_labels}).to_csv(
        OUT_DIR / "oof_scores.csv", index=False
    )
    pd.DataFrame({"id": public["id"], "score": public_scores, "pred": public_labels}).to_csv(
        OUT_DIR / "public_scores.csv", index=False
    )
    pd.DataFrame({"id": private["id"], "score": private_scores, "pred": private_labels}).to_csv(
        OUT_DIR / "private_scores.csv", index=False
    )

    combo = pd.concat(
        [pd.DataFrame({"id": public["id"], "Label": public_labels}),
         pd.DataFrame({"id": private["id"], "Label": private_labels})],
        ignore_index=True,
    )
    submission = sample[["id"]].merge(combo, on="id", how="left")
    submission["Label"] = submission["Label"].astype(int)
    submission.to_csv(OUT_DIR / "openalex_anchor_submission.csv", index=False)

    metrics = {
        "method": "openalex_anchor",
        "model": "HuberRegressor over OpenAlex DOI+title features + venue/year/author target encoding",
        "folds": FOLDS,
        "seeds": SEEDS,
        "n_models": n_models,
        "oof_qwk_constrained": float(oof_qwk),
        "oof_qwk_round": float(cohen_kappa_score(y, np.clip(np.round(oof_scores), 1, 5).astype(int), weights="quadratic")),
        "oof_mae": float(mean_absolute_error(y, oof_labels)),
        "oof_macro_f1": float(f1_score(y, oof_labels, average="macro")),
        "thresholds": [float(t) for t in thresholds],
        "label_distribution_combined": {int(k): int(v) for k, v in pd.Series(np.concatenate([public_labels, private_labels])).value_counts().sort_index().items()},
        "label_distribution_public": {int(k): int(v) for k, v in pd.Series(public_labels).value_counts().sort_index().items()},
        "label_distribution_private": {int(k): int(v) for k, v in pd.Series(private_labels).value_counts().sort_index().items()},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nartefacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
