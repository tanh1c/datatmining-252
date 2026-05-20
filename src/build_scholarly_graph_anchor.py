from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_YEAR = 2026
SEEDS = [252, 253, 254]
FOLDS = 5
LABEL = "Label"
ROOT = Path(__file__).resolve().parents[1]


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "semanticscholar.org" in text.lower():
        return ""
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    return text if text.lower().startswith("10.") else ""


def semantic_scholar_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "semanticscholar.org/paper/" not in text.lower():
        return ""
    return text.rstrip("/").split("/")[-1]


def first_author_surname(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    tokens = re.findall(r"[A-Za-zÀ-ỹÁ-ỹ'’-]+", str(value).split(",")[0])
    return tokens[-1].lower() if tokens else ""


def add_basic_columns(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    out["source_split"] = split
    out["doi_norm"] = out["doi"].map(normalize_doi)
    out["s2_id"] = out["doi"].map(semantic_scholar_id)
    out["first_surname"] = out["authors"].map(first_author_surname)
    out["has_authors"] = out["authors"].notna().astype(int)
    out["title_len"] = out["title"].fillna("").astype(str).str.len()
    out["author_count"] = out["authors"].fillna("").map(lambda v: 0 if not str(v).strip() else len(str(v).split(",")))
    return out.reset_index(drop=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def coalesce_by_source(rows: pd.DataFrame, doi_col: str, title_col: str) -> pd.Series:
    return np.where(rows["oa_source"].eq("doi"), rows.get(doi_col), np.where(rows["oa_source"].eq("title"), rows.get(title_col), np.nan))


def attach_openalex(rows: pd.DataFrame, external_dir: Path) -> pd.DataFrame:
    rows = rows.copy()
    doi = load_csv(external_dir / "openalex_doi_features.csv")
    title = load_csv(external_dir / "openalex_title_features.csv")
    abstracts = load_csv(external_dir / "abstracts_merged_v3.csv")

    if not doi.empty:
        doi = doi.rename(columns={
            "openalex_found": "oa_doi_found", "openalex_id": "oa_doi_id", "openalex_title": "oa_doi_title",
            "openalex_year": "oa_doi_year", "cited_by_count": "oa_doi_cited_by_count",
            "referenced_works_count": "oa_doi_referenced_works_count", "fwci": "oa_doi_fwci",
            "is_retracted": "oa_doi_is_retracted",
        })
        rows = rows.merge(doi, on="doi_norm", how="left")
    else:
        rows["oa_doi_found"] = False

    if not title.empty:
        title = title.rename(columns={
            "title_search_found": "oa_title_found", "openalex_id": "oa_title_id", "openalex_title": "oa_title_title",
            "openalex_year": "oa_title_year", "cited_by_count": "oa_title_cited_by_count",
            "referenced_works_count": "oa_title_referenced_works_count", "fwci": "oa_title_fwci",
            "is_retracted": "oa_title_is_retracted",
        })
        rows = rows.merge(title, on=["source_split", "id"], how="left")
    else:
        rows["oa_title_found"] = False

    use_doi = rows["oa_doi_found"].map(lambda v: False if pd.isna(v) else bool(v))
    use_title = (~use_doi) & rows["oa_title_found"].map(lambda v: False if pd.isna(v) else bool(v))
    rows["oa_found"] = (use_doi | use_title).astype(int)
    rows["oa_source"] = np.where(use_doi, "doi", np.where(use_title, "title", "none"))
    for field in ["id", "year", "cited_by_count", "referenced_works_count", "fwci", "is_retracted"]:
        rows[f"oa_{field}"] = coalesce_by_source(rows, f"oa_doi_{field}", f"oa_title_{field}")
    rows["oa_title_match_score"] = pd.to_numeric(rows.get("title_match_score", 0.0), errors="coerce").fillna(0.0)
    rows["oa_title_year_delta"] = pd.to_numeric(rows.get("title_year_delta", np.nan), errors="coerce")

    if not abstracts.empty:
        cols = [c for c in ["source_split", "id", "oa_type", "oa_type_crossref", "abstract_source", "abstract_len", "has_abstract", "s2_fields_of_study", "s2_publication_types"] if c in abstracts.columns]
        rows = rows.merge(abstracts[cols], on=["source_split", "id"], how="left", suffixes=("", "_abs"))
    return rows


def attach_semantic_scholar(rows: pd.DataFrame, external_dir: Path) -> pd.DataFrame:
    s2 = load_csv(external_dir / "semantic_scholar_features.csv")
    if s2.empty:
        rows["s2_found"] = 0
        rows["s2_source"] = "none"
        return rows
    doi_s2 = s2[s2["lookup_type"].eq("doi")].add_prefix("doi_")
    id_s2 = s2[s2["lookup_type"].eq("s2_id")].add_prefix("id_")
    rows = rows.merge(doi_s2, left_on="doi_norm", right_on="doi_lookup_value", how="left")
    rows = rows.merge(id_s2, left_on="s2_id", right_on="id_lookup_value", how="left")
    use_id = rows["id_s2_found"].map(lambda v: False if pd.isna(v) else bool(v)) if "id_s2_found" in rows else pd.Series(False, index=rows.index)
    use_doi = rows["doi_s2_found"].map(lambda v: False if pd.isna(v) else bool(v)) if "doi_s2_found" in rows else pd.Series(False, index=rows.index)
    rows["s2_found"] = (use_id | use_doi).astype(int)
    rows["s2_source"] = np.where(use_id, "s2_id", np.where(use_doi, "doi", "none"))
    for field in ["reference_count", "citation_count", "influential_citation_count", "year", "is_open_access", "venue", "publication_types", "fields_of_study"]:
        rows[f"s2_{field}"] = np.where(use_id, rows.get(f"id_s2_{field}"), np.where(use_doi, rows.get(f"doi_s2_{field}"), np.nan))
    return rows


def attach_other_sources(rows: pd.DataFrame, external_dir: Path) -> pd.DataFrame:
    crossref = load_csv(external_dir / "crossref_features.csv")
    coci = load_csv(external_dir / "opencitations_features.csv")
    if not crossref.empty:
        rows = rows.merge(crossref, on="doi_norm", how="left")
    if not coci.empty:
        rows = rows.merge(coci, on="doi_norm", how="left")
    return rows


def first_pipe_value(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "none"
    return str(value).split("|")[0].strip().lower() or "none"


def contains_any(value: object, terms: tuple[str, ...]) -> int:
    text = "" if pd.isna(value) else str(value).lower()
    return int(any(term in text for term in terms))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    count_cols = [
        "oa_cited_by_count", "oa_referenced_works_count", "oa_fwci",
        "s2_reference_count", "s2_citation_count", "s2_influential_citation_count",
        "crossref_reference_count", "crossref_is_referenced_by_count", "coci_citation_count",
    ]
    for col in count_cols + ["year", "oa_year", "s2_year", "abstract_len"]:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in count_cols:
        out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))

    out["paper_age"] = (CURRENT_YEAR - pd.to_numeric(out["year"], errors="coerce") + 1).clip(lower=1)
    out["best_citation_count"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].max(axis=1)
    out["best_reference_count"] = out[["oa_referenced_works_count", "s2_reference_count", "crossref_reference_count"]].max(axis=1)
    out["citation_velocity"] = out["best_citation_count"] / out["paper_age"]
    out["citation_count_mean"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].mean(axis=1)
    out["citation_disagreement"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].std(axis=1)
    out["citation_source_count"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].notna().sum(axis=1)
    out["influential_ratio"] = out["s2_influential_citation_count"] / out["s2_citation_count"].replace(0, np.nan)

    for flag in ["oa_found", "s2_found", "crossref_found", "coci_found", "has_abstract"]:
        if flag not in out:
            out[flag] = False
    out["external_found_count"] = out[["oa_found", "s2_found", "crossref_found", "coci_found"]].fillna(False).astype(int).sum(axis=1)
    out["top_s2_field"] = out.get("s2_fields_of_study", pd.Series("none", index=out.index)).map(first_pipe_value)
    out["top_s2_pub_type"] = out.get("s2_publication_types", pd.Series("none", index=out.index)).map(first_pipe_value)
    out["crossref_primary_subject"] = out.get("crossref_subjects", pd.Series("none", index=out.index)).map(first_pipe_value)
    concept_text = (
        out.get("s2_fields_of_study", pd.Series("", index=out.index)).fillna("").astype(str) + " | "
        + out.get("crossref_subjects", pd.Series("", index=out.index)).fillna("").astype(str) + " | "
        + out.get("venue", pd.Series("", index=out.index)).fillna("").astype(str) + " | "
        + out.get("s2_venue", pd.Series("", index=out.index)).fillna("").astype(str)
    )
    out["concept_has_logic"] = concept_text.map(lambda v: contains_any(v, ("logic", "reasoning", "knowledge representation")))
    out["concept_has_ai"] = concept_text.map(lambda v: contains_any(v, ("artificial intelligence", "computer science", "machine learning", "planning")))
    out["concept_has_asp"] = concept_text.map(lambda v: contains_any(v, ("answer set", "logic programming", "declarative")))

    for col in ["best_citation_count", "best_reference_count", "citation_velocity", "oa_fwci", "s2_influential_citation_count"]:
        out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))
        out[f"{col}_global_pct"] = out[col].rank(pct=True)
        for name, keys in {"venue": ["venue"], "year": ["year"], "venue_year": ["venue", "year"]}.items():
            out[f"{col}_{name}_pct"] = out.groupby(keys)[col].rank(pct=True).fillna(out[f"{col}_global_pct"])
    return out


def oof_target_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str, smooth: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = train_df[LABEL].astype(float).to_numpy()
    y_class = train_df[LABEL].astype(int).to_numpy()
    prior = float(y.mean())
    oof = np.zeros(len(train_df), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in cv.split(train_df, y_class):
        stats = pd.DataFrame({"key": train_df.iloc[tr][column].fillna("none").astype(str), "target": y[tr]}).groupby("key")["target"].agg(["count", "mean"])
        mapping = ((stats["mean"] * stats["count"] + prior * smooth) / (stats["count"] + smooth)).to_dict()
        oof[va] = [mapping.get(k, prior) for k in train_df.iloc[va][column].fillna("none").astype(str)]
    stats = pd.DataFrame({"key": train_df[column].fillna("none").astype(str), "target": y}).groupby("key")["target"].agg(["count", "mean"])
    mapping = ((stats["mean"] * stats["count"] + prior * smooth) / (stats["count"] + smooth)).to_dict()
    test = np.array([mapping.get(k, prior) for k in test_df[column].fillna("none").astype(str)], dtype=float)
    return oof, test


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def predicted_dist(labels: np.ndarray) -> np.ndarray:
    return pd.Series(labels).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, train_dist: np.ndarray, lambd: float = 0.5, seed: int = 42) -> tuple[np.ndarray, float]:
    def objective(raw: np.ndarray) -> float:
        thr = np.sort(raw)
        gap = np.min(np.diff(thr))
        gap_pen = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(scores, thr)
        qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
        return -qwk + gap_pen + lambd * float(np.sum(np.abs(predicted_dist(labels) - train_dist)))
    res = differential_evolution(objective, [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)], seed=seed, maxiter=120, popsize=15, polish=True, updating="immediate", workers=1)
    thr = np.sort(res.x)
    return thr, float(cohen_kappa_score(y_true, scores_to_labels(scores, thr), weights="quadratic"))


def make_model(kind: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical_cols),
    ])
    if kind == "ridge":
        model = Ridge(alpha=5.0)
    elif kind == "extratrees":
        model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=8, max_features=0.75, random_state=42, n_jobs=-1)
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.035, l2_regularization=0.05, min_samples_leaf=20, random_state=42)
    else:
        model = HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=2000)
    return Pipeline([("prep", pre), ("model", model)])


def train_candidate(kind: str, train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = train_df[LABEL].astype(int).to_numpy()
    y_float = y.astype(float)
    oof_sum = np.zeros(len(train_df))
    oof_count = np.zeros(len(train_df))
    test_sum = np.zeros(len(test_df))
    n_models = 0
    for seed in SEEDS:
        for col, smooth in [("venue", 20.0), ("year", 20.0), ("first_surname", 8.0), ("top_s2_field", 12.0), ("top_s2_pub_type", 12.0), ("crossref_primary_subject", 12.0)]:
            train_df[f"{col}_te"], test_df[f"{col}_te"] = oof_target_encode(train_df, test_df, col, smooth, seed)
        x_train = train_df[numeric_cols + categorical_cols]
        x_test = test_df[numeric_cols + categorical_cols]
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(cv.split(x_train, y), start=1):
            model = make_model(kind, numeric_cols, categorical_cols)
            model.fit(x_train.iloc[tr], y_float[tr])
            val_scores = np.clip(model.predict(x_train.iloc[va]), 1.0, 5.0)
            test_scores = np.clip(model.predict(x_test), 1.0, 5.0)
            oof_sum[va] += val_scores
            oof_count[va] += 1
            test_sum += test_scores
            n_models += 1
            print(f"{kind} seed={seed} fold={fold} val_mae={mean_absolute_error(y_float[va], val_scores):.4f}")
    return oof_sum / np.clip(oof_count, 1, None), test_sum / n_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--external-dir", default="outputs/external")
    parser.add_argument("--output-dir", default="outputs/scholarly_graph_anchor")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--models", default="huber,ridge,extratrees,hgb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ROOT / args.data_dir
    external_dir = ROOT / args.external_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_dir = ROOT / "outputs" / "submissions"
    submit_dir.mkdir(parents=True, exist_ok=True)

    train = add_basic_columns(pd.read_csv(data_dir / "train.csv"), "train")
    public = add_basic_columns(pd.read_csv(data_dir / "public_test.csv"), "public_test")
    private = add_basic_columns(pd.read_csv(data_dir / "private_test.csv"), "private_test")
    sample = pd.read_csv(data_dir / "Test_Submission.csv")
    test = pd.concat([public, private], ignore_index=True)
    all_rows = pd.concat([train.drop(columns=[LABEL]), test], ignore_index=True, sort=False)

    enriched = attach_openalex(all_rows, external_dir)
    enriched = attach_semantic_scholar(enriched, external_dir)
    enriched = attach_other_sources(enriched, external_dir)
    enriched = add_engineered_features(enriched)
    enriched.to_csv(out_dir / "scholarly_graph_enriched_rows.csv", index=False)

    train_e = train.merge(enriched, on=["source_split", "id", "title", "venue", "year", "authors", "doi", "doi_norm", "s2_id", "first_surname", "has_authors", "title_len", "author_count"], how="left")
    test_e = test.merge(enriched, on=["source_split", "id", "title", "venue", "year", "authors", "doi", "doi_norm", "s2_id", "first_surname", "has_authors", "title_len", "author_count"], how="left")

    base_numeric = [
        "year", "title_len", "author_count", "has_authors", "oa_found", "oa_title_match_score", "oa_title_year_delta",
        "log_oa_cited_by_count", "log_oa_referenced_works_count", "log_oa_fwci", "log_s2_reference_count", "log_s2_citation_count",
        "log_s2_influential_citation_count", "log_crossref_reference_count", "log_crossref_is_referenced_by_count", "log_coci_citation_count",
        "log_best_citation_count", "log_best_reference_count", "log_citation_velocity", "citation_count_mean", "citation_source_count",
        "citation_disagreement", "influential_ratio", "external_found_count", "paper_age", "abstract_len", "has_abstract",
        "concept_has_logic", "concept_has_ai", "concept_has_asp", "best_citation_count_venue_pct", "best_citation_count_year_pct",
        "best_citation_count_venue_year_pct", "citation_velocity_venue_pct", "citation_velocity_year_pct", "citation_velocity_venue_year_pct",
        "oa_fwci_venue_year_pct", "s2_influential_citation_count_venue_year_pct", "best_reference_count_venue_year_pct",
        "venue_te", "year_te", "first_surname_te", "top_s2_field_te", "top_s2_pub_type_te", "crossref_primary_subject_te",
    ]
    categorical = ["venue", "oa_source", "s2_source", "crossref_type", "oa_type", "oa_type_crossref", "abstract_source", "top_s2_field", "top_s2_pub_type", "crossref_primary_subject"]
    numeric_cols = [c for c in dict.fromkeys(base_numeric) if c in train_e.columns]
    categorical_cols = [c for c in categorical if c in train_e.columns]
    y = train_e[LABEL].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    rows = []
    best = None
    test_ids = test_e["id"].to_numpy()
    for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
        oof_scores, test_scores = train_candidate(kind, train_e.copy(), test_e.copy(), numeric_cols, categorical_cols)
        thresholds, qwk = tune_thresholds(y, oof_scores, train_dist)
        oof_labels = scores_to_labels(oof_scores, thresholds)
        test_labels = scores_to_labels(test_scores, thresholds)
        public_scores = test_scores[: len(public)]
        private_scores = test_scores[len(public):]
        public_labels = test_labels[: len(public)]
        private_labels = test_labels[len(public):]
        test_l1 = float(np.sum(np.abs(predicted_dist(test_labels) - train_dist)))
        row = {
            "model": kind,
            "oof_qwk": qwk,
            "oof_round_qwk": float(cohen_kappa_score(y, np.clip(np.round(oof_scores), 1, 5).astype(int), weights="quadratic")),
            "test_l1": test_l1,
            "oof_mae": float(mean_absolute_error(y, oof_labels)),
            "oof_macro_f1": float(f1_score(y, oof_labels, average="macro")),
            "thresholds": ",".join(f"{v:.6f}" for v in thresholds),
            "test_distribution": str(pd.Series(test_labels).value_counts().sort_index().astype(int).to_dict()),
        }
        rows.append(row)
        model_dir = out_dir / kind
        model_dir.mkdir(exist_ok=True)
        pd.DataFrame({"id": train["id"], "Label": y, "oof_score": oof_scores, "oof_pred": oof_labels}).to_csv(model_dir / "oof_scores.csv", index=False)
        pd.DataFrame({"id": public["id"], "score": public_scores, "pred": public_labels}).to_csv(model_dir / "public_scores.csv", index=False)
        pd.DataFrame({"id": private["id"], "score": private_scores, "pred": private_labels}).to_csv(model_dir / "private_scores.csv", index=False)
        combo = pd.DataFrame({"id": test_ids, "Label": test_labels})
        sub = sample[["id"]].merge(combo, on="id", how="left")
        sub["Label"] = sub["Label"].astype(int)
        sub.to_csv(model_dir / "submission.csv", index=False)
        if best is None or qwk > best["row"]["oof_qwk"]:
            best = {"kind": kind, "row": row, "model_dir": model_dir}
        print(f"{kind}: OOF={qwk:.4f} round={row['oof_round_qwk']:.4f} test_L1={test_l1:.4f}")

    report = pd.DataFrame(rows).sort_values("oof_qwk", ascending=False)
    report.to_csv(out_dir / "model_report.csv", index=False)
    best_dir = best["model_dir"]
    for name in ["oof_scores.csv", "public_scores.csv", "private_scores.csv", "submission.csv"]:
        pd.read_csv(best_dir / name).to_csv(out_dir / name if name != "submission.csv" else out_dir / "scholarly_graph_anchor_submission.csv", index=False)
    pd.read_csv(best_dir / "submission.csv").to_csv(submit_dir / "next_scholarly_graph_anchor_submission.csv", index=False)
    metrics = {
        "method": "scholarly_graph_anchor",
        "best_model": best["kind"],
        "best": best["row"],
        "models": rows,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "coverage": {c: float(train_e[c].notna().mean()) for c in ["oa_found", "s2_found", "crossref_found", "coci_found", "has_abstract"] if c in train_e.columns},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(report.to_string(index=False))
    print(f"best -> {best['kind']} artefacts in {out_dir}")


if __name__ == "__main__":
    main()
