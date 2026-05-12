from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from scipy.optimize import differential_evolution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths


CURRENT_YEAR = 2026
RIDGE_ANCHOR_EXPERIMENT = "ridge_threshold_5fold_5seed_891e6ee4"


@dataclass(frozen=True)
class SourcePaths:
    external_dir: Path
    openalex_doi: Path
    openalex_title: Path
    semantic_scholar: Path
    crossref: Path
    opencitations: Path


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


def add_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["doi_norm"] = out["doi"].map(normalize_doi)
    out["s2_id"] = out["doi"].map(semantic_scholar_id)
    out["first_surname"] = out["authors"].map(first_author_surname)
    out["has_authors"] = out["authors"].notna().astype(int)
    out["title_len"] = out["title"].fillna("").astype(str).str.len()
    out["author_count"] = out["authors"].fillna("").map(lambda value: 0 if not str(value).strip() else len(str(value).split(",")))
    return out


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retries: int = 3,
    sleep_seconds: float = 1.0,
    **kwargs: object,
) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(retries):
        response = session.request(method, url, timeout=30, **kwargs)
        last_response = response
        if response.status_code in {429, 500, 502, 503, 504} and attempt < retries - 1:
            time.sleep(sleep_seconds * (attempt + 1))
            continue
        return response
    if last_response is None:
        raise RuntimeError("request failed before response")
    return last_response


def source_paths(project_root: Path, external_dir: str) -> SourcePaths:
    root = (project_root / external_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return SourcePaths(
        external_dir=root,
        openalex_doi=root / "openalex_doi_features.csv",
        openalex_title=root / "openalex_title_features.csv",
        semantic_scholar=root / "semantic_scholar_features.csv",
        crossref=root / "crossref_features.csv",
        opencitations=root / "opencitations_features.csv",
    )


def fetch_semantic_scholar(all_rows: pd.DataFrame, path: Path, batch_size: int, sleep_seconds: float) -> pd.DataFrame:
    if path.exists():
        cache = pd.read_csv(path)
    else:
        cache = pd.DataFrame()

    identifiers: list[tuple[str, str, str]] = []
    for _, row in all_rows.iterrows():
        if row["s2_id"]:
            identifiers.append((f"S2:{row['s2_id']}", "s2_id", row["s2_id"]))
        elif row["doi_norm"]:
            identifiers.append((f"DOI:{row['doi_norm']}", "doi", row["doi_norm"]))
    identifiers = sorted(set(identifiers))

    cached = set(cache["lookup_id"].astype(str)) if not cache.empty and "lookup_id" in cache.columns else set()
    remaining = [item for item in identifiers if item[0] not in cached]
    print(f"Semantic Scholar remaining={len(remaining)} cached={len(cached)}")
    if not remaining:
        return cache

    session = requests.Session()
    headers = {}
    api_key = os.environ.get("S2_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    fields = ",".join(
        [
            "title",
            "venue",
            "year",
            "referenceCount",
            "citationCount",
            "influentialCitationCount",
            "publicationTypes",
            "fieldsOfStudy",
            "isOpenAccess",
        ]
    )
    rows = []
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        ids = [item[0] for item in batch]
        try:
            response = request_with_retries(
                session,
                "POST",
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": fields},
                json={"ids": ids},
                headers=headers,
                sleep_seconds=max(sleep_seconds, 1.0),
            )
            if response.status_code == 200:
                payload = response.json()
                for item, data in zip(batch, payload, strict=False):
                    lookup_id, lookup_type, lookup_value = item
                    data = data or {}
                    rows.append(
                        {
                            "lookup_id": lookup_id,
                            "lookup_type": lookup_type,
                            "lookup_value": lookup_value,
                            "s2_found": bool(data.get("paperId")),
                            "s2_error": "",
                            "s2_paper_id": data.get("paperId", ""),
                            "s2_title": data.get("title", ""),
                            "s2_venue": data.get("venue", ""),
                            "s2_year": data.get("year", np.nan),
                            "s2_reference_count": data.get("referenceCount", np.nan),
                            "s2_citation_count": data.get("citationCount", np.nan),
                            "s2_influential_citation_count": data.get("influentialCitationCount", np.nan),
                            "s2_publication_types": "|".join(data.get("publicationTypes") or []),
                            "s2_fields_of_study": "|".join(data.get("fieldsOfStudy") or []),
                            "s2_is_open_access": data.get("isOpenAccess", np.nan),
                        }
                    )
            else:
                for item in batch:
                    rows.append(
                        {
                            "lookup_id": item[0],
                            "lookup_type": item[1],
                            "lookup_value": item[2],
                            "s2_found": False,
                            "s2_error": f"{response.status_code}: {response.text[:160]}",
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            for item in batch:
                rows.append(
                    {
                        "lookup_id": item[0],
                        "lookup_type": item[1],
                        "lookup_value": item[2],
                        "s2_found": False,
                        "s2_error": repr(exc),
                    }
                )
        if rows:
            out = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            out = out.drop_duplicates("lookup_id", keep="last")
            out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Semantic Scholar progress {min(start + batch_size, len(remaining))}/{len(remaining)}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.read_csv(path)


def fetch_crossref(all_rows: pd.DataFrame, path: Path, mailto: str, sleep_seconds: float) -> pd.DataFrame:
    if path.exists():
        cache = pd.read_csv(path)
    else:
        cache = pd.DataFrame()
    dois = sorted(set(all_rows["doi_norm"]) - {""})
    cached = set(cache["doi_norm"].astype(str)) if not cache.empty and "doi_norm" in cache.columns else set()
    remaining = [doi for doi in dois if doi not in cached]
    print(f"Crossref remaining={len(remaining)} cached={len(cached)}")
    session = requests.Session()
    rows = []
    for idx, doi in enumerate(remaining, start=1):
        record = {"doi_norm": doi, "crossref_found": False, "crossref_error": ""}
        try:
            response = request_with_retries(
                session,
                "GET",
                "https://api.crossref.org/works/" + quote(doi, safe=""),
                params={"mailto": mailto},
                sleep_seconds=max(sleep_seconds, 1.0),
            )
            if response.status_code == 200:
                msg = response.json().get("message", {})
                subjects = msg.get("subject") or []
                licenses = msg.get("license") or []
                record.update(
                    {
                        "crossref_found": True,
                        "crossref_title": " ".join(msg.get("title") or []),
                        "crossref_publisher": msg.get("publisher", ""),
                        "crossref_type": msg.get("type", ""),
                        "crossref_subjects": "|".join(subjects),
                        "crossref_subject_count": len(subjects),
                        "crossref_reference_count": msg.get("reference-count", np.nan),
                        "crossref_is_referenced_by_count": msg.get("is-referenced-by-count", np.nan),
                        "crossref_license_count": len(licenses),
                        "crossref_has_abstract": int(bool(msg.get("abstract"))),
                    }
                )
            else:
                record["crossref_error"] = f"{response.status_code}: {response.text[:160]}"
        except Exception as exc:  # noqa: BLE001
            record["crossref_error"] = repr(exc)
        rows.append(record)
        if idx % 50 == 0 or idx == len(remaining):
            out = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            out = out.drop_duplicates("doi_norm", keep="last")
            out.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"Crossref progress {idx}/{len(remaining)}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.read_csv(path) if path.exists() else cache


def fetch_opencitations(all_rows: pd.DataFrame, path: Path, sleep_seconds: float) -> pd.DataFrame:
    if path.exists():
        cache = pd.read_csv(path)
    else:
        cache = pd.DataFrame()
    dois = sorted(set(all_rows["doi_norm"]) - {""})
    cached = set(cache["doi_norm"].astype(str)) if not cache.empty and "doi_norm" in cache.columns else set()
    remaining = [doi for doi in dois if doi not in cached]
    print(f"OpenCitations remaining={len(remaining)} cached={len(cached)}")
    session = requests.Session()
    rows = []
    for idx, doi in enumerate(remaining, start=1):
        record = {"doi_norm": doi, "coci_found": False, "coci_error": ""}
        try:
            response = request_with_retries(
                session,
                "GET",
                "https://api.opencitations.net/index/v1/citation-count/" + doi,
                sleep_seconds=max(sleep_seconds, 1.0),
            )
            if response.status_code == 200:
                data = response.json()
                count = data[0].get("count") if data else np.nan
                record.update({"coci_found": True, "coci_citation_count": count})
            else:
                record["coci_error"] = f"{response.status_code}: {response.text[:160]}"
        except Exception as exc:  # noqa: BLE001
            record["coci_error"] = repr(exc)
        rows.append(record)
        if idx % 50 == 0 or idx == len(remaining):
            out = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            out = out.drop_duplicates("doi_norm", keep="last")
            out.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"OpenCitations progress {idx}/{len(remaining)}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.read_csv(path) if path.exists() else cache


def load_anchor_scores(project_root: Path) -> tuple[np.ndarray, np.ndarray]:
    report_dir = project_root / "outputs" / "ridge_threshold" / "reports"
    oof = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_oof.csv")["oof_score"].to_numpy()
    public_scores = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_public_scores.csv")["score"].to_numpy()
    private_scores = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_private_scores.csv")["score"].to_numpy()
    return oof, np.concatenate([public_scores, private_scores])


def attach_openalex(all_rows: pd.DataFrame, paths: SourcePaths) -> pd.DataFrame:
    rows = all_rows.copy()
    doi_cache = pd.read_csv(paths.openalex_doi)
    title_cache = pd.read_csv(paths.openalex_title) if paths.openalex_title.exists() else pd.DataFrame()
    doi_cache = doi_cache.rename(
        columns={
            "openalex_found": "oa_doi_found",
            "openalex_year": "oa_doi_year",
            "cited_by_count": "oa_doi_cited_by_count",
            "referenced_works_count": "oa_doi_referenced_works_count",
            "fwci": "oa_doi_fwci",
            "is_retracted": "oa_doi_is_retracted",
        }
    )
    rows = rows.merge(doi_cache, on="doi_norm", how="left")
    if not title_cache.empty:
        title_cache = title_cache.rename(
            columns={
                "title_search_found": "oa_title_found",
                "openalex_year": "oa_title_year",
                "cited_by_count": "oa_title_cited_by_count",
                "referenced_works_count": "oa_title_referenced_works_count",
                "fwci": "oa_title_fwci",
                "is_retracted": "oa_title_is_retracted",
            }
        )
        rows = rows.merge(title_cache, on=["source_split", "id"], how="left")
    else:
        rows["oa_title_found"] = False

    use_doi = rows["oa_doi_found"].map(lambda value: False if pd.isna(value) else bool(value))
    use_title = (~use_doi) & rows["oa_title_found"].map(lambda value: False if pd.isna(value) else bool(value))
    rows["oa_found"] = (use_doi | use_title).astype(int)
    rows["oa_source"] = np.where(use_doi, "doi", np.where(use_title, "title", "none"))
    for field in ["year", "cited_by_count", "referenced_works_count", "fwci", "is_retracted"]:
        rows[f"oa_{field}"] = np.where(
            use_doi,
            rows.get(f"oa_doi_{field}"),
            np.where(use_title, rows.get(f"oa_title_{field}"), np.nan),
        )
    rows["oa_title_match_score"] = pd.to_numeric(rows.get("title_match_score", 0.0), errors="coerce").fillna(0.0)
    return rows


def attach_external(all_rows: pd.DataFrame, paths: SourcePaths) -> pd.DataFrame:
    rows = attach_openalex(all_rows, paths)
    for pfx, path in [("crossref", paths.crossref), ("coci", paths.opencitations)]:
        if path.exists():
            rows = rows.merge(pd.read_csv(path), on="doi_norm", how="left", suffixes=("", f"_{pfx}"))
    if paths.semantic_scholar.exists():
        s2 = pd.read_csv(paths.semantic_scholar)
        doi_s2 = s2[s2["lookup_type"].eq("doi")].add_prefix("doi_")
        s2id_s2 = s2[s2["lookup_type"].eq("s2_id")].add_prefix("s2id_")
        rows = rows.merge(doi_s2, left_on="doi_norm", right_on="doi_lookup_value", how="left")
        rows = rows.merge(s2id_s2, left_on="s2_id", right_on="s2id_lookup_value", how="left")
        use_s2id = rows["s2id_s2_found"].map(lambda value: False if pd.isna(value) else bool(value)) if "s2id_s2_found" in rows else pd.Series(False, index=rows.index)
        use_doi = rows["doi_s2_found"].map(lambda value: False if pd.isna(value) else bool(value)) if "doi_s2_found" in rows else pd.Series(False, index=rows.index)
        rows["s2_found"] = (use_s2id | use_doi).astype(int)
        rows["s2_source"] = np.where(use_s2id, "s2_id", np.where(use_doi, "doi", "none"))
        for field in ["reference_count", "citation_count", "influential_citation_count", "year", "is_open_access"]:
            rows[f"s2_{field}"] = np.where(
                use_s2id,
                rows.get(f"s2id_s2_{field}"),
                np.where(use_doi, rows.get(f"doi_s2_{field}"), np.nan),
            )
    else:
        rows["s2_found"] = 0
        rows["s2_source"] = "none"
    return rows


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    count_cols = [
        "oa_cited_by_count",
        "oa_referenced_works_count",
        "oa_fwci",
        "crossref_reference_count",
        "crossref_is_referenced_by_count",
        "coci_citation_count",
        "s2_reference_count",
        "s2_citation_count",
        "s2_influential_citation_count",
    ]
    for col in count_cols + ["oa_year", "s2_year", "year"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in count_cols:
        if col not in out:
            out[col] = np.nan
        out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))

    out["best_citation_count"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].max(axis=1)
    out["best_reference_count"] = out[["oa_referenced_works_count", "s2_reference_count", "crossref_reference_count"]].max(axis=1)
    out["citation_count_mean"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].mean(axis=1)
    out["citation_source_count"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].notna().sum(axis=1)
    out["citation_disagreement"] = out[["oa_cited_by_count", "s2_citation_count", "crossref_is_referenced_by_count", "coci_citation_count"]].std(axis=1)
    out["paper_age"] = (CURRENT_YEAR - pd.to_numeric(out["year"], errors="coerce") + 1).clip(lower=1)
    out["citation_velocity"] = out["best_citation_count"] / out["paper_age"]
    out["influential_ratio"] = out["s2_influential_citation_count"] / out["s2_citation_count"].replace(0, np.nan)
    for source_flag in ["oa_found", "crossref_found", "coci_found", "s2_found"]:
        if source_flag not in out:
            out[source_flag] = False
    out["external_found_count"] = out[["oa_found", "crossref_found", "coci_found", "s2_found"]].fillna(False).astype(int).sum(axis=1)

    for col in ["best_citation_count", "citation_velocity", "oa_fwci", "s2_influential_citation_count", "best_reference_count"]:
        out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))
        out[f"{col}_global_pct"] = out[col].rank(pct=True)
        for name, keys in {"venue": ["venue"], "year": ["year"], "venue_year": ["venue", "year"]}.items():
            out[f"{col}_{name}_pct"] = out.groupby(keys)[col].rank(pct=True).fillna(out[f"{col}_global_pct"])
    return out


def add_oof_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str, smooth: float) -> tuple[np.ndarray, np.ndarray]:
    y = train_df[LABEL_COLUMN].astype(float).to_numpy()
    y_class = train_df[LABEL_COLUMN].astype(int).to_numpy()
    prior = float(np.mean(y))
    oof = np.zeros(len(train_df), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=252)
    for train_idx, valid_idx in cv.split(train_df, y_class):
        fold = pd.DataFrame({"key": train_df.iloc[train_idx][column].fillna("").astype(str), "target": y[train_idx]})
        stats = fold.groupby("key")["target"].agg(["count", "mean"])
        mapping = ((stats["mean"] * stats["count"] + prior * smooth) / (stats["count"] + smooth)).to_dict()
        oof[valid_idx] = [mapping.get(key, prior) for key in train_df.iloc[valid_idx][column].fillna("").astype(str)]
    full = pd.DataFrame({"key": train_df[column].fillna("").astype(str), "target": y})
    stats = full.groupby("key")["target"].agg(["count", "mean"])
    mapping = ((stats["mean"] * stats["count"] + prior * smooth) / (stats["count"] + smooth)).to_dict()
    test_values = np.array([mapping.get(key, prior) for key in test_df[column].fillna("").astype(str)], dtype=float)
    return oof, test_values


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, seed: int) -> tuple[np.ndarray, float, float, dict[int, int]]:
    def objective(raw_thresholds: np.ndarray) -> float:
        thresholds = np.sort(raw_thresholds)
        min_gap = np.min(np.diff(thresholds))
        penalty = 0.0 if min_gap >= 0.03 else (0.03 - min_gap) * 5.0
        labels = scores_to_labels(scores, thresholds)
        return -cohen_kappa_score(y_true, labels, weights="quadratic") + penalty

    result = differential_evolution(
        objective,
        bounds=[(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed,
        maxiter=70,
        popsize=10,
        polish=True,
        updating="immediate",
        workers=1,
    )
    thresholds = np.sort(result.x)
    labels = scores_to_labels(scores, thresholds)
    return (
        thresholds,
        float(cohen_kappa_score(y_true, labels, weights="quadratic")),
        float(mean_absolute_error(y_true, labels)),
        pd.Series(labels).value_counts().sort_index().astype(int).to_dict(),
    )


def train_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    output_dir: Path,
    report_dir: Path,
    project_root: Path,
) -> pd.DataFrame:
    anchor_oof, anchor_test = load_anchor_scores(project_root)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["ridge_score"] = anchor_oof
    test_df["ridge_score"] = anchor_test
    for column, smooth in [("venue", 20.0), ("year", 20.0), ("first_surname", 8.0)]:
        train_df[f"{column}_te"], test_df[f"{column}_te"] = add_oof_target_encoding(train_df, test_df, column, smooth)

    numeric_cols = [
        "ridge_score",
        "year",
        "title_len",
        "author_count",
        "has_authors",
        "venue_te",
        "year_te",
        "first_surname_te",
        "oa_found",
        "oa_title_match_score",
        "log_oa_cited_by_count",
        "log_oa_referenced_works_count",
        "log_oa_fwci",
        "log_crossref_reference_count",
        "log_crossref_is_referenced_by_count",
        "log_coci_citation_count",
        "log_s2_reference_count",
        "log_s2_citation_count",
        "log_s2_influential_citation_count",
        "log_best_citation_count",
        "log_citation_velocity",
        "log_s2_influential_citation_count",
        "log_best_reference_count",
        "citation_count_mean",
        "citation_source_count",
        "citation_disagreement",
        "influential_ratio",
        "external_found_count",
        "best_citation_count_venue_pct",
        "best_citation_count_year_pct",
        "best_citation_count_venue_year_pct",
        "citation_velocity_venue_pct",
        "citation_velocity_year_pct",
        "citation_velocity_venue_year_pct",
        "oa_fwci_venue_year_pct",
        "s2_influential_citation_count_venue_year_pct",
        "best_reference_count_venue_year_pct",
    ]
    numeric_cols = list(dict.fromkeys([col for col in numeric_cols if col in train_df.columns]))
    categorical_cols = [col for col in ["venue", "oa_source", "s2_source", "crossref_type"] if col in train_df.columns]
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    models = {
        "huber_scholarly": Pipeline([("prep", preprocessor), ("model", HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000))]),
        "ridge_scholarly": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=5.0))]),
    }
    x_train = train_df[numeric_cols + categorical_cols]
    x_test = test_df[numeric_cols + categorical_cols]
    y = train_df[LABEL_COLUMN].astype(int).to_numpy()
    y_float = y.astype(float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        print(f"training {model_name}")
        oof = np.clip(cross_val_predict(model, x_train, y_float, cv=cv), 1.0, 5.0)
        model.fit(x_train, y_float)
        test_scores = np.clip(model.predict(x_test), 1.0, 5.0)
        for weight in [1.0, 0.85, 0.75, 0.65, 0.5, 0.35, 0.25]:
            blend_oof = oof if weight == 1.0 else (1 - weight) * anchor_oof + weight * oof
            blend_test = test_scores if weight == 1.0 else (1 - weight) * anchor_test + weight * test_scores
            thresholds, qwk, mae, train_dist = tune_thresholds(y, blend_oof, 42)
            labels = scores_to_labels(blend_test, thresholds)
            name = f"{model_name}_w{weight:.2f}"
            submission_path = output_dir / f"{name}_submission.csv"
            pd.DataFrame({"id": sample_df["id"], LABEL_COLUMN: labels}).to_csv(submission_path, index=False)
            rows.append(
                {
                    "name": name,
                    "model": model_name,
                    "external_weight": weight,
                    "ridge_weight": 1 - weight,
                    "oof_qwk": qwk,
                    "mae": mae,
                    "thresholds": ",".join(f"{value:.12f}" for value in thresholds),
                    "train_distribution": train_dist,
                    "test_distribution": pd.Series(labels).value_counts().sort_index().astype(int).to_dict(),
                    "submission": str(submission_path),
                }
            )
    report = pd.DataFrame(rows).sort_values("oof_qwk", ascending=False)
    report.to_csv(report_dir / "scholarly_meta_candidates.csv", index=False)
    best = report.iloc[0]
    best_copy = project_root / "outputs" / "submissions" / "next_scholarly_meta_submission.csv"
    best_copy.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(best["submission"]).to_csv(best_copy, index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex + Semantic Scholar + Crossref + OpenCitations meta candidates.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/scholarly_meta/submissions")
    parser.add_argument("--report-dir", default="outputs/scholarly_meta/reports")
    parser.add_argument("--external-dir", default="outputs/external")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--mailto", default="codex@example.com")
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--s2-batch-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train_df, public_df, private_df, sample_df = load_data(paths)
    train_df = add_basic_columns(train_df).assign(source_split="train").reset_index(drop=True)
    public_df = add_basic_columns(public_df).assign(source_split="public_test").reset_index(drop=True)
    private_df = add_basic_columns(private_df).assign(source_split="private_test").reset_index(drop=True)
    test_df = pd.concat([public_df, private_df], ignore_index=True)
    all_rows = pd.concat([train_df.drop(columns=[LABEL_COLUMN]), test_df], ignore_index=True, sort=False)
    spaths = source_paths(paths.project_root, args.external_dir)
    if not args.skip_fetch:
        fetch_semantic_scholar(all_rows, spaths.semantic_scholar, args.s2_batch_size, args.sleep)
        fetch_crossref(all_rows, spaths.crossref, args.mailto, args.sleep)
        fetch_opencitations(all_rows, spaths.opencitations, args.sleep)

    enriched = attach_external(all_rows, spaths)
    enriched = add_engineered_features(enriched)
    report_dir = (paths.project_root / args.report_dir).resolve()
    output_dir = (paths.project_root / args.output_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(report_dir / "scholarly_enriched_rows.csv", index=False, encoding="utf-8-sig")

    join_cols = ["source_split", "id", "title", "venue", "year", "authors", "doi", "doi_norm", "s2_id", "first_surname", "has_authors", "title_len", "author_count"]
    train_enriched = train_df.merge(enriched, on=join_cols, how="left")
    test_enriched = test_df.merge(enriched, on=join_cols, how="left")
    report = train_candidates(train_enriched, test_enriched, sample_df, output_dir, report_dir, paths.project_root)

    coverage_cols = [col for col in ["oa_source", "crossref_found", "coci_found", "s2_source"] if col in enriched.columns]
    coverage = enriched.groupby("source_split")[coverage_cols].agg(lambda values: values.astype(str).value_counts().to_dict())
    coverage.to_csv(report_dir / "scholarly_source_coverage.csv")
    summary = [
        "# Scholarly Meta Candidates",
        "",
        "Features combine OpenAlex, Semantic Scholar, Crossref, and OpenCitations caches.",
        "",
        "## Top Candidates",
        "",
        report.head(12).to_markdown(index=False),
        "",
        "## Coverage",
        "",
        coverage.to_markdown(),
    ]
    (report_dir / "scholarly_meta_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(report.head(12).to_string(index=False))
    print(f"best copy: {paths.project_root / 'outputs' / 'submissions' / 'next_scholarly_meta_submission.csv'}")


if __name__ == "__main__":
    main()
