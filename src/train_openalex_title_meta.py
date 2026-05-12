from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
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
RANDOM_STATE = 42
RIDGE_ANCHOR_EXPERIMENT = "ridge_threshold_5fold_5seed_891e6ee4"


@dataclass(frozen=True)
class Candidate:
    name: str
    model: Pipeline


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(value: object) -> str:
    return clean_text(value)


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


def title_similarity(left: object, right: object) -> float:
    a = normalize_title(left)
    b = normalize_title(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def openalex_get(session: requests.Session, url: str, params: dict[str, object], retries: int = 3) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(retries):
        response = session.get(url, params=params, timeout=30)
        last_response = response
        if response.status_code in {429, 500, 502, 503, 504} and attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
        return response
    if last_response is None:
        raise RuntimeError("OpenAlex request failed before receiving a response.")
    return last_response


def openalex_fields() -> str:
    return ",".join(
        [
            "id",
            "doi",
            "display_name",
            "publication_year",
            "cited_by_count",
            "referenced_works_count",
            "fwci",
            "is_retracted",
        ]
    )


def select_title_match(
    results: list[dict[str, object]],
    source_title: str,
    source_year: object,
    min_match: float,
) -> dict[str, object]:
    best: dict[str, object] = {
        "title_search_found": False,
        "title_search_error": "",
        "title_match_score": 0.0,
        "title_year_delta": np.nan,
    }
    source_year_num = pd.to_numeric(source_year, errors="coerce")
    best_rank_score = -1.0

    for item in results:
        candidate_title = item.get("display_name", "")
        similarity = title_similarity(source_title, candidate_title)
        openalex_year = pd.to_numeric(item.get("publication_year"), errors="coerce")
        year_delta = np.nan
        year_bonus = 0.0
        if not pd.isna(source_year_num) and not pd.isna(openalex_year):
            year_delta = float(openalex_year - source_year_num)
            if abs(year_delta) <= 1:
                year_bonus = 0.05
            elif abs(year_delta) <= 3:
                year_bonus = 0.02
        rank_score = similarity + year_bonus
        if rank_score > best_rank_score:
            best_rank_score = rank_score
            best = {
                "title_search_found": similarity >= min_match,
                "title_search_error": "",
                "title_match_score": float(similarity),
                "title_year_delta": year_delta,
                "openalex_id": item.get("id", ""),
                "openalex_doi": item.get("doi", ""),
                "openalex_title": candidate_title,
                "openalex_year": item.get("publication_year", np.nan),
                "cited_by_count": item.get("cited_by_count", np.nan),
                "referenced_works_count": item.get("referenced_works_count", np.nan),
                "fwci": item.get("fwci", np.nan),
                "is_retracted": item.get("is_retracted", np.nan),
            }
    if not best.get("title_search_found", False):
        best["title_search_found"] = False
    return best


def fetch_title_search_cache(
    all_rows: pd.DataFrame,
    cache_path: Path,
    min_match: float,
    per_page: int,
    sleep_seconds: float,
    mailto: str,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key_cols = ["source_split", "id"]
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=key_cols)

    cached_keys = set()
    if not cache.empty:
        cached_keys = set(zip(cache["source_split"].astype(str), cache["id"].astype(int)))

    targets = all_rows[all_rows["doi_norm"].eq("")].copy()
    targets = targets[~targets.apply(lambda row: (str(row["source_split"]), int(row["id"])) in cached_keys, axis=1)]
    print(f"title-search rows: {len(targets)} remaining, cached={len(cached_keys)}")

    session = requests.Session()
    rows: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(targets.iterrows(), start=1):
        record = {
            "source_split": row["source_split"],
            "id": int(row["id"]),
            "source_title": row.get("title", ""),
            "source_year": row.get("year", np.nan),
        }
        query = normalize_title(row.get("title", ""))
        if not query:
            record.update(
                {
                    "title_search_found": False,
                    "title_search_error": "empty title",
                    "title_match_score": 0.0,
                    "title_year_delta": np.nan,
                }
            )
        else:
            try:
                response = openalex_get(
                    session,
                    "https://api.openalex.org/works",
                    {
                        "search": query,
                        "per-page": per_page,
                        "select": openalex_fields(),
                        "mailto": mailto,
                    },
                )
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    record.update(select_title_match(results, row.get("title", ""), row.get("year"), min_match))
                else:
                    record.update(
                        {
                            "title_search_found": False,
                            "title_search_error": f"{response.status_code}: {response.text[:160]}",
                            "title_match_score": 0.0,
                            "title_year_delta": np.nan,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - cache the failure and continue.
                record.update(
                    {
                        "title_search_found": False,
                        "title_search_error": repr(exc),
                        "title_match_score": 0.0,
                        "title_year_delta": np.nan,
                    }
                )
        rows.append(record)

        if idx % 50 == 0 or idx == len(targets):
            new_cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            new_cache.to_csv(cache_path, index=False, encoding="utf-8-sig")
            found = int(pd.Series([item.get("title_search_found", False) for item in rows]).sum())
            print(f"title-search progress {idx}/{len(targets)} new_found={found}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates(key_cols, keep="last")
        cache.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return cache


def load_anchor_scores(project_root: Path) -> tuple[np.ndarray, np.ndarray]:
    report_dir = project_root / "outputs" / "ridge_threshold" / "reports"
    oof = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_oof.csv")["oof_score"].to_numpy()
    public_scores = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_public_scores.csv")["score"].to_numpy()
    private_scores = pd.read_csv(report_dir / f"{RIDGE_ANCHOR_EXPERIMENT}_private_scores.csv")["score"].to_numpy()
    return oof, np.concatenate([public_scores, private_scores])


def attach_openalex_features(all_rows: pd.DataFrame, doi_cache: pd.DataFrame, title_cache: pd.DataFrame) -> pd.DataFrame:
    rows = all_rows.copy()
    doi_cache = doi_cache.copy()
    title_cache = title_cache.copy()

    doi_columns = {
        "openalex_found": "doi_openalex_found",
        "openalex_id": "doi_openalex_id",
        "openalex_title": "doi_openalex_title",
        "openalex_year": "doi_openalex_year",
        "cited_by_count": "doi_cited_by_count",
        "referenced_works_count": "doi_referenced_works_count",
        "fwci": "doi_fwci",
        "is_retracted": "doi_is_retracted",
    }
    rows = rows.merge(doi_cache.rename(columns=doi_columns), on="doi_norm", how="left")

    title_columns = {
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
        rows = rows.merge(
            title_cache.rename(columns=title_columns),
            on=["source_split", "id"],
            how="left",
        )
    else:
        for column in title_columns.values():
            rows[column] = np.nan
        rows["title_match_score"] = np.nan
        rows["title_year_delta"] = np.nan

    use_doi = rows["doi_openalex_found"].map(lambda value: False if pd.isna(value) else bool(value))
    title_found = rows["title_openalex_found"].map(lambda value: False if pd.isna(value) else bool(value))
    use_title = (~use_doi) & title_found
    rows["openalex_source"] = np.where(use_doi, "doi", np.where(use_title, "title", "none"))
    rows["openalex_found"] = use_doi | use_title

    for field in ["openalex_id", "openalex_title", "openalex_year", "cited_by_count", "referenced_works_count", "fwci", "is_retracted"]:
        doi_field = f"doi_{field}" if field.startswith("openalex") or field == "is_retracted" else f"doi_{field}"
        title_field = f"title_{field}" if field.startswith("openalex") or field == "is_retracted" else f"title_{field}"
        if field == "cited_by_count":
            doi_field = "doi_cited_by_count"
            title_field = "title_cited_by_count"
        elif field == "referenced_works_count":
            doi_field = "doi_referenced_works_count"
            title_field = "title_referenced_works_count"
        elif field == "fwci":
            doi_field = "doi_fwci"
            title_field = "title_fwci"
        rows[field] = np.where(use_doi, rows.get(doi_field), np.where(use_title, rows.get(title_field), np.nan))

    return rows


def add_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["cited_by_count", "referenced_works_count", "fwci", "openalex_year"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["openalex_found"] = out["openalex_found"].map(lambda value: 0 if pd.isna(value) else int(bool(value)))
    out["title_match_score"] = pd.to_numeric(out.get("title_match_score", np.nan), errors="coerce").fillna(0.0)
    out["title_year_delta"] = pd.to_numeric(out.get("title_year_delta", np.nan), errors="coerce")
    out["citation_age"] = (CURRENT_YEAR - out["openalex_year"] + 1).clip(lower=1)
    out["citation_velocity"] = out["cited_by_count"] / out["citation_age"]

    for column in ["cited_by_count", "referenced_works_count", "fwci", "citation_velocity"]:
        out[f"log_{column}"] = np.log1p(out[column].clip(lower=0))
        out[f"{column}_global_pct"] = out[column].rank(pct=True)

    group_keys = {
        "venue": ["venue"],
        "year": ["year"],
        "venue_year": ["venue", "year"],
    }
    for name, keys in group_keys.items():
        for column in ["cited_by_count", "citation_velocity", "fwci"]:
            out[f"{column}_{name}_pct"] = out.groupby(keys)[column].rank(pct=True)
            out[f"{column}_{name}_pct"] = out[f"{column}_{name}_pct"].fillna(out[f"{column}_global_pct"])

    return out


def add_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["doi_norm"] = out["doi"].map(normalize_doi)
    out["first_surname"] = out["authors"].map(first_author_surname)
    out["has_authors"] = out["authors"].notna().astype(int)
    out["title_len"] = out["title"].fillna("").astype(str).str.len()
    out["author_count"] = out["authors"].fillna("").map(lambda value: 0 if not str(value).strip() else len(str(value).split(",")))
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
    qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
    mae = mean_absolute_error(y_true, labels)
    distribution = pd.Series(labels).value_counts().sort_index().astype(int).to_dict()
    return thresholds, float(qwk), float(mae), distribution


def make_candidates(numeric_columns: list[str], categorical_columns: list[str]) -> list[Candidate]:
    numeric_scaled = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_scaled, numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    return [
        Candidate(
            "huber_meta",
            Pipeline(
                [
                    ("prep", preprocessor),
                    ("model", HuberRegressor(alpha=0.01, epsilon=1.5, max_iter=1000)),
                ]
            ),
        ),
        Candidate(
            "ridge_meta",
            Pipeline(
                [
                    ("prep", preprocessor),
                    ("model", Ridge(alpha=5.0)),
                ]
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex title-search enrichment and meta-model candidates.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/openalex_title_meta/submissions")
    parser.add_argument("--report-dir", default="outputs/openalex_title_meta/reports")
    parser.add_argument("--external-dir", default="outputs/external")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--doi-cache", default="openalex_doi_features.csv")
    parser.add_argument("--title-cache", default="openalex_title_features.csv")
    parser.add_argument("--skip-title-search", action="store_true")
    parser.add_argument("--min-title-match", type=float, default=0.86)
    parser.add_argument("--per-page", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--mailto", default="codex@example.com")
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

    external_dir = (paths.project_root / args.external_dir).resolve()
    output_dir = (paths.project_root / args.output_dir).resolve()
    report_dir = (paths.project_root / args.report_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    doi_cache = pd.read_csv(external_dir / args.doi_cache)
    title_cache_path = external_dir / args.title_cache
    if args.skip_title_search and title_cache_path.exists():
        title_cache = pd.read_csv(title_cache_path)
    elif args.skip_title_search:
        title_cache = pd.DataFrame()
    else:
        title_cache = fetch_title_search_cache(
            all_rows,
            title_cache_path,
            min_match=args.min_title_match,
            per_page=args.per_page,
            sleep_seconds=args.sleep,
            mailto=args.mailto,
        )

    enriched_all = attach_openalex_features(all_rows, doi_cache, title_cache)
    enriched_all = add_percentile_features(enriched_all)
    enriched_all.to_csv(report_dir / "openalex_enriched_rows.csv", index=False, encoding="utf-8-sig")

    train_enriched = train_df.merge(
        enriched_all.drop(columns=[column for column in [LABEL_COLUMN] if column in enriched_all.columns]),
        on=["source_split", "id", "title", "venue", "year", "authors", "doi", "doi_norm", "first_surname", "has_authors", "title_len", "author_count"],
        how="left",
    )
    test_enriched = test_df.merge(
        enriched_all,
        on=["source_split", "id", "title", "venue", "year", "authors", "doi", "doi_norm", "first_surname", "has_authors", "title_len", "author_count"],
        how="left",
    )

    anchor_oof, anchor_test = load_anchor_scores(paths.project_root)
    train_enriched["ridge_score"] = anchor_oof
    test_enriched["ridge_score"] = anchor_test

    for column, smooth in [("venue", 20.0), ("year", 20.0), ("first_surname", 8.0)]:
        train_values, test_values = add_oof_target_encoding(train_enriched, test_enriched, column, smooth)
        train_enriched[f"{column}_te"] = train_values
        test_enriched[f"{column}_te"] = test_values

    numeric_columns = [
        "ridge_score",
        "openalex_found",
        "title_match_score",
        "title_year_delta",
        "log_cited_by_count",
        "log_referenced_works_count",
        "log_fwci",
        "log_citation_velocity",
        "cited_by_count_global_pct",
        "cited_by_count_venue_pct",
        "cited_by_count_year_pct",
        "cited_by_count_venue_year_pct",
        "citation_velocity_global_pct",
        "citation_velocity_venue_pct",
        "citation_velocity_year_pct",
        "citation_velocity_venue_year_pct",
        "fwci_global_pct",
        "fwci_venue_pct",
        "fwci_year_pct",
        "fwci_venue_year_pct",
        "openalex_year",
        "citation_age",
        "title_len",
        "author_count",
        "has_authors",
        "venue_te",
        "year_te",
        "first_surname_te",
    ]
    categorical_columns = ["venue", "openalex_source"]

    y = train_enriched[LABEL_COLUMN].astype(int).to_numpy()
    y_float = y.astype(float)
    x_train = train_enriched[numeric_columns + categorical_columns]
    x_test = test_enriched[numeric_columns + categorical_columns]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    rows: list[dict[str, object]] = []

    for candidate in make_candidates(numeric_columns, categorical_columns):
        print(f"training {candidate.name}")
        oof_scores = np.clip(cross_val_predict(candidate.model, x_train, y_float, cv=cv), 1.0, 5.0)
        candidate.model.fit(x_train, y_float)
        test_scores = np.clip(candidate.model.predict(x_test), 1.0, 5.0)

        for meta_weight in [1.0, 0.75, 0.5, 0.35, 0.25, 0.15]:
            blended_oof = oof_scores if meta_weight == 1.0 else (1 - meta_weight) * anchor_oof + meta_weight * oof_scores
            blended_test = test_scores if meta_weight == 1.0 else (1 - meta_weight) * anchor_test + meta_weight * test_scores
            thresholds, qwk, mae, train_distribution = tune_thresholds(y, blended_oof, RANDOM_STATE)
            labels = scores_to_labels(blended_test, thresholds)
            submission = pd.DataFrame({"id": sample_df["id"], LABEL_COLUMN: labels})
            name = f"openalex_title_{candidate.name}_w{meta_weight:.2f}"
            submission_path = output_dir / f"{name}_submission.csv"
            submission.to_csv(submission_path, index=False)
            rows.append(
                {
                    "name": name,
                    "base_model": candidate.name,
                    "meta_weight": meta_weight,
                    "oof_qwk": qwk,
                    "mae": mae,
                    "thresholds": ",".join(f"{value:.12f}" for value in thresholds),
                    "train_distribution": train_distribution,
                    "test_distribution": pd.Series(labels).value_counts().sort_index().astype(int).to_dict(),
                    "submission": str(submission_path),
                }
            )

    report = pd.DataFrame(rows).sort_values("oof_qwk", ascending=False)
    report.to_csv(report_dir / "openalex_title_meta_candidates.csv", index=False)

    coverage = enriched_all.groupby(["source_split", "openalex_source"]).size().unstack(fill_value=0)
    coverage.to_csv(report_dir / "openalex_title_coverage.csv")

    summary_lines = [
        "# OpenAlex Title Meta Candidates",
        "",
        "This run expands DOI-only OpenAlex enrichment with title-search matches for rows without DOI.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(),
        "",
        "## Top Candidates",
        "",
        report.head(12).to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `meta_weight=1.0` uses only the OpenAlex meta-model score.",
        "- Smaller `meta_weight` values are light blends with the Ridge 5x5 score anchor used by the `0.62820` submission.",
        "- Citation velocity and venue/year percentile features are included to reduce raw citation-count bias.",
    ]
    (report_dir / "openalex_title_meta_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    best = report.iloc[0]
    best_copy = paths.project_root / "outputs" / "submissions" / "next_openalex_title_meta_submission.csv"
    best_copy.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(best["submission"]).to_csv(best_copy, index=False)
    print(report.head(12).to_string(index=False))
    print(f"best copy: {best_copy}")


if __name__ == "__main__":
    main()
