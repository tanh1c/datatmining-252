"""Phase-2 abstract crawler — fill in the 195 rows that crawl_abstracts.py missed.

Three new sources:

1. **arXiv API** (`export.arxiv.org/api/query`): many CAV/LICS/KR papers also
   appear on arXiv. Search by title.
2. **Semantic Scholar `search/match` endpoint**: a different S2 entrypoint than
   the batch endpoint we already hit. Sometimes succeeds where batch returned
   empty abstract.
3. **OpenAlex search with year filter**: tighter title-search than the
   unfiltered one we ran in phase 1; the year constraint disambiguates papers
   with similar titles.

Outputs (resume-friendly per-source caches):

- outputs/external/arxiv_abstracts.csv
- outputs/external/s2_match_abstracts.csv
- outputs/external/openalex_year_abstracts.csv
- outputs/external/abstracts_merged_v3.csv  <- the new merged cache

Run:

    python crawl_abstracts_v2.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "outputs" / "external"
LABEL_COLUMN = "Label"
DEFAULT_USER_AGENT = "asp-paper-classification (codex@example.com)"

ARXIV_URL = "http://export.arxiv.org/api/query"
S2_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
OPENALEX_URL = "https://api.openalex.org/works"


def normalize_title(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("&apos;", "'").replace("&amp;", "&")
    text = re.sub(r"[^a-z0-9\- ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    a, b = normalize_title(a), normalize_title(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def reconstruct_inverted(inverted) -> str:
    if not isinstance(inverted, dict) or not inverted:
        return ""
    positions = []
    for word, idxs in inverted.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            if isinstance(i, int):
                positions.append((i, str(word)))
    positions.sort()
    return " ".join(w for _, w in positions)


def safe_write_csv(df: pd.DataFrame, path: Path, retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    last = None
    for k in range(retries):
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.5 * (k + 1))
    raise PermissionError(f"could not replace {path}") from last


def append_cache(cache_path: Path, rows: list[dict], key_cols: list[str]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
        safe_write_csv(merged, cache_path)
    else:
        safe_write_csv(new_df, cache_path)


def load_target_rows() -> pd.DataFrame:
    """Return all train/public/private rows that are missing in v2 merged cache."""

    train = pd.read_csv(DATA_DIR / "train.csv").assign(source_split="train")
    public = pd.read_csv(DATA_DIR / "public_test.csv").assign(source_split="public_test")
    private = pd.read_csv(DATA_DIR / "private_test.csv").assign(source_split="private_test")
    cols = ["source_split", "id", "title", "year"]
    rows = pd.concat([train[cols], public[cols], private[cols]], ignore_index=True)

    v2 = pd.read_csv(EXTERNAL_DIR / "abstracts_merged_v2.csv")
    have = v2.loc[v2["has_abstract"] == True, ["source_split", "id"]]
    have_keys = set(zip(have["source_split"], have["id"].astype(int)))

    rows["have_abstract"] = rows.apply(
        lambda r: (r["source_split"], int(r["id"])) in have_keys, axis=1
    )
    missing = rows[~rows["have_abstract"]].reset_index(drop=True)
    print(f"v2 cache has {len(have_keys)} rows with abstract; "
          f"{len(missing)} rows still missing (target queue for v2 crawler)")
    return missing


# ---------------------------------------------------------------------------
# arXiv source
# ---------------------------------------------------------------------------


def phase_arxiv(target: pd.DataFrame, cache_path: Path,
                sleep_seconds: float, min_match: float) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["source_split", "id"])
    cached = set(zip(cache.get("source_split", pd.Series([])).astype(str),
                     cache.get("id", pd.Series([])).astype(int))) if not cache.empty else set()

    queue = [(r["source_split"], int(r["id"]), str(r["title"]) if pd.notna(r["title"]) else "",
              int(r["year"]) if pd.notna(r["year"]) else 0)
             for _, r in target.iterrows()
             if (r["source_split"], int(r["id"])) not in cached]
    print(f"[arXiv] cached={len(cached)}, to fetch={len(queue)}")

    pending: list[dict] = []
    session = requests.Session()
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for idx, (split, pid, title, year) in enumerate(queue, start=1):
        record = {"source_split": split, "id": pid, "source_title": title,
                  "found": False, "abstract": "", "match_score": 0.0,
                  "arxiv_id": "", "arxiv_title": "", "arxiv_year": np.nan,
                  "error": ""}
        cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", title).strip()
        if cleaned:
            try:
                response = session.get(
                    ARXIV_URL,
                    params={"search_query": f"ti:\"{cleaned}\"",
                            "max_results": 5, "sortBy": "relevance"},
                    headers=headers, timeout=30,
                )
                if response.status_code == 200:
                    root = ET.fromstring(response.text)
                    best_score = -1.0
                    for entry in root.findall("atom:entry", ns):
                        cand_title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                        cand_summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                        cand_id_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
                        cand_published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
                        cand_year = int(cand_published[:4]) if cand_published[:4].isdigit() else 0
                        sim = title_similarity(title, cand_title)
                        # year bonus: papers published <= source year + 1
                        year_bonus = 0.05 if 0 < cand_year <= year + 1 else 0.0
                        rank = sim + year_bonus
                        if rank > best_score:
                            best_score = rank
                            record["match_score"] = sim
                            record["arxiv_id"] = cand_id_url
                            record["arxiv_title"] = cand_title
                            record["arxiv_year"] = cand_year if cand_year else np.nan
                            record["abstract"] = re.sub(r"\s+", " ", cand_summary)
                    record["found"] = bool(record["match_score"] >= min_match)
                else:
                    record["error"] = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001
                record["error"] = repr(exc)
        pending.append(record)

        if idx % 20 == 0 or idx == len(queue):
            append_cache(cache_path, pending, key_cols=["source_split", "id"])
            found_so_far = sum(1 for r in pending if r["found"])
            print(f"[arXiv] {idx}/{len(queue)} flushed (new found in last batch={found_so_far})")
            pending = []
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if pending:
        append_cache(cache_path, pending, key_cols=["source_split", "id"])
    return pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Semantic Scholar match endpoint
# ---------------------------------------------------------------------------


def phase_s2_match(target: pd.DataFrame, cache_path: Path,
                   sleep_seconds: float, min_match: float) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["source_split", "id"])
    cached = set(zip(cache.get("source_split", pd.Series([])).astype(str),
                     cache.get("id", pd.Series([])).astype(int))) if not cache.empty else set()

    queue = [(r["source_split"], int(r["id"]), str(r["title"]) if pd.notna(r["title"]) else "")
             for _, r in target.iterrows()
             if (r["source_split"], int(r["id"])) not in cached]
    print(f"[S2 match] cached={len(cached)}, to fetch={len(queue)}")

    fields = "title,abstract,tldr,year,venue,externalIds"
    pending: list[dict] = []
    session = requests.Session()
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    for idx, (split, pid, title) in enumerate(queue, start=1):
        record = {"source_split": split, "id": pid, "source_title": title,
                  "found": False, "abstract": "", "tldr": "",
                  "match_score": 0.0, "s2_paper_id": "", "s2_title": "",
                  "error": ""}
        cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", title).strip()
        if cleaned:
            try:
                response = session.get(
                    S2_MATCH_URL,
                    params={"query": cleaned, "fields": fields},
                    headers=headers, timeout=30,
                )
                if response.status_code == 200:
                    payload = response.json()
                    data = payload.get("data") if isinstance(payload, dict) else None
                    item = data[0] if isinstance(data, list) and data else None
                    if item:
                        cand_title = str(item.get("title") or "")
                        sim = title_similarity(title, cand_title)
                        record["match_score"] = sim
                        record["s2_paper_id"] = str(item.get("paperId") or "")
                        record["s2_title"] = cand_title
                        record["abstract"] = str(item.get("abstract") or "")
                        if isinstance(item.get("tldr"), dict):
                            record["tldr"] = str(item["tldr"].get("text") or "")
                        record["found"] = sim >= min_match and bool(record["abstract"] or record["tldr"])
                elif response.status_code == 429:
                    print("[S2 match] 429 rate limited, sleeping 30s")
                    time.sleep(30.0)
                else:
                    record["error"] = f"HTTP {response.status_code}: {response.text[:120]}"
            except Exception as exc:  # noqa: BLE001
                record["error"] = repr(exc)
        pending.append(record)

        if idx % 25 == 0 or idx == len(queue):
            append_cache(cache_path, pending, key_cols=["source_split", "id"])
            found_so_far = sum(1 for r in pending if r["found"])
            print(f"[S2 match] {idx}/{len(queue)} flushed (new found in last batch={found_so_far})")
            pending = []
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if pending:
        append_cache(cache_path, pending, key_cols=["source_split", "id"])
    return pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# OpenAlex search with year filter
# ---------------------------------------------------------------------------


def phase_openalex_year(target: pd.DataFrame, cache_path: Path,
                        sleep_seconds: float, mailto: str,
                        min_match: float, per_page: int) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["source_split", "id"])
    cached = set(zip(cache.get("source_split", pd.Series([])).astype(str),
                     cache.get("id", pd.Series([])).astype(int))) if not cache.empty else set()

    queue = [(r["source_split"], int(r["id"]),
              str(r["title"]) if pd.notna(r["title"]) else "",
              int(r["year"]) if pd.notna(r["year"]) else 0)
             for _, r in target.iterrows()
             if (r["source_split"], int(r["id"])) not in cached]
    print(f"[OpenAlex year] cached={len(cached)}, to fetch={len(queue)}")

    select = ",".join(["id", "doi", "display_name", "publication_year",
                       "abstract_inverted_index", "cited_by_count",
                       "referenced_works_count", "fwci"])
    pending: list[dict] = []
    session = requests.Session()
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    for idx, (split, pid, title, year) in enumerate(queue, start=1):
        record = {"source_split": split, "id": pid, "source_title": title,
                  "found": False, "abstract": "", "match_score": 0.0,
                  "openalex_id": "", "openalex_title": "", "openalex_year": np.nan,
                  "error": ""}
        cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", title).strip()
        if cleaned and year:
            try:
                params = {
                    "search": cleaned, "per-page": per_page,
                    "select": select, "mailto": mailto,
                    "filter": f"publication_year:{year - 1}|{year}|{year + 1}",
                }
                response = session.get(OPENALEX_URL, params=params,
                                       headers=headers, timeout=30)
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    best = -1.0
                    for item in results:
                        cand_title = str(item.get("display_name") or "")
                        sim = title_similarity(title, cand_title)
                        if sim > best:
                            best = sim
                            record["match_score"] = sim
                            record["openalex_id"] = str(item.get("id") or "")
                            record["openalex_title"] = cand_title
                            record["openalex_year"] = item.get("publication_year") or np.nan
                            record["abstract"] = reconstruct_inverted(
                                item.get("abstract_inverted_index"))
                    record["found"] = bool(record["match_score"] >= min_match
                                            and record["abstract"])
                else:
                    record["error"] = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001
                record["error"] = repr(exc)
        pending.append(record)

        if idx % 30 == 0 or idx == len(queue):
            append_cache(cache_path, pending, key_cols=["source_split", "id"])
            found_so_far = sum(1 for r in pending if r["found"])
            print(f"[OpenAlex year] {idx}/{len(queue)} flushed (new found in last batch={found_so_far})")
            pending = []
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if pending:
        append_cache(cache_path, pending, key_cols=["source_split", "id"])
    return pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Merge with v2 to produce v3
# ---------------------------------------------------------------------------


def merge_v3() -> pd.DataFrame:
    v2 = pd.read_csv(EXTERNAL_DIR / "abstracts_merged_v2.csv")
    arxiv = pd.read_csv(EXTERNAL_DIR / "arxiv_abstracts.csv") if (EXTERNAL_DIR / "arxiv_abstracts.csv").exists() else pd.DataFrame()
    s2m = pd.read_csv(EXTERNAL_DIR / "s2_match_abstracts.csv") if (EXTERNAL_DIR / "s2_match_abstracts.csv").exists() else pd.DataFrame()
    oay = pd.read_csv(EXTERNAL_DIR / "openalex_year_abstracts.csv") if (EXTERNAL_DIR / "openalex_year_abstracts.csv").exists() else pd.DataFrame()

    out = v2.copy()
    out["abstract"] = out["abstract"].fillna("")
    out["abstract_source"] = out["abstract_source"].fillna("")
    out["has_abstract"] = out["has_abstract"].fillna(False).astype(bool)

    def filled_abstract(row, df, source_label):
        if row["has_abstract"]:
            return row["abstract"], row["abstract_source"], True
        match = df[(df["source_split"] == row["source_split"]) & (df["id"] == row["id"])]
        if match.empty:
            return row["abstract"], row["abstract_source"], False
        m = match.iloc[0]
        ab = str(m.get("abstract") or "").strip()
        if len(ab) > 30:
            return ab, source_label, True
        # try tldr field as fallback for s2_match
        if "tldr" in match.columns:
            tldr = str(m.get("tldr") or "").strip()
            if len(tldr) > 30:
                return tldr, source_label.replace("_abstract", "_tldr"), True
        return row["abstract"], row["abstract_source"], False

    # Priority order: arxiv -> s2_match -> openalex_year
    for df, label in [(arxiv, "arxiv_abstract"), (s2m, "s2_match_abstract"),
                      (oay, "openalex_year_abstract")]:
        if df.empty:
            continue
        rows = out.apply(lambda r: filled_abstract(r, df, label), axis=1)
        out["abstract"] = rows.map(lambda x: x[0])
        out["abstract_source"] = rows.map(lambda x: x[1])
        out["has_abstract"] = rows.map(lambda x: x[2])

    out["abstract_len"] = out["abstract"].fillna("").str.len()
    safe_write_csv(out, EXTERNAL_DIR / "abstracts_merged_v3.csv")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-arxiv", action="store_true")
    p.add_argument("--skip-s2-match", action="store_true")
    p.add_argument("--skip-openalex-year", action="store_true")
    p.add_argument("--only-merge", action="store_true")
    p.add_argument("--arxiv-sleep", type=float, default=3.5,
                   help="arXiv API recommends >=3s between requests")
    p.add_argument("--s2-sleep", type=float, default=1.5)
    p.add_argument("--openalex-sleep", type=float, default=0.05)
    p.add_argument("--mailto", default="codex@example.com")
    p.add_argument("--min-match", type=float, default=0.84)
    p.add_argument("--openalex-per-page", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    target = load_target_rows()

    if not args.only_merge:
        if not args.skip_arxiv:
            phase_arxiv(target, EXTERNAL_DIR / "arxiv_abstracts.csv",
                        sleep_seconds=args.arxiv_sleep, min_match=args.min_match)
        if not args.skip_s2_match:
            phase_s2_match(target, EXTERNAL_DIR / "s2_match_abstracts.csv",
                           sleep_seconds=args.s2_sleep, min_match=args.min_match)
        if not args.skip_openalex_year:
            phase_openalex_year(target, EXTERNAL_DIR / "openalex_year_abstracts.csv",
                                sleep_seconds=args.openalex_sleep, mailto=args.mailto,
                                min_match=args.min_match,
                                per_page=args.openalex_per_page)

    merged = merge_v3()
    print()
    print("=== abstracts_merged_v3 coverage ===")
    print(merged.groupby("source_split")["has_abstract"].agg(["count", "sum", "mean"]).round(3))
    print()
    print("=== source breakdown ===")
    print(merged.groupby(["source_split", "abstract_source"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
