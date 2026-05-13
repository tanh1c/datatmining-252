"""Multi-source abstract crawler for the ASP paper-classification dataset.

Why multi-source? Each provider has gaps and gives different fields:

- Semantic Scholar: best coverage for CS papers, returns `abstract` + `tldr.text`,
  and supports a batch endpoint (500 ids per call) which is far faster than
  per-paper queries. Many rows in our dataset already have a direct
  `semanticscholar.org/paper/<paper_id>` URL stored in the `doi` column, so we
  can call the batch endpoint with that paper id directly.
- OpenAlex: very high DOI coverage (1810/1813 in our cache), returns
  `abstract_inverted_index` that we reconstruct into plain text.
- Crossref: many Springer / IEEE / ACM papers expose `abstract` in JatsXML.
  Useful as a secondary source when S2/OpenAlex are missing.
- OpenAlex title-search fallback for rows without DOI (already cached, we just
  re-fetch with `abstract_inverted_index`).

The final merged cache picks the first non-empty abstract in priority order.

Caches (resumable, written incrementally):

- outputs/external/s2_abstracts.csv
- outputs/external/openalex_abstracts.csv
- outputs/external/crossref_abstracts.csv
- outputs/external/openalex_title_abstracts.csv
- outputs/external/abstracts_merged.csv

Run:

    python crawl_abstracts.py --skip-title-search   # safe partial run
    python crawl_abstracts.py                       # full run

The crawler is idempotent: every phase reads its own cache, only requests rows
that are not yet cached, and saves the cache periodically. You can stop at any
time with Ctrl+C; rerun the same command to resume.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "outputs" / "external"
LABEL_COLUMN = "Label"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
S2_FIELDS = "title,abstract,tldr,year,venue,externalIds,publicationTypes,fieldsOfStudy,citationCount,referenceCount"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

DEFAULT_USER_AGENT = "asp-paper-classification (codex@example.com)"


# ---------------------------------------------------------------------------
# Helpers shared across phases.
# ---------------------------------------------------------------------------


def normalize_doi(value: object) -> str:
    """Return a clean DOI string `10.xxx/yyy` or empty if not a real DOI.

    Rows whose `doi` column is actually a Semantic Scholar URL are returned as
    empty here; those are handled by `extract_s2_paper_id` below.
    """

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "semanticscholar.org" in text.lower():
        return ""
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    return text if text.lower().startswith("10.") else ""


def extract_s2_paper_id(value: object) -> str:
    """Return the Semantic Scholar paper id when the `doi` column is a S2 URL."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    match = re.search(r"semanticscholar\.org/paper/([0-9a-fA-F]{40})", text)
    return match.group(1).lower() if match else ""


def reconstruct_inverted_abstract(inverted: dict | None) -> str:
    """Rebuild plain text from OpenAlex `abstract_inverted_index`."""

    if not inverted:
        return ""
    if not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        if not isinstance(idxs, list):
            continue
        for idx in idxs:
            if isinstance(idx, int):
                positions.append((idx, str(word)))
    if not positions:
        return ""
    positions.sort()
    return " ".join(word for _, word in positions)


def strip_xml_tags(text: str) -> str:
    """Best-effort cleanup of Crossref / Jats abstracts (quick and pragmatic)."""

    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunked(iterable: Iterable, size: int) -> list[list]:
    chunk: list = []
    chunks: list[list] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def safe_write_csv(df: pd.DataFrame, target: Path, retries: int = 5, sleep_seconds: float = 0.5) -> None:
    """Atomic CSV write with retries to survive transient Windows locks (e.g. Excel)."""

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            tmp.replace(target)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(sleep_seconds * (attempt + 1))
    if last_exc is not None:
        # Could not replace the target. Keep the .tmp file so the data is not lost
        # and surface the issue with a clearer instruction.
        raise PermissionError(
            f"Could not replace {target} after {retries} retries. The file is likely "
            f"open in another program (e.g. Excel). Data was saved to {tmp} instead."
        ) from last_exc


def append_cache(cache_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=[c for c in merged.columns if c.startswith("lookup")], keep="last")
        safe_write_csv(merged, cache_path)
    else:
        safe_write_csv(new_df, cache_path)


# ---------------------------------------------------------------------------
# Build the unified ID frame.
# ---------------------------------------------------------------------------


@dataclass
class IdFrame:
    rows: pd.DataFrame  # source_split, id, title, year, doi_norm, s2_paper_id


def build_id_frame() -> IdFrame:
    train = pd.read_csv(DATA_DIR / "train.csv").assign(source_split="train")
    public = pd.read_csv(DATA_DIR / "public_test.csv").assign(source_split="public_test")
    private = pd.read_csv(DATA_DIR / "private_test.csv").assign(source_split="private_test")
    keep_cols = ["source_split", "id", "title", "year", "doi"]
    rows = pd.concat([train[keep_cols + ([LABEL_COLUMN] if LABEL_COLUMN in train.columns else [])],
                      public[keep_cols],
                      private[keep_cols]], ignore_index=True)
    rows["doi_norm"] = rows["doi"].map(normalize_doi)
    rows["s2_paper_id"] = rows["doi"].map(extract_s2_paper_id)
    return IdFrame(rows=rows)


# ---------------------------------------------------------------------------
# Phase 1: Semantic Scholar batch (DOI + paper id).
# ---------------------------------------------------------------------------


def phase_semantic_scholar(
    id_frame: IdFrame,
    cache_path: Path,
    batch_size: int,
    sleep_seconds: float,
    api_key: str | None,
) -> pd.DataFrame:
    """Crawl S2 abstracts in chunks. lookup_id has form `DOI:...` or `S2:...`."""

    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["lookup_id"])
    cached_ids = set(cache["lookup_id"].astype(str)) if not cache.empty else set()

    queue: list[tuple[str, str]] = []  # list of (lookup_id, raw_id_for_request)
    seen: set[str] = set()
    for _, row in id_frame.rows.iterrows():
        if row["doi_norm"]:
            lookup = f"DOI:{row['doi_norm']}"
            if lookup not in cached_ids and lookup not in seen:
                queue.append((lookup, row["doi_norm"]))
                seen.add(lookup)
        elif row["s2_paper_id"]:
            lookup = f"S2:{row['s2_paper_id']}"
            if lookup not in cached_ids and lookup not in seen:
                queue.append((lookup, row["s2_paper_id"]))
                seen.add(lookup)

    print(f"[S2] cached={len(cached_ids)}, to fetch={len(queue)} (batch_size={batch_size})")

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key

    session = requests.Session()
    pending_rows: list[dict] = []
    for chunk in chunked(queue, batch_size):
        ids_for_request = []
        for lookup_id, raw_id in chunk:
            if lookup_id.startswith("DOI:"):
                ids_for_request.append(f"DOI:{raw_id}")
            else:
                ids_for_request.append(raw_id)
        try:
            response = session.post(
                S2_BATCH_URL,
                params={"fields": S2_FIELDS},
                json={"ids": ids_for_request},
                headers=headers,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[S2] batch error: {exc!r}")
            time.sleep(5.0)
            continue

        if response.status_code == 429:
            print("[S2] 429 rate limited, sleeping 30s")
            time.sleep(30.0)
            continue
        if response.status_code != 200:
            print(f"[S2] HTTP {response.status_code}: {response.text[:200]}")
            time.sleep(5.0)
            continue

        try:
            payload = response.json()
        except json.JSONDecodeError:
            print("[S2] invalid JSON, skipping chunk")
            time.sleep(2.0)
            continue

        for (lookup_id, raw_id), record in zip(chunk, payload):
            if record is None:
                pending_rows.append({"lookup_id": lookup_id, "found": False, "abstract": "", "tldr": "",
                                     "s2_paper_id": "", "title": "", "year": np.nan, "venue": "",
                                     "publication_types": "", "fields_of_study": "",
                                     "citation_count": np.nan, "reference_count": np.nan,
                                     "external_ids_json": ""})
                continue

            tldr_text = ""
            if isinstance(record.get("tldr"), dict):
                tldr_text = str(record["tldr"].get("text", "") or "")
            external_ids = record.get("externalIds") or {}
            pending_rows.append({
                "lookup_id": lookup_id,
                "found": True,
                "abstract": str(record.get("abstract") or ""),
                "tldr": tldr_text,
                "s2_paper_id": str(record.get("paperId") or ""),
                "title": str(record.get("title") or ""),
                "year": record.get("year") if record.get("year") is not None else np.nan,
                "venue": str(record.get("venue") or ""),
                "publication_types": "|".join(record.get("publicationTypes") or []),
                "fields_of_study": "|".join(record.get("fieldsOfStudy") or []),
                "citation_count": record.get("citationCount", np.nan),
                "reference_count": record.get("referenceCount", np.nan),
                "external_ids_json": json.dumps(external_ids, ensure_ascii=False),
            })

        if len(pending_rows) >= 200:
            append_cache(cache_path, pending_rows)
            print(f"[S2] flushed {len(pending_rows)} rows -> {cache_path.name}")
            pending_rows = []
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if pending_rows:
        append_cache(cache_path, pending_rows)

    return pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 2: OpenAlex DOI abstracts.
# ---------------------------------------------------------------------------


def phase_openalex_doi(
    id_frame: IdFrame,
    cache_path: Path,
    sleep_seconds: float,
    mailto: str,
) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["lookup_id"])
    cached_ids = set(cache["lookup_id"].astype(str)) if not cache.empty else set()

    queue: list[str] = []
    seen: set[str] = set()
    for _, row in id_frame.rows.iterrows():
        doi = row["doi_norm"]
        if not doi:
            continue
        lookup = f"DOI:{doi}"
        if lookup in cached_ids or lookup in seen:
            continue
        queue.append(doi)
        seen.add(lookup)

    print(f"[OpenAlex DOI] cached={len(cached_ids)}, to fetch={len(queue)}")
    session = requests.Session()
    pending_rows: list[dict] = []
    select = ",".join([
        "id", "doi", "display_name", "publication_year",
        "abstract_inverted_index", "cited_by_count",
        "referenced_works_count", "fwci", "is_retracted",
        "type", "type_crossref",
    ])

    for idx, doi in enumerate(queue, start=1):
        url = f"{OPENALEX_WORKS_URL}/doi:{doi}"
        try:
            response = session.get(url, params={"select": select, "mailto": mailto}, timeout=30)
        except Exception as exc:  # noqa: BLE001
            print(f"[OpenAlex DOI] error for {doi}: {exc!r}")
            time.sleep(2.0)
            continue
        if response.status_code == 429:
            print("[OpenAlex DOI] 429 rate limited, sleeping 30s")
            time.sleep(30.0)
            continue
        record = {"lookup_id": f"DOI:{doi}", "doi_norm": doi, "found": False, "abstract": "",
                  "openalex_id": "", "title": "", "year": np.nan,
                  "cited_by_count": np.nan, "referenced_works_count": np.nan,
                  "fwci": np.nan, "is_retracted": False,
                  "type": "", "type_crossref": ""}
        if response.status_code == 200:
            try:
                payload = response.json()
            except json.JSONDecodeError:
                payload = None
            if payload:
                record["found"] = True
                record["openalex_id"] = str(payload.get("id") or "")
                record["title"] = str(payload.get("display_name") or "")
                record["year"] = payload.get("publication_year")
                record["cited_by_count"] = payload.get("cited_by_count")
                record["referenced_works_count"] = payload.get("referenced_works_count")
                record["fwci"] = payload.get("fwci")
                record["is_retracted"] = bool(payload.get("is_retracted") or False)
                record["type"] = str(payload.get("type") or "")
                record["type_crossref"] = str(payload.get("type_crossref") or "")
                record["abstract"] = reconstruct_inverted_abstract(payload.get("abstract_inverted_index"))
        elif response.status_code != 404:
            print(f"[OpenAlex DOI] HTTP {response.status_code} for {doi}: {response.text[:120]}")

        pending_rows.append(record)
        if idx % 100 == 0 or idx == len(queue):
            append_cache(cache_path, pending_rows)
            print(f"[OpenAlex DOI] {idx}/{len(queue)} flushed (found in last batch={sum(1 for r in pending_rows if r['found'])})")
            pending_rows = []
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if pending_rows:
        append_cache(cache_path, pending_rows)

    return pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 3: Crossref abstracts.
# ---------------------------------------------------------------------------


def phase_crossref(
    id_frame: IdFrame,
    cache_path: Path,
    sleep_seconds: float,
    mailto: str,
) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["lookup_id"])
    cached_ids = set(cache["lookup_id"].astype(str)) if not cache.empty else set()

    queue: list[str] = []
    seen: set[str] = set()
    for _, row in id_frame.rows.iterrows():
        doi = row["doi_norm"]
        if not doi:
            continue
        lookup = f"DOI:{doi}"
        if lookup in cached_ids or lookup in seen:
            continue
        queue.append(doi)
        seen.add(lookup)

    print(f"[Crossref] cached={len(cached_ids)}, to fetch={len(queue)}")
    headers = {"User-Agent": f"{DEFAULT_USER_AGENT} (mailto:{mailto})"}
    session = requests.Session()
    pending_rows: list[dict] = []

    for idx, doi in enumerate(queue, start=1):
        url = f"https://api.crossref.org/works/{doi}"
        try:
            response = session.get(url, headers=headers, timeout=30)
        except Exception as exc:  # noqa: BLE001
            print(f"[Crossref] error for {doi}: {exc!r}")
            time.sleep(2.0)
            continue
        record = {"lookup_id": f"DOI:{doi}", "doi_norm": doi, "found": False,
                  "abstract": "", "publisher": "", "type": "",
                  "container_title": "", "subjects": ""}
        if response.status_code == 200:
            try:
                payload = response.json().get("message", {})
            except json.JSONDecodeError:
                payload = {}
            if payload:
                record["found"] = True
                record["publisher"] = str(payload.get("publisher") or "")
                record["type"] = str(payload.get("type") or "")
                container = payload.get("container-title") or []
                if isinstance(container, list) and container:
                    record["container_title"] = str(container[0])
                subjects = payload.get("subject") or []
                if isinstance(subjects, list):
                    record["subjects"] = "|".join(str(s) for s in subjects)
                record["abstract"] = strip_xml_tags(str(payload.get("abstract") or ""))
        elif response.status_code != 404:
            print(f"[Crossref] HTTP {response.status_code} for {doi}: {response.text[:120]}")

        pending_rows.append(record)
        if idx % 100 == 0 or idx == len(queue):
            append_cache(cache_path, pending_rows)
            print(f"[Crossref] {idx}/{len(queue)} flushed")
            pending_rows = []
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if pending_rows:
        append_cache(cache_path, pending_rows)

    return pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 4: OpenAlex title-search abstracts (rows without DOI).
# ---------------------------------------------------------------------------


def phase_openalex_title(
    id_frame: IdFrame,
    cache_path: Path,
    sleep_seconds: float,
    mailto: str,
    min_match: float,
    per_page: int,
) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["lookup_id"])
    cached_ids = set(cache["lookup_id"].astype(str)) if not cache.empty else set()

    rows_without_doi = id_frame.rows[id_frame.rows["doi_norm"].eq("")]
    queue: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, row in rows_without_doi.iterrows():
        lookup = f"TITLE:{row['source_split']}:{row['id']}"
        if lookup in cached_ids or lookup in seen:
            continue
        queue.append((lookup, str(row["title"]) if not pd.isna(row["title"]) else ""))
        seen.add(lookup)

    print(f"[OpenAlex title] cached={len(cached_ids)}, to fetch={len(queue)}")
    session = requests.Session()
    pending_rows: list[dict] = []
    select = ",".join([
        "id", "doi", "display_name", "publication_year",
        "abstract_inverted_index", "cited_by_count",
        "referenced_works_count", "fwci", "is_retracted",
    ])

    def title_similarity(a: str, b: str) -> float:
        from difflib import SequenceMatcher
        a = re.sub(r"[^a-z0-9 ]", " ", a.lower()).strip()
        b = re.sub(r"[^a-z0-9 ]", " ", b.lower()).strip()
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    for idx, (lookup_id, source_title) in enumerate(queue, start=1):
        record = {"lookup_id": lookup_id, "source_title": source_title, "found": False,
                  "abstract": "", "match_score": 0.0,
                  "openalex_id": "", "openalex_title": "", "year": np.nan,
                  "cited_by_count": np.nan, "referenced_works_count": np.nan,
                  "fwci": np.nan, "is_retracted": False, "doi": ""}
        cleaned_query = re.sub(r"[^a-zA-Z0-9 ]", " ", source_title or "").strip()
        if cleaned_query:
            try:
                response = session.get(
                    OPENALEX_WORKS_URL,
                    params={"search": cleaned_query, "per-page": per_page,
                            "select": select, "mailto": mailto},
                    timeout=30,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[OpenAlex title] error idx={idx}: {exc!r}")
                response = None
            if response is not None and response.status_code == 200:
                try:
                    results = response.json().get("results", [])
                except json.JSONDecodeError:
                    results = []
                best_score = -1.0
                for item in results:
                    candidate_title = str(item.get("display_name") or "")
                    similarity = title_similarity(source_title, candidate_title)
                    if similarity > best_score:
                        best_score = similarity
                        record["match_score"] = similarity
                        record["openalex_id"] = str(item.get("id") or "")
                        record["doi"] = str(item.get("doi") or "")
                        record["openalex_title"] = candidate_title
                        record["year"] = item.get("publication_year")
                        record["cited_by_count"] = item.get("cited_by_count")
                        record["referenced_works_count"] = item.get("referenced_works_count")
                        record["fwci"] = item.get("fwci")
                        record["is_retracted"] = bool(item.get("is_retracted") or False)
                        record["abstract"] = reconstruct_inverted_abstract(item.get("abstract_inverted_index"))
                record["found"] = bool(record["match_score"] >= min_match)

        pending_rows.append(record)
        if idx % 50 == 0 or idx == len(queue):
            append_cache(cache_path, pending_rows)
            print(f"[OpenAlex title] {idx}/{len(queue)} flushed")
            pending_rows = []
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if pending_rows:
        append_cache(cache_path, pending_rows)

    return pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 5: Merge all sources.
# ---------------------------------------------------------------------------


def phase_merge(id_frame: IdFrame,
                s2_cache: Path,
                openalex_cache: Path,
                crossref_cache: Path,
                openalex_title_cache: Path,
                merged_path: Path) -> pd.DataFrame:
    rows = id_frame.rows.copy()
    rows["s2_lookup_id"] = np.where(
        rows["doi_norm"].ne(""),
        "DOI:" + rows["doi_norm"],
        np.where(rows["s2_paper_id"].ne(""), "S2:" + rows["s2_paper_id"], ""),
    )
    rows["doi_lookup_id"] = np.where(rows["doi_norm"].ne(""), "DOI:" + rows["doi_norm"], "")
    rows["title_lookup_id"] = np.where(
        rows["doi_norm"].eq(""),
        "TITLE:" + rows["source_split"] + ":" + rows["id"].astype(str),
        "",
    )

    def load(path: Path, prefix: str) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = df.rename(columns={c: (c if c == "lookup_id" else f"{prefix}_{c}") for c in df.columns if c != "lookup_id"})
        return df

    s2 = load(s2_cache, "s2")
    oa = load(openalex_cache, "oa")
    cr = load(crossref_cache, "cr")
    oat = load(openalex_title_cache, "oat")

    if not s2.empty:
        rows = rows.merge(s2.rename(columns={"lookup_id": "s2_lookup_id"}), on="s2_lookup_id", how="left")
    if not oa.empty:
        rows = rows.merge(oa.rename(columns={"lookup_id": "doi_lookup_id"}), on="doi_lookup_id", how="left")
    if not cr.empty:
        rows = rows.merge(cr.rename(columns={"lookup_id": "doi_lookup_id"}), on="doi_lookup_id", how="left", suffixes=("", "_cr"))
    if not oat.empty:
        rows = rows.merge(oat.rename(columns={"lookup_id": "title_lookup_id"}), on="title_lookup_id", how="left")

    def pick_first_nonempty(values: list[object]) -> str:
        for v in values:
            if isinstance(v, float) and np.isnan(v):
                continue
            if v is None:
                continue
            text = str(v).strip()
            if text and text.lower() != "nan":
                return text
        return ""

    abstract_columns = []
    for prefix, label in [("s2", "abstract"), ("s2", "tldr"), ("oa", "abstract"),
                           ("cr", "abstract"), ("oat", "abstract")]:
        col = f"{prefix}_{label}"
        if col in rows.columns:
            abstract_columns.append(col)

    rows["abstract"] = rows[abstract_columns].apply(
        lambda r: pick_first_nonempty(r.tolist()), axis=1
    )
    rows["abstract_source"] = rows[abstract_columns].apply(
        lambda r: next((col for col, value in zip(abstract_columns, r.tolist())
                        if isinstance(value, str) and value.strip() and str(value).lower() != "nan"),
                       ""), axis=1
    )
    rows["abstract_len"] = rows["abstract"].fillna("").map(len)
    rows["has_abstract"] = rows["abstract_len"] > 30  # discard junk like single dot

    keep_columns = [
        "source_split", "id", "title", "year", "doi", "doi_norm", "s2_paper_id",
        "abstract", "abstract_source", "abstract_len", "has_abstract",
    ]
    final = rows[keep_columns + [c for c in rows.columns if c.startswith(("s2_", "oa_", "cr_", "oat_")) and c not in keep_columns]]
    safe_write_csv(final, merged_path)

    return final


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2-batch-size", type=int, default=200)
    parser.add_argument("--s2-sleep", type=float, default=0.5)
    parser.add_argument("--s2-key", default=None, help="Optional S2 API key (x-api-key header).")
    parser.add_argument("--openalex-sleep", type=float, default=0.05)
    parser.add_argument("--crossref-sleep", type=float, default=0.05)
    parser.add_argument("--mailto", default="codex@example.com")
    parser.add_argument("--openalex-title-min-match", type=float, default=0.84)
    parser.add_argument("--openalex-title-per-page", type=int, default=5)
    parser.add_argument("--skip-s2", action="store_true")
    parser.add_argument("--skip-openalex-doi", action="store_true")
    parser.add_argument("--skip-crossref", action="store_true")
    parser.add_argument("--skip-title-search", action="store_true")
    parser.add_argument("--only-merge", action="store_true",
                        help="Skip all crawl phases, just rebuild abstracts_merged.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    id_frame = build_id_frame()

    print(f"Total rows in train+public+private: {len(id_frame.rows)}")
    print(f"Rows with real DOI: {id_frame.rows['doi_norm'].ne('').sum()}")
    print(f"Rows with direct S2 paper id: {id_frame.rows['s2_paper_id'].ne('').sum()}")
    print(f"Rows with neither (need title search): {((id_frame.rows['doi_norm'].eq('')) & (id_frame.rows['s2_paper_id'].eq(''))).sum()}")
    print()

    s2_cache = EXTERNAL_DIR / "s2_abstracts.csv"
    openalex_cache = EXTERNAL_DIR / "openalex_abstracts.csv"
    crossref_cache = EXTERNAL_DIR / "crossref_abstracts.csv"
    openalex_title_cache = EXTERNAL_DIR / "openalex_title_abstracts.csv"
    merged_path = EXTERNAL_DIR / "abstracts_merged_v2.csv"

    if not args.only_merge:
        if not args.skip_s2:
            phase_semantic_scholar(id_frame, s2_cache, args.s2_batch_size, args.s2_sleep, args.s2_key)
        if not args.skip_openalex_doi:
            phase_openalex_doi(id_frame, openalex_cache, args.openalex_sleep, args.mailto)
        if not args.skip_crossref:
            phase_crossref(id_frame, crossref_cache, args.crossref_sleep, args.mailto)
        if not args.skip_title_search:
            phase_openalex_title(
                id_frame, openalex_title_cache,
                sleep_seconds=args.openalex_sleep,
                mailto=args.mailto,
                min_match=args.openalex_title_min_match,
                per_page=args.openalex_title_per_page,
            )

    merged = phase_merge(id_frame, s2_cache, openalex_cache, crossref_cache, openalex_title_cache, merged_path)

    print()
    print("=== Coverage ===")
    print(merged.groupby("source_split")["has_abstract"].agg(["count", "sum", "mean"]).round(3))
    print()
    print("=== Source breakdown ===")
    print(merged.groupby(["source_split", "abstract_source"]).size().unstack(fill_value=0))
    print()
    print(f"Merged cache written to {merged_path}")


if __name__ == "__main__":
    main()
