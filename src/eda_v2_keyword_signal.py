"""EDA v2 — Confirm the hypothesis that Label 1-5 measures ASP / AI-symbolic
relevance, and audit cross-split risks (duplicate title, author leakage,
missing-author bias) before we move to SPECTER2/abstract enrichment.

Outputs:

- outputs/eda_v2/keyword_label_table.csv
- outputs/eda_v2/keyword_venue_label_table.csv
- outputs/eda_v2/missingness_label.csv
- outputs/eda_v2/duplicate_title_cross_split.csv
- outputs/eda_v2/first_author_leakage.csv
- outputs/eda_v2/test_split_keyword_coverage.csv
- outputs/eda_v2/eda_v2_report.md

This is read-only on data/raw/* and writes only into outputs/eda_v2/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "eda_v2"
LABEL_COLUMN = "Label"


# ---------------------------------------------------------------------------
# Keyword groups encode the working hypothesis.
#
# - asp_core: ASP / Answer Set Programming core terms.
# - asp_ecosystem: tools/solvers/extensions tightly coupled to ASP.
# - logic_neighbour: logic programming neighbours (Prolog, Datalog, Argumentation,
#   NMR, Description Logic, Epistemic). Useful to test whether label 5 prefers
#   ASP over generic logic.
# - ai_symbolic: AI / ML / NN topics that integrate with logic (the "modern AI
#   meets logic" cluster). cav and lics label 5 lean on this.
# - low_signal: structural cues that usually mean "not a real paper" or "not
#   primary research". Strong prior to label 1.
# ---------------------------------------------------------------------------

KEYWORD_GROUPS: dict[str, list[str]] = {
    "asp_core": [
        r"\basp\b",
        r"answer set",
        r"answer sets",
        r"\baspq\b",
        r"asp\(q\)",
        r"clingo",
        r"dlv",
        r"hex program",
        r"stable model",
        r"stable models",
    ],
    "asp_ecosystem": [
        r"telingo",
        r"asp solver",
        r"asp solving",
        r"asp encoding",
        r"asp program",
        r"asp programs",
        r"asp-based",
        r"asp based",
        r"grounder",
        r"grounding",
    ],
    "logic_neighbour": [
        r"\bprolog\b",
        r"\bdatalog\b",
        r"argumentation",
        r"description logic",
        r"epistemic",
        r"non-?monotonic",
        r"default logic",
        r"abductive",
        r"abduction",
        r"\bilp\b",
        r"inductive logic",
        r"horn clause",
    ],
    "ai_symbolic": [
        r"neural",
        r"neuro-?symbolic",
        r"neurosymbolic",
        r"deep learning",
        r"\bllm\b",
        r"\bllms\b",
        r"transformer",
        r"reinforcement learning",
        r"machine learning",
        r"\bgnn\b",
        r"graph neural",
        r"explainab",
        r"learning",
    ],
    "low_signal_title": [
        r"proceedings",
        r"\(short paper\)",
        r"short paper",
        r"invited talk",
        r"invited paper",
        r"\(extended abstract\)",
        r"extended abstract",
        r"abstract\)",
        r"\bdoctoral\b",
        r"doctoral consortium",
        r"phd thesis",
        r"workshop summary",
        r"\bsystem demonstration\b",
        r"system description",
    ],
}


def normalize_title(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = text.replace("&apos;", "'").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def keyword_flags(titles: pd.Series) -> pd.DataFrame:
    flags = {}
    for group, patterns in KEYWORD_GROUPS.items():
        compiled = re.compile("|".join(patterns), flags=re.IGNORECASE)
        flags[group] = titles.fillna("").map(lambda value: bool(compiled.search(str(value).lower())))
    return pd.DataFrame(flags)


def first_surname(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    first_author = str(value).split(",")[0]
    tokens = re.findall(r"[A-Za-zÀ-ỹÁ-ỹ'’-]+", first_author)
    return tokens[-1].lower() if tokens else ""


def write_keyword_label_table(train_df: pd.DataFrame, flags: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    label_overall_mean = train_df[LABEL_COLUMN].mean()
    for column in flags.columns:
        present = train_df[flags[column]]
        absent = train_df[~flags[column]]
        if len(present) == 0:
            continue
        rows.append(
            {
                "keyword_group": column,
                "n_with": int(len(present)),
                "n_without": int(len(absent)),
                "label_mean_with": float(present[LABEL_COLUMN].mean()),
                "label_mean_without": float(absent[LABEL_COLUMN].mean()),
                "label_mean_delta": float(present[LABEL_COLUMN].mean() - absent[LABEL_COLUMN].mean()),
                "share_label_5_with": float((present[LABEL_COLUMN] == 5).mean()),
                "share_label_5_without": float((absent[LABEL_COLUMN] == 5).mean()),
                "share_label_1_with": float((present[LABEL_COLUMN] == 1).mean()),
                "share_label_1_without": float((absent[LABEL_COLUMN] == 1).mean()),
                "label_overall_mean": float(label_overall_mean),
            }
        )
    table = pd.DataFrame(rows).sort_values("label_mean_delta", ascending=False)
    table.to_csv(out_path, index=False)
    return table


def write_keyword_venue_label_table(train_df: pd.DataFrame, flags: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for column in flags.columns:
        for venue, sub in train_df.groupby("venue"):
            sub_flags = flags.loc[sub.index, column]
            present = sub[sub_flags]
            absent = sub[~sub_flags]
            if len(present) == 0:
                continue
            rows.append(
                {
                    "keyword_group": column,
                    "venue": venue,
                    "n_with": int(len(present)),
                    "n_without": int(len(absent)),
                    "label_mean_with": float(present[LABEL_COLUMN].mean()),
                    "label_mean_without": float(absent[LABEL_COLUMN].mean()),
                    "label_mean_delta": float(present[LABEL_COLUMN].mean() - absent[LABEL_COLUMN].mean()),
                    "share_label_5_with": float((present[LABEL_COLUMN] == 5).mean()),
                    "share_label_1_with": float((present[LABEL_COLUMN] == 1).mean()),
                }
            )
    table = pd.DataFrame(rows).sort_values(["keyword_group", "label_mean_delta"], ascending=[True, False])
    table.to_csv(out_path, index=False)
    return table


def write_missingness_label(train_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    has_authors = train_df["authors"].notna() & train_df["authors"].astype(str).str.strip().ne("")
    rows = [
        {
            "group": "has_authors",
            "n": int(has_authors.sum()),
            "label_mean": float(train_df.loc[has_authors, LABEL_COLUMN].mean()),
            "share_label_5": float((train_df.loc[has_authors, LABEL_COLUMN] == 5).mean()),
            "share_label_1": float((train_df.loc[has_authors, LABEL_COLUMN] == 1).mean()),
        },
        {
            "group": "missing_authors",
            "n": int((~has_authors).sum()),
            "label_mean": float(train_df.loc[~has_authors, LABEL_COLUMN].mean()),
            "share_label_5": float((train_df.loc[~has_authors, LABEL_COLUMN] == 5).mean()),
            "share_label_1": float((train_df.loc[~has_authors, LABEL_COLUMN] == 1).mean()),
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    return table


def write_duplicate_title_cross_split(
    train_df: pd.DataFrame,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    train_norm = train_df.assign(title_norm=train_df["title"].map(normalize_title))
    public_norm = public_df.assign(title_norm=public_df["title"].map(normalize_title))
    private_norm = private_df.assign(title_norm=private_df["title"].map(normalize_title))
    train_titles = set(train_norm["title_norm"]) - {""}

    rows = []
    for split_name, split_df in [("public", public_norm), ("private", private_norm)]:
        for _, row in split_df.iterrows():
            if row["title_norm"] in train_titles and row["title_norm"]:
                match = train_norm[train_norm["title_norm"] == row["title_norm"]]
                rows.append(
                    {
                        "split": split_name,
                        "test_id": row["id"],
                        "test_title": row["title"],
                        "train_ids": ",".join(map(str, match["id"].tolist())),
                        "train_labels": ",".join(map(str, match[LABEL_COLUMN].tolist())),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    return table


def write_first_author_leakage(
    train_df: pd.DataFrame,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    train_aug = train_df.assign(first_surname=train_df["authors"].map(first_surname))
    public_aug = public_df.assign(first_surname=public_df["authors"].map(first_surname))
    private_aug = private_df.assign(first_surname=private_df["authors"].map(first_surname))

    train_stats = (
        train_aug[train_aug["first_surname"] != ""]
        .groupby("first_surname")[LABEL_COLUMN]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "train_count", "mean": "train_label_mean"})
    )
    train_stats["train_label_std"] = train_stats["std"].fillna(0.0)

    rows = []
    for split_name, split_df in [("public", public_aug), ("private", private_aug)]:
        merged = split_df.merge(train_stats, on="first_surname", how="left")
        n_total = len(merged)
        n_with_match = int(merged["train_count"].notna().sum())
        n_single_anchor = int((merged["train_count"] == 1).sum())
        rows.append(
            {
                "split": split_name,
                "test_rows": n_total,
                "with_first_author_train_match": n_with_match,
                "share_with_first_author_train_match": n_with_match / max(n_total, 1),
                "single_train_paper_anchor_rows": n_single_anchor,
                "share_single_anchor": n_single_anchor / max(n_total, 1),
                "train_count_mean_when_matched": float(merged["train_count"].dropna().mean()) if n_with_match else float("nan"),
                "train_label_std_mean_when_matched": float(merged["train_label_std"].dropna().mean()) if n_with_match else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    return table


def write_test_split_keyword_coverage(
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    rows = []
    for split_name, split_df in [("public", public_df), ("private", private_df)]:
        flags = keyword_flags(split_df["title"])
        for column in flags.columns:
            rows.append(
                {
                    "split": split_name,
                    "keyword_group": column,
                    "share_with": float(flags[column].mean()),
                    "n_with": int(flags[column].sum()),
                    "n_total": int(len(split_df)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    return table


def keyword_proxy_score(flags: pd.DataFrame) -> np.ndarray:
    """Assign a tiny rule-based ordinal score from keyword flags.

    This is *only* a sanity probe for the hypothesis. It gives:

    - +2 for asp_core or asp_ecosystem hit (strong push toward 5)
    - +1 for ai_symbolic hit (push toward 4-5)
    - -1 for logic_neighbour without ASP (mild push toward 1-2)
    - -2 for low_signal_title hit (proceedings, short paper, ...)
    """
    asp = flags["asp_core"] | flags["asp_ecosystem"]
    score = np.zeros(len(flags), dtype=float)
    score = np.where(asp, score + 2.0, score)
    score = np.where(flags["ai_symbolic"] & ~asp, score + 1.0, score)
    score = np.where(flags["logic_neighbour"] & ~asp, score - 1.0, score)
    score = np.where(flags["low_signal_title"], score - 2.0, score)
    return score


def proxy_score_to_label(score: np.ndarray) -> np.ndarray:
    # Crude bin: score <= -2 -> 1, == -1 -> 2, 0 -> 3, +1 -> 4, +2 or more -> 5.
    labels = np.full(score.shape, 3, dtype=int)
    labels = np.where(score >= 2, 5, labels)
    labels = np.where(score == 1, 4, labels)
    labels = np.where(score == 0, 3, labels)
    labels = np.where(score == -1, 2, labels)
    labels = np.where(score <= -2, 1, labels)
    return labels


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    public = pd.read_csv(DATA_DIR / "public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test.csv")

    train_flags = keyword_flags(train["title"])

    keyword_label = write_keyword_label_table(train, train_flags, OUT_DIR / "keyword_label_table.csv")
    keyword_venue = write_keyword_venue_label_table(train, train_flags, OUT_DIR / "keyword_venue_label_table.csv")
    missingness = write_missingness_label(train, OUT_DIR / "missingness_label.csv")
    duplicates = write_duplicate_title_cross_split(train, public, private, OUT_DIR / "duplicate_title_cross_split.csv")
    leakage = write_first_author_leakage(train, public, private, OUT_DIR / "first_author_leakage.csv")
    coverage = write_test_split_keyword_coverage(public, private, OUT_DIR / "test_split_keyword_coverage.csv")

    proxy_score = keyword_proxy_score(train_flags)
    proxy_label = proxy_score_to_label(proxy_score)
    proxy_qwk = cohen_kappa_score(train[LABEL_COLUMN].astype(int).to_numpy(), proxy_label, weights="quadratic")
    proxy_dist = pd.Series(proxy_label).value_counts().sort_index().to_dict()
    real_dist = train[LABEL_COLUMN].value_counts().sort_index().to_dict()

    proxy_df = pd.DataFrame(
        {
            "id": train["id"],
            "venue": train["venue"],
            "year": train["year"],
            "true_label": train[LABEL_COLUMN].astype(int),
            "proxy_score": proxy_score,
            "proxy_label": proxy_label,
        }
    )
    proxy_df.to_csv(OUT_DIR / "keyword_proxy_predictions.csv", index=False)

    # Build a markdown report.
    lines: list[str] = ["# EDA v2 — ASP-relevance Hypothesis", ""]
    lines.append("## 1. Keyword groups vs Label (train)")
    lines.append("")
    lines.append(keyword_label.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- **asp_core** / **asp_ecosystem** with positive `label_mean_delta` and high `share_label_5_with` confirms ASP-related papers are pushed toward higher labels.")
    lines.append("- **low_signal_title** with negative delta confirms proceedings / short paper / abstract titles cluster at label 1.")
    lines.append("- **ai_symbolic** delta tells whether neural / ML / explainability cues drive label 4-5 across venues, which is what cav/lics label-5 examples suggested.")
    lines.append("- **logic_neighbour** delta tells whether non-ASP logic topics (Argumentation, Description Logic, ...) drift toward lower labels at iclp/lpnmr; this matters because the model needs to learn 'logic but not ASP' is not the same as 'ASP'.")
    lines.append("")

    lines.append("## 2. Keyword × venue (train)")
    lines.append("")
    lines.append(keyword_venue.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("Reading: even within the same venue, the ASP keyword should still increase the mean label. If it does, ASP signal is venue-independent — exactly what we need a semantic embedding to capture, since TF-IDF n-grams already exploit some of this but cannot generalize across paraphrases like 'ASP' vs 'answer set programming' vs 'stable model'.")
    lines.append("")

    lines.append("## 3. Missing authors vs Label")
    lines.append("")
    lines.append(missingness.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("If `missing_authors` clusters at label 1, that matches the proceedings/abstract pattern and explains why the current Ridge anchor uses `authors_clean` as a useful but not dominant feature.")
    lines.append("")

    lines.append("## 4. Cross-split duplicate titles")
    lines.append("")
    if duplicates.empty:
        lines.append(f"No exact normalized title overlap between train and public/private. Test rows = {len(public) + len(private)}.")
    else:
        lines.append(f"Found {len(duplicates)} duplicate-title rows across train and test splits:")
        lines.append("")
        lines.append(duplicates.head(50).to_markdown(index=False))
    lines.append("")

    lines.append("## 5. First-author leakage")
    lines.append("")
    lines.append(leakage.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("If a large share of test rows share a first-author surname with train, the existing TF-IDF model already exploits that. SPECTER2 will not lose that signal because we will keep `authors_clean` as a parallel TF-IDF feature in the stack.")
    lines.append("")

    lines.append("## 6. Keyword coverage on the test splits")
    lines.append("")
    lines.append(coverage.round(3).to_markdown(index=False))
    lines.append("")
    lines.append("This tells us how often each ASP / AI-symbolic cue appears on public vs private. If coverage is balanced, a semantic model that captures these cues should transfer cleanly between public and private leaderboard halves.")
    lines.append("")

    lines.append("## 7. Sanity probe — rule-based proxy QWK")
    lines.append("")
    lines.append(f"- Proxy QWK on train: **{proxy_qwk:.4f}** (no model fit, only the keyword rules above).")
    lines.append(f"- Proxy label distribution: `{proxy_dist}`")
    lines.append(f"- True label distribution:  `{real_dist}`")
    lines.append("")
    lines.append("Reading:")
    lines.append("")
    lines.append("- A non-trivial proxy QWK from 5 hand-written keyword groups is direct evidence that Label 1-5 is **content / topic relevance**, not a quality score. TF-IDF already approximates this with thousands of features, but it cannot generalise the way a sentence transformer can. This is exactly what we exploit in step 3.")
    lines.append("- The proxy distribution is intentionally peaked at label 3 because most papers do not match any of the curated keyword groups. Once we move to SPECTER2 embeddings, the signal will spread across all 5 labels because the embedding captures the rest of the paper's topic, not just hand-picked keywords.")
    lines.append("")

    lines.append("## 8. Decision before step 2")
    lines.append("")
    lines.append("If the keyword-group table shows that ASP-core papers have a clearly higher mean label than non-ASP papers within the same venue, we proceed with the SPECTER2 + abstract crawl plan. The remaining steps are:")
    lines.append("")
    lines.append("1. Crawl `abstract` and `tldr` from Semantic Scholar (already partially cached) and OpenAlex.")
    lines.append("2. Encode `title (+ abstract)` with allenai/specter2_base or sentence-transformers/all-MiniLM-L6-v2 as a CPU fallback.")
    lines.append("3. Train a Ridge ordinal head on the embeddings, repeat the 5x5 CV pattern, threshold-tune on OOF.")
    lines.append("4. Add an LLM zero/few-shot 1-5 score as a meta feature.")
    lines.append("5. Stack with the existing 0.62820 Ridge anchor and 0.63064 OpenAlex anchor.")

    (OUT_DIR / "eda_v2_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Print a short stdout summary so the user can sanity-check immediately.
    print("EDA v2 complete. Report written to", OUT_DIR / "eda_v2_report.md")
    print()
    print("Keyword × Label (train):")
    print(keyword_label.round(3).to_string(index=False))
    print()
    print("Missing authors vs Label:")
    print(missingness.round(3).to_string(index=False))
    print()
    print("Cross-split duplicates: rows =", len(duplicates))
    print()
    print("Keyword proxy QWK:", round(float(proxy_qwk), 4))
    print("Proxy distribution:", proxy_dist)
    print("True distribution: ", real_dist)


if __name__ == "__main__":
    main()
