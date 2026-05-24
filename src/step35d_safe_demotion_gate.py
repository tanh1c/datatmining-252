from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "outputs" / "external"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
STEP25_DIR = ROOT / "outputs" / "bge_m3_frozen_anchor" / "best_safe"
OUT = ROOT / "outputs" / "step35d_safe_demotion_gate"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

STEP25_OOF_QWK = 0.6612605583792329
SAFE_L1_CAP = 0.147
MAX_TEST_CHANGED = 12
CS = [0.5, 1.0, 2.0]
MARGIN_CAPS = [0.015, 0.025, 0.04, 0.06, 0.08]
CONF_GAPS = [0.04, 0.06, 0.08, 0.10, 0.14]
MAX_ROWS = [3, 4, 5, 6, 8, 10, 12]
THRESHOLDS = np.asarray([1.900004, 2.489778, 3.222418, 4.101986], dtype=float)

RUBRICS = {
    1: "irrelevant not about answer set programming stable models logic programming nonmonotonic reasoning ASP solving or knowledge representation",
    2: "adjacent broad artificial intelligence machine learning automata verification databases formal methods or reasoning with little direct ASP relevance",
    3: "related symbolic reasoning knowledge representation logic planning argumentation constraints verification or declarative methods indirectly connected to ASP",
    4: "relevant clear logic programming answer set programming stable model semantics ASP solvers grounding nonmonotonic reasoning or KR with substantial ASP relevance",
    5: "core central answer set programming paper ASP theory stable model semantics clingo dlv clasp gringo grounding solving optimization applications of ASP LPNMR ICLP",
}

PATTERNS = {
    "core_asp": r"\b(answer set programming|answer-set programming|asp\b|answer sets?\b|stable model|stable models|clingo|dlv|clasp|gringo|grounding|equilibrium logic|lpnmr)\b",
    "logic_programming": r"\b(logic programming|logic programs|logic program|prolog|datalog|declarative programming|nonmonotonic|non-monotonic|default logic|default theories)\b",
    "kr_reasoning": r"\b(knowledge representation|reasoning|argumentation|abduction|belief revision|planning|preferences|epistemic|inconsistency)\b",
    "formal_adjacent": r"\b(model checking|verification|synthesis|temporal logic|automata|reachability|theorem proving|invariant|program repair)\b",
    "ml_adjacent": r"\b(machine learning|deep learning|neural|reinforcement learning|large language model|llm|generative ai)\b",
    "meta": r"\b(proceedings|preface|editorial|front matter|invited talk|short paper|extended abstract|doctoral consortium)\b",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def predicted_dist(labels: np.ndarray) -> np.ndarray:
    return pd.Series(labels).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()


def load_abstracts() -> pd.DataFrame:
    for path in [EXTERNAL_DIR / "abstracts_merged_v4_clean.csv", EXTERNAL_DIR / "abstracts_merged_v3.csv"]:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame(columns=["source_split", "id", "abstract", "s2_venue", "s2_fields_of_study"])


def attach_abstracts(df: pd.DataFrame, split: str, abstracts: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_split = {"train": "train", "public": "public_test", "private": "private_test"}[split]
    if len(abstracts):
        cols = [c for c in ["source_split", "id", "abstract", "s2_venue", "s2_fields_of_study"] if c in abstracts.columns]
        out = out.merge(abstracts[abstracts["source_split"].eq(source_split)][cols], on="id", how="left")
    for col in ["abstract", "s2_venue", "s2_fields_of_study"]:
        if col not in out:
            out[col] = ""
        out[col] = out[col].fillna("")
    return out


def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        "title: " + df["title"].map(clean_text)
        + " venue: " + df["venue"].map(clean_text)
        + " year: " + df["year"].map(clean_text)
        + " abstract: " + df["abstract"].map(clean_text)
        + " semantic venue: " + df["s2_venue"].map(clean_text)
        + " fields: " + df["s2_fields_of_study"].map(clean_text)
    )


def add_pattern_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = build_text(out).str.lower()
    venue = out["venue"].fillna("").str.lower()
    title = out["title"].fillna("").str.lower()
    for name, pattern in PATTERNS.items():
        out[name] = text.str.contains(pattern, regex=True).astype(int)
    out["protected_venue"] = venue.str.contains(r"\biclp\b|\blpnmr\b|\blics\b|\bkr\b|logic programming|knowledge representation", regex=True).astype(int)
    out["meta_title"] = title.str.contains(r"invited talk|proceedings|preface|editorial|front matter|short paper|extended abstract|doctoral consortium", regex=True).astype(int)
    out["cav_formal_venue"] = venue.str.contains(r"\bcav\b|computer aided verification|formal methods", regex=True).astype(int)
    out["protected_logic"] = ((out["protected_venue"].eq(1) | out["core_asp"].eq(1) | out["logic_programming"].eq(1) | out["kr_reasoning"].eq(1)) & out["meta"].eq(0) & out["cav_formal_venue"].eq(0)).astype(int)
    out["safe_demotion_context"] = (out["meta"].eq(1) | out["meta_title"].eq(1) | out["cav_formal_venue"].eq(1) | ((out["formal_adjacent"].eq(1) | out["ml_adjacent"].eq(1)) & out["core_asp"].eq(0) & out["logic_programming"].eq(0))).astype(int)
    return out


def load_step25_scores(split: str) -> pd.DataFrame:
    file_name = "oof_scores.csv" if split == "train" else f"{split}_scores.csv"
    df = pd.read_csv(STEP25_DIR / file_name)
    score_col = "oof_score" if "oof_score" in df.columns else "score"
    pred_col = "oof_pred" if "oof_pred" in df.columns else "pred"
    return df[["id", score_col, pred_col]].rename(columns={score_col: "step25_score", pred_col: "step25_pred"})


def build_split(raw: pd.DataFrame, split: str, abstracts: pd.DataFrame) -> pd.DataFrame:
    df = attach_abstracts(raw, split, abstracts)
    df = add_pattern_columns(df)
    df["paper_text"] = build_text(df)
    df = df.merge(load_step25_scores(split), on="id", how="left")
    score = df["step25_score"].to_numpy(float)
    df["threshold_margin"] = np.min(np.abs(score[:, None] - THRESHOLDS[None, :]), axis=1)
    return df


def expand_pairs(df: pd.DataFrame, labels: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray | None]:
    texts = []
    y_bin = [] if labels is not None else None
    for row_pos, (_, row) in enumerate(df.iterrows()):
        hints = " ".join(name for name in PATTERNS if int(row[name]) == 1)
        for cls in [1, 2, 3, 4, 5]:
            texts.append(f"rubric class {cls}: {RUBRICS[cls]} pattern hints: {hints} paper: {row['paper_text']}")
            if labels is not None:
                y_bin.append(int(int(labels[row_pos]) == cls))
    return pd.Series(texts), np.asarray(y_bin, dtype=int) if y_bin is not None else None


def make_model(c: float):
    word = make_pipeline(FunctionTransformer(lambda x: x, validate=False), TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents="unicode", max_features=180000))
    char = make_pipeline(FunctionTransformer(lambda x: x, validate=False), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=120000))
    return make_pipeline(FeatureUnion([("word", word), ("char", char)]), LogisticRegression(C=c, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=252))


def probs_to_matrix(probs: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = probs.reshape(n_rows, 5).astype(float)
    matrix = np.clip(matrix, 1e-6, None)
    return matrix / matrix.sum(axis=1, keepdims=True)


def cv_rubric_probs(train: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame, y: np.ndarray, c: float) -> dict[str, np.ndarray]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=352)
    oof = np.zeros((len(train), 5), dtype=float)
    pub_probs = []
    pri_probs = []
    public_texts, _ = expand_pairs(public)
    private_texts, _ = expand_pairs(private)
    for train_idx, valid_idx in folds.split(train, y):
        tr_texts, tr_y = expand_pairs(train.iloc[train_idx].reset_index(drop=True), y[train_idx])
        va_texts, _ = expand_pairs(train.iloc[valid_idx].reset_index(drop=True))
        model = make_model(c)
        model.fit(tr_texts, tr_y)
        oof[valid_idx] = probs_to_matrix(model.predict_proba(va_texts)[:, 1], len(valid_idx))
        pub_probs.append(probs_to_matrix(model.predict_proba(public_texts)[:, 1], len(public)))
        pri_probs.append(probs_to_matrix(model.predict_proba(private_texts)[:, 1], len(private)))
    return {"oof": oof, "pub": np.mean(pub_probs, axis=0), "pri": np.mean(pri_probs, axis=0)}


def proposal_frame(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["rubric_pred"] = probs.argmax(axis=1) + 1
    out["rubric_conf"] = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    out["rubric_gap"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["rubric_expected"] = probs @ np.arange(1, 6, dtype=float)
    out["demotion_strength"] = out["step25_pred"].astype(float) - out["rubric_expected"]
    out["proposed_pred"] = (out["step25_pred"].astype(int) - 1).clip(1, 5)
    return out


def select_mask(df: pd.DataFrame, margin_cap: float, conf_gap: float, max_rows: int, require_meta_or_cav: bool, protect_all_logic: bool) -> pd.Series:
    mask = (
        df["step25_pred"].astype(int).ge(2)
        & df["rubric_pred"].astype(int).lt(df["step25_pred"].astype(int))
        & df["threshold_margin"].le(margin_cap)
        & df["rubric_gap"].ge(conf_gap)
        & df["safe_demotion_context"].eq(1)
    )
    if require_meta_or_cav:
        mask &= (df["meta"].eq(1) | df["meta_title"].eq(1) | df["cav_formal_venue"].eq(1))
    if protect_all_logic:
        mask &= df["protected_logic"].eq(0)
    idx = df.index[mask].tolist()
    if len(idx) <= max_rows:
        return mask
    sub = df.loc[idx].copy()
    sub["priority"] = sub["demotion_strength"] + sub["rubric_gap"] - sub["threshold_margin"] + 0.05 * sub["safe_demotion_context"]
    keep = set(sub.sort_values("priority", ascending=False).head(max_rows).index)
    return df.index.to_series().isin(keep)


def apply_rule(df: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    labels = df["step25_pred"].astype(int).to_numpy().copy()
    mask = select_mask(df, float(rule["margin_cap"]), float(rule["conf_gap"]), int(rule["max_rows"]), bool(rule["require_meta_or_cav"]), bool(rule["protect_all_logic"])).to_numpy(bool)
    labels[mask] = df.loc[mask, "proposed_pred"].astype(int).to_numpy()
    return labels


def change_summary(y: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray, pub_base: np.ndarray, pub_new: np.ndarray, pri_base: np.ndarray, pri_new: np.ndarray, train: pd.DataFrame) -> dict[str, Any]:
    changed = base_pred != new_pred
    if changed.any():
        delta_abs_error = np.abs(base_pred[changed] - y[changed]) - np.abs(new_pred[changed] - y[changed])
        sub = train.loc[changed]
        protected_demotions = int(((new_pred[changed] < base_pred[changed]) & sub["protected_logic"].to_numpy(bool)).sum())
    else:
        delta_abs_error = np.array([])
        protected_demotions = 0
    return {
        "changed": int(changed.sum()),
        "improved": int((delta_abs_error > 0).sum()) if len(delta_abs_error) else 0,
        "worsened": int((delta_abs_error < 0).sum()) if len(delta_abs_error) else 0,
        "mean_delta_abs_error": float(delta_abs_error.mean()) if len(delta_abs_error) else 0.0,
        "protected_demotions": protected_demotions,
        "public_changed": int((pub_base != pub_new).sum()),
        "private_changed": int((pri_base != pri_new).sum()),
    }


def evaluate(rule: dict[str, Any], train: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame, y: np.ndarray, train_dist: np.ndarray) -> dict[str, Any]:
    train_pred = apply_rule(train, rule)
    pub_pred = apply_rule(public, rule)
    pri_pred = apply_rule(private, rule)
    base_train = train["step25_pred"].astype(int).to_numpy()
    base_pub = public["step25_pred"].astype(int).to_numpy()
    base_pri = private["step25_pred"].astype(int).to_numpy()
    test_pred = np.concatenate([pub_pred, pri_pred])
    qwk = float(cohen_kappa_score(y, train_pred, weights="quadratic"))
    return {
        **rule,
        "oof_qwk": qwk,
        "oof_lift_vs_step25": float(qwk - STEP25_OOF_QWK),
        "test_l1": float(np.sum(np.abs(predicted_dist(test_pred) - train_dist))),
        **change_summary(y, base_train, train_pred, base_pub, pub_pred, base_pri, pri_pred, train),
        "pub_dist": str({int(k): int(v) for k, v in pd.Series(pub_pred).value_counts().sort_index().items()}),
        "pri_dist": str({int(k): int(v) for k, v in pd.Series(pri_pred).value_counts().sort_index().items()}),
    }


def write_pick(label: str, row: pd.Series, train: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame, sample: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    rule = {key: row[key] for key in ["c", "margin_cap", "conf_gap", "max_rows", "require_meta_or_cav", "protect_all_logic"]}
    out_dir = OUT / label
    out_dir.mkdir(exist_ok=True)
    train_pred = apply_rule(train, rule)
    pub_pred = apply_rule(public, rule)
    pri_pred = apply_rule(private, rule)
    pd.DataFrame({"id": train["id"], "Label": y, "oof_score": train["step25_score"], "oof_pred": train_pred}).to_csv(out_dir / "oof_scores.csv", index=False)
    pd.DataFrame({"id": public["id"], "score": public["step25_score"], "pred": pub_pred}).to_csv(out_dir / "public_scores.csv", index=False)
    pd.DataFrame({"id": private["id"], "score": private["step25_score"], "pred": pri_pred}).to_csv(out_dir / "private_scores.csv", index=False)
    for split, df, pred in [("public", public, pub_pred), ("private", private, pri_pred)]:
        mask = pred != df["step25_pred"].astype(int).to_numpy()
        df.loc[mask, ["id", "title", "venue", "step25_pred", "rubric_pred", "rubric_expected", "rubric_gap", "threshold_margin", "safe_demotion_context", "protected_logic", "proposed_pred"]].assign(split=split).to_csv(out_dir / f"changed_{split}.csv", index=False)
    combo = pd.concat([pd.DataFrame({"id": public["id"], "Label": pub_pred}), pd.DataFrame({"id": private["id"], "Label": pri_pred})], ignore_index=True)
    sub = sample[["id"]].merge(combo, on="id", how="left")
    sub["Label"] = sub["Label"].astype(int)
    path = SUBMIT_DIR / f"next_step35d_{label}_submission.csv"
    sub.to_csv(path, index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)
    return {"label": label, "submission": str(path), "oof_qwk": float(row["oof_qwk"]), "oof_lift_vs_step25": float(row["oof_lift_vs_step25"]), "test_l1": float(row["test_l1"]), "changed": int(row["public_changed"] + row["private_changed"])}


def main() -> None:
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    raw_public = pd.read_csv(DATA_DIR / "public_test.csv")
    raw_private = pd.read_csv(DATA_DIR / "private_test.csv")
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")
    y = raw_train["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()
    abstracts = load_abstracts()
    train = build_split(raw_train, "train", abstracts)
    public = build_split(raw_public, "public", abstracts)
    private = build_split(raw_private, "private", abstracts)

    all_rows = []
    frames_by_c = {}
    for c in CS:
        probs = cv_rubric_probs(train, public, private, y, c)
        c_train = proposal_frame(train, probs["oof"])
        c_public = proposal_frame(public, probs["pub"])
        c_private = proposal_frame(private, probs["pri"])
        frames_by_c[c] = (c_train, c_public, c_private)
        for margin_cap in MARGIN_CAPS:
            for conf_gap in CONF_GAPS:
                for max_rows in MAX_ROWS:
                    for require_meta_or_cav in [False, True]:
                        for protect_all_logic in [True, False]:
                            rule = {"c": c, "margin_cap": margin_cap, "conf_gap": conf_gap, "max_rows": max_rows, "require_meta_or_cav": require_meta_or_cav, "protect_all_logic": protect_all_logic}
                            all_rows.append(evaluate(rule, c_train, c_public, c_private, y, train_dist))

    candidates = pd.DataFrame(all_rows).sort_values(["oof_lift_vs_step25", "test_l1", "changed"], ascending=[False, True, True])
    candidates.to_csv(OUT / "candidates.csv", index=False)
    safe = candidates[
        (candidates["oof_lift_vs_step25"] > 0)
        & (candidates["test_l1"] <= SAFE_L1_CAP)
        & ((candidates["public_changed"] + candidates["private_changed"]) <= MAX_TEST_CHANGED)
        & (candidates["improved"] >= candidates["worsened"])
        & (candidates["protected_demotions"] <= 1)
        & ((candidates["public_changed"] + candidates["private_changed"]) > 0)
    ].sort_values(["oof_lift_vs_step25", "mean_delta_abs_error", "test_l1"], ascending=[False, False, True])
    safe.to_csv(OUT / "safe_candidates.csv", index=False)

    picks = []
    for label, (_, row) in zip(["best_safe", "second_safe"], safe.head(2).iterrows()):
        c_train, c_public, c_private = frames_by_c[float(row["c"])]
        picks.append(write_pick(label, row, c_train, c_public, c_private, sample, y))

    summary = {"step25_oof_qwk": STEP25_OOF_QWK, "safe_l1_cap": SAFE_L1_CAP, "max_test_changed": MAX_TEST_CHANGED, "safe_candidates": int(len(safe)), "picks": picks}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    top = safe.head(20) if len(safe) else candidates.head(20)
    show_cols = ["c", "margin_cap", "conf_gap", "max_rows", "require_meta_or_cav", "protect_all_logic", "oof_qwk", "oof_lift_vs_step25", "test_l1", "changed", "improved", "worsened", "protected_demotions", "public_changed", "private_changed"]
    lines = ["# Step35D safe demotion gate", ""]
    lines.append("This step restricts the Step35C rubric signal to one-step demotions in safe over-promotion contexts, while protecting ICLP/KR/LICS/LPNMR/core logic rows unless they have explicit meta/CAV/formal cues.")
    lines.extend(["", "## Top safe candidates" if len(safe) else "## Top candidates — no safe candidate passed filters", ""])
    lines.append(top[show_cols].to_markdown(index=False))
    lines.extend(["", "## Picks", "", "```json", json.dumps(picks, indent=2, ensure_ascii=False), "```", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== Step35D safe demotion gate ===")
    print(top[show_cols].head(15).to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
