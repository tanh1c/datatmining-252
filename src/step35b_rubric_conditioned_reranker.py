from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
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
OUT = ROOT / "outputs" / "step35b_rubric_conditioned_reranker"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

STEP25_OOF_QWK = 0.6612605583792329
SAFE_L1_CAP = 0.147
MAX_TEST_CHANGED = 15
BLEND_WEIGHTS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
THRESHOLD_LAMBDAS = [0.5, 1.0, 2.0, 4.0]
CS = [0.25, 0.5, 1.0, 2.0]

RUBRICS = {
    1: "irrelevant: not about answer set programming, stable models, logic programming, nonmonotonic reasoning, ASP solving, or closely related knowledge representation topics",
    2: "adjacent: broad artificial intelligence, machine learning, automata, verification, databases, formal methods, or reasoning topic with little direct ASP or logic programming relevance",
    3: "related: symbolic reasoning, knowledge representation, logic, planning, argumentation, constraints, verification, or declarative methods connected to ASP only indirectly",
    4: "relevant: clear logic programming, answer set programming, stable model semantics, ASP solvers, grounding, nonmonotonic reasoning, or KR methods with substantial ASP relevance",
    5: "core: central answer set programming paper about ASP theory, stable model semantics, ASP systems such as clingo dlv clasp gringo, grounding, solving, optimization, applications of ASP, or LPNMR ICLP core logic programming work",
}

PATTERNS = {
    "core_asp": r"\b(answer set programming|answer-set programming|asp\b|answer sets?\b|stable model|stable models|clingo|dlv|clasp|gringo|grounding|lpnmr)\b",
    "logic_programming": r"\b(logic programming|logic programs|logic program|prolog|datalog|declarative programming|nonmonotonic|non-monotonic)\b",
    "kr_reasoning": r"\b(knowledge representation|reasoning|argumentation|abduction|belief revision|planning|preferences|epistemic)\b",
    "formal_adjacent": r"\b(model checking|verification|synthesis|temporal logic|automata|reachability|theorem proving|invariant)\b",
    "ml_adjacent": r"\b(machine learning|deep learning|neural|reinforcement learning|large language model|llm|generative ai)\b",
    "meta": r"\b(proceedings|preface|editorial|front matter|invited talk|short paper|extended abstract|doctoral consortium)\b",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def scores_to_labels(scores: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def predicted_dist(labels: np.ndarray) -> np.ndarray:
    return pd.Series(labels).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, train_dist: np.ndarray, lambd: float, seed: int = 42) -> tuple[np.ndarray, float]:
    def objective(raw: np.ndarray) -> float:
        thresholds = np.sort(raw)
        gap = np.min(np.diff(thresholds))
        gap_penalty = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(scores, thresholds)
        qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
        dist_penalty = float(np.sum(np.abs(predicted_dist(labels) - train_dist)))
        return -qwk + gap_penalty + lambd * dist_penalty

    result = differential_evolution(
        objective,
        [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed,
        maxiter=100,
        popsize=12,
        polish=True,
        updating="immediate",
        workers=1,
    )
    thresholds = np.sort(result.x)
    qwk = float(cohen_kappa_score(y_true, scores_to_labels(scores, thresholds), weights="quadratic"))
    return thresholds, qwk


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
    for name, pattern in PATTERNS.items():
        out[name] = text.str.contains(pattern, regex=True).astype(int)
    out["signal_sum"] = out[list(PATTERNS)].sum(axis=1)
    return out


def load_scores(split: str, directory: Path, prefix: str) -> pd.DataFrame:
    file_name = "oof_scores.csv" if split == "train" else f"{split}_scores.csv"
    df = pd.read_csv(directory / file_name)
    score_col = "oof_score" if "oof_score" in df.columns else "score"
    pred_col = "oof_pred" if "oof_pred" in df.columns else "pred"
    return df[["id", score_col, pred_col]].rename(columns={score_col: f"{prefix}_score", pred_col: f"{prefix}_pred"})


def build_split(raw: pd.DataFrame, split: str, abstracts: pd.DataFrame) -> pd.DataFrame:
    df = attach_abstracts(raw, split, abstracts)
    df = add_pattern_columns(df)
    df["paper_text"] = build_text(df)
    df = df.merge(load_scores(split, STEP25_DIR, "step25"), on="id", how="left")
    return df


def expand_pairs(df: pd.DataFrame, labels: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray | None, np.ndarray]:
    texts = []
    y_bin = [] if labels is not None else None
    row_ids = []
    for row_pos, (_, row) in enumerate(df.iterrows()):
        paper_text = row["paper_text"]
        pattern_hint = " ".join(name for name in PATTERNS if int(row[name]) == 1)
        for cls in [1, 2, 3, 4, 5]:
            texts.append(f"rubric class {cls}: {RUBRICS[cls]} pattern hints: {pattern_hint} paper: {paper_text}")
            row_ids.append(row_pos)
            if labels is not None:
                y_bin.append(int(int(labels[row_pos]) == cls))
    return pd.Series(texts), np.asarray(y_bin, dtype=int) if y_bin is not None else None, np.asarray(row_ids, dtype=int)


def make_model(c: float):
    word = make_pipeline(
        FunctionTransformer(lambda x: x, validate=False),
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, strip_accents="unicode", max_features=180000),
    )
    char = make_pipeline(
        FunctionTransformer(lambda x: x, validate=False),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=120000),
    )
    features = FeatureUnion([("word", word), ("char", char)])
    return make_pipeline(features, LogisticRegression(C=c, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=252))


def pair_prob_to_score(probs: np.ndarray, row_ids: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.zeros((n_rows, 5), dtype=float)
    for idx, prob in zip(row_ids, probs):
        cls_slot = int((np.where(row_ids == idx)[0][0]) % 5) if False else 0
    cursor = 0
    for row_idx in range(n_rows):
        raw = probs[cursor:cursor + 5].astype(float)
        cursor += 5
        raw = np.clip(raw, 1e-6, None)
        matrix[row_idx] = raw / raw.sum()
    classes = np.arange(1, 6, dtype=float)
    return matrix @ classes


def cv_rubric_scores(train: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame, y: np.ndarray, c: float) -> dict[str, np.ndarray]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=352)
    oof = np.zeros(len(train), dtype=float)
    pub_scores = []
    pri_scores = []
    public_texts, _, public_rows = expand_pairs(public)
    private_texts, _, private_rows = expand_pairs(private)
    for fold_train_idx, fold_valid_idx in folds.split(train, y):
        tr_texts, tr_y, _ = expand_pairs(train.iloc[fold_train_idx].reset_index(drop=True), y[fold_train_idx])
        va_texts, _, va_rows = expand_pairs(train.iloc[fold_valid_idx].reset_index(drop=True))
        model = make_model(c)
        model.fit(tr_texts, tr_y)
        va_prob = model.predict_proba(va_texts)[:, 1]
        pub_prob = model.predict_proba(public_texts)[:, 1]
        pri_prob = model.predict_proba(private_texts)[:, 1]
        oof[fold_valid_idx] = pair_prob_to_score(va_prob, va_rows, len(fold_valid_idx))
        pub_scores.append(pair_prob_to_score(pub_prob, public_rows, len(public)))
        pri_scores.append(pair_prob_to_score(pri_prob, private_rows, len(private)))
    return {"oof": oof, "pub": np.mean(pub_scores, axis=0), "pri": np.mean(pri_scores, axis=0)}


def align_scale(candidate: dict[str, np.ndarray], base_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    train = candidate["oof"].astype(float)
    base = base_scores["oof"].astype(float)
    cand_std = float(np.std(train))
    base_std = float(np.std(base))
    if cand_std < 1e-9:
        return {k: np.full_like(v, float(np.mean(base)), dtype=float) for k, v in candidate.items()}
    return {k: (v - float(np.mean(train))) / cand_std * base_std + float(np.mean(base)) for k, v in candidate.items()}


def change_summary(y: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray, pub_base: np.ndarray, pub_new: np.ndarray, pri_base: np.ndarray, pri_new: np.ndarray, train: pd.DataFrame) -> dict[str, Any]:
    changed = base_pred != new_pred
    if changed.any():
        delta_abs_error = np.abs(base_pred[changed] - y[changed]) - np.abs(new_pred[changed] - y[changed])
        changed_train = train.loc[changed].copy()
        core = changed_train["core_asp"].to_numpy(bool) | changed_train["logic_programming"].to_numpy(bool)
        risky = (changed_train["formal_adjacent"].to_numpy(bool) | changed_train["ml_adjacent"].to_numpy(bool)) & ~core
        risky_promotions = int(((new_pred[changed] > base_pred[changed]) & risky).sum())
        core_demotions = int(((new_pred[changed] < base_pred[changed]) & core).sum())
    else:
        delta_abs_error = np.array([])
        risky_promotions = 0
        core_demotions = 0
    return {
        "changed": int(changed.sum()),
        "improved": int((delta_abs_error > 0).sum()) if len(delta_abs_error) else 0,
        "worsened": int((delta_abs_error < 0).sum()) if len(delta_abs_error) else 0,
        "mean_delta_abs_error": float(delta_abs_error.mean()) if len(delta_abs_error) else 0.0,
        "risky_promotions": risky_promotions,
        "core_demotions": core_demotions,
        "public_changed": int((pub_base != pub_new).sum()),
        "private_changed": int((pri_base != pri_new).sum()),
    }


def evaluate(name: str, kind: str, c: float, weight: float, lambd: float, scores: dict[str, np.ndarray], y: np.ndarray, train_dist: np.ndarray, train: pd.DataFrame, base_preds: dict[str, np.ndarray]) -> dict[str, Any]:
    thresholds, qwk = tune_thresholds(y, scores["oof"], train_dist, lambd=lambd)
    preds = {
        "oof": scores_to_labels(scores["oof"], thresholds),
        "pub": scores_to_labels(scores["pub"], thresholds),
        "pri": scores_to_labels(scores["pri"], thresholds),
    }
    test_pred = np.concatenate([preds["pub"], preds["pri"]])
    summary = change_summary(y, base_preds["oof"], preds["oof"], base_preds["pub"], preds["pub"], base_preds["pri"], preds["pri"], train)
    return {
        "name": name,
        "kind": kind,
        "c": c,
        "weight": weight,
        "threshold_lambda": lambd,
        "oof_qwk": qwk,
        "oof_lift_vs_step25": float(qwk - STEP25_OOF_QWK),
        "test_l1": float(np.sum(np.abs(predicted_dist(test_pred) - train_dist))),
        "thresholds": thresholds.tolist(),
        "scores": scores,
        "preds": preds,
        "pub_dist": {int(k): int(v) for k, v in pd.Series(preds["pub"]).value_counts().sort_index().items()},
        "pri_dist": {int(k): int(v) for k, v in pd.Series(preds["pri"]).value_counts().sort_index().items()},
        **summary,
    }


def write_submission(label: str, result: dict[str, Any], train: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    out_dir = OUT / label
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"id": train["id"], "Label": train["Label"], "oof_score": result["scores"]["oof"], "oof_pred": result["preds"]["oof"]}).to_csv(out_dir / "oof_scores.csv", index=False)
    pd.DataFrame({"id": public["id"], "score": result["scores"]["pub"], "pred": result["preds"]["pub"]}).to_csv(out_dir / "public_scores.csv", index=False)
    pd.DataFrame({"id": private["id"], "score": result["scores"]["pri"], "pred": result["preds"]["pri"]}).to_csv(out_dir / "private_scores.csv", index=False)
    combo = pd.concat([
        pd.DataFrame({"id": public["id"], "Label": result["preds"]["pub"]}),
        pd.DataFrame({"id": private["id"], "Label": result["preds"]["pri"]}),
    ], ignore_index=True)
    sub = sample[["id"]].merge(combo, on="id", how="left")
    sub["Label"] = sub["Label"].astype(int)
    path = SUBMIT_DIR / f"next_step35b_{label}_submission.csv"
    sub.to_csv(path, index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)
    return {"label": label, "name": result["name"], "submission": str(path), "oof_qwk": result["oof_qwk"], "oof_lift_vs_step25": result["oof_lift_vs_step25"], "test_l1": result["test_l1"], "changed": result["public_changed"] + result["private_changed"]}


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
    base_scores = {
        "oof": train["step25_score"].to_numpy(float),
        "pub": public["step25_score"].to_numpy(float),
        "pri": private["step25_score"].to_numpy(float),
    }
    base_preds = {
        "oof": train["step25_pred"].astype(int).to_numpy(),
        "pub": public["step25_pred"].astype(int).to_numpy(),
        "pri": private["step25_pred"].astype(int).to_numpy(),
    }

    model_scores: dict[str, dict[str, np.ndarray]] = {}
    for c in CS:
        raw_scores = cv_rubric_scores(train, public, private, y, c)
        model_scores[f"rubric_lr_c{str(c).replace('.', 'p')}"] = align_scale(raw_scores, base_scores)

    results = []
    result_by_name = {}
    for lambd in THRESHOLD_LAMBDAS:
        reference = evaluate(f"step25_reference_retuned_l{lambd}".replace(".", "p"), "reference", 0.0, 0.0, lambd, base_scores, y, train_dist, train, base_preds)
        results.append(reference)
        result_by_name[reference["name"]] = reference
        for model_name, diag_scores in model_scores.items():
            c = float(model_name.rsplit("c", 1)[1].replace("p", "."))
            standalone = evaluate(f"{model_name}_standalone_l{lambd}".replace(".", "p"), "standalone", c, 1.0, lambd, diag_scores, y, train_dist, train, base_preds)
            results.append(standalone)
            result_by_name[standalone["name"]] = standalone
            for weight in BLEND_WEIGHTS:
                blended = {split: (1.0 - weight) * base_scores[split] + weight * diag_scores[split] for split in base_scores}
                name = f"step25_{model_name}_blend_w{weight:.4f}_l{lambd}".replace(".", "p")
                result = evaluate(name, "label_blend", c, weight, lambd, blended, y, train_dist, train, base_preds)
                results.append(result)
                result_by_name[name] = result

    rows = []
    for result in results:
        rows.append({k: v for k, v in result.items() if k not in {"scores", "preds", "thresholds"}} | {"thresholds": ",".join(f"{x:.6f}" for x in result["thresholds"]), "pub_dist": str(result["pub_dist"]), "pri_dist": str(result["pri_dist"])})
    candidates = pd.DataFrame(rows).sort_values(["oof_lift_vs_step25", "test_l1", "changed"], ascending=[False, True, True])
    candidates.to_csv(OUT / "candidates.csv", index=False)

    safe = candidates[
        candidates["kind"].eq("label_blend")
        & (candidates["oof_lift_vs_step25"] > 0)
        & (candidates["test_l1"] <= SAFE_L1_CAP)
        & ((candidates["public_changed"] + candidates["private_changed"]) <= MAX_TEST_CHANGED)
        & (candidates["improved"] >= candidates["worsened"])
        & (candidates["risky_promotions"] <= 2)
        & (candidates["core_demotions"] <= 2)
    ].sort_values(["oof_lift_vs_step25", "mean_delta_abs_error", "test_l1"], ascending=[False, False, True])
    safe.to_csv(OUT / "safe_candidates.csv", index=False)

    picks = []
    for label, (_, row) in zip(["best_safe", "second_safe"], safe.head(2).iterrows()):
        picks.append(write_submission(label, result_by_name[row["name"]], train, public, private, sample))

    summary = {"step25_oof_qwk": STEP25_OOF_QWK, "safe_l1_cap": SAFE_L1_CAP, "max_test_changed": MAX_TEST_CHANGED, "safe_candidates": int(len(safe)), "picks": picks}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    top = safe.head(20) if len(safe) else candidates.head(20)
    show_cols = ["name", "kind", "c", "weight", "threshold_lambda", "oof_qwk", "oof_lift_vs_step25", "test_l1", "changed", "improved", "worsened", "public_changed", "private_changed", "risky_promotions", "core_demotions"]
    lines = ["# Step35B rubric-conditioned reranker", ""]
    lines.append("This step trains a generalized rubric-conditioned binary reranker: each paper is paired with each of the five class rubrics, trained OOF on train labels only, converted into an expected label score, and blended lightly with Step25 for public/private uniformly.")
    lines.extend(["", "## Top safe candidates" if len(safe) else "## Top candidates — no safe candidate passed filters", ""])
    lines.append(top[show_cols].to_markdown(index=False))
    lines.extend(["", "## Picks", "", "```json", json.dumps(picks, indent=2, ensure_ascii=False), "```", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== Step35B rubric-conditioned reranker ===")
    print(top[show_cols].head(15).to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
