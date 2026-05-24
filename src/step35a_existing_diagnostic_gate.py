from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
STEP25_DIR = ROOT / "outputs" / "bge_m3_frozen_anchor" / "best_safe"
OUT = ROOT / "outputs" / "step35a_existing_diagnostic_gate"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

STEP25_OOF_QWK = 0.6612605583792329
SAFE_L1_CAP = 0.147
MAX_TEST_CHANGED = 15
BLEND_WEIGHTS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
RESID_WEIGHTS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
THRESHOLD_LAMBDAS = [0.5, 1.0, 2.0, 4.0]

SCORE_DIRS = {
    "modernbert_best": ROOT / "outputs" / "modernbert_reranker_diagnostic" / "best_safe",
    "modernbert_second": ROOT / "outputs" / "modernbert_reranker_diagnostic" / "second_safe",
    "pairwise_e5": ROOT / "outputs" / "pairwise_e5_reranker",
    "pairwise_specter2": ROOT / "outputs" / "specter2_pairwise_reranker",
    "gte_qwen_diag": ROOT / "outputs" / "gte_qwen_frozen_diagnostic" / "best_safe",
    "llm_zeroshot": ROOT / "outputs" / "llm_zeroshot",
    "llm_zeroshot_v2": ROOT / "outputs" / "llm_zeroshot_v2",
    "scibert_finetune": ROOT / "outputs" / "scibert_finetune",
    "deberta_v3": ROOT / "outputs" / "deberta_v3_finetune_outputs" / "deberta_v3_finetune",
    "openalex_anchor": ROOT / "outputs" / "openalex_anchor",
    "specter2_finetune": ROOT / "outputs" / "specter2_finetune",
}


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


def load_score_split(directory: Path, split: str, prefix: str) -> pd.DataFrame | None:
    file_name = "oof_scores.csv" if split == "train" else f"{split}_scores.csv"
    path = directory / file_name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    score_col = "oof_score" if "oof_score" in df.columns else "score" if "score" in df.columns else None
    pred_col = "oof_pred" if "oof_pred" in df.columns else "pred" if "pred" in df.columns else None
    if score_col is None:
        return None
    cols = ["id", score_col] + ([pred_col] if pred_col else [])
    out = df[cols].copy().rename(columns={score_col: f"{prefix}_score"})
    if pred_col:
        out = out.rename(columns={pred_col: f"{prefix}_pred"})
    return out


def load_score_family(name: str, directory: Path, raw: dict[str, pd.DataFrame]) -> dict[str, np.ndarray] | None:
    frames = {}
    for split in ["train", "public", "private"]:
        loaded = load_score_split(directory, split, name)
        if loaded is None:
            return None
        merged = raw[split][["id"]].merge(loaded, on="id", how="left")
        score_col = f"{name}_score"
        if merged[score_col].isna().any():
            return None
        frames[split] = merged[score_col].astype(float).to_numpy()
    return {"oof": frames["train"], "pub": frames["public"], "pri": frames["private"]}


def load_step25(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores = load_score_family("step25", STEP25_DIR, raw)
    if scores is None:
        raise FileNotFoundError(f"Missing Step25 score files in {STEP25_DIR}")
    preds = {}
    for split, key in [("train", "oof"), ("public", "pub"), ("private", "pri")]:
        loaded = load_score_split(STEP25_DIR, split, "step25")
        assert loaded is not None
        pred_col = "step25_pred"
        merged = raw[split][["id"]].merge(loaded, on="id", how="left")
        if pred_col not in merged:
            raise ValueError(f"Missing Step25 predictions for {split}")
        preds[key] = merged[pred_col].astype(int).to_numpy()
    return scores, preds


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
        sub = train.loc[changed].copy()
        title = sub["title"].fillna("").str.lower()
        venue = sub["venue"].fillna("").str.lower()
        coreish = title.str.contains("answer set|answer-set|asp\\b|stable model|clingo|dlv|grounding", regex=True) | venue.str.contains("iclp|lpnmr|logic programming", regex=True)
        risky = title.str.contains("verification|synthesis|model checking|temporal logic|automata|invariant|neural|machine learning", regex=True) | venue.str.contains("cav|lics|lpar", regex=True)
        risky_promotions = int(((new_pred[changed] > base_pred[changed]) & risky.to_numpy(bool) & ~coreish.to_numpy(bool)).sum())
        core_demotions = int(((new_pred[changed] < base_pred[changed]) & coreish.to_numpy(bool)).sum())
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


def evaluate(name: str, kind: str, weight: float, lambd: float, scores: dict[str, np.ndarray], y: np.ndarray, train_dist: np.ndarray, raw_train: pd.DataFrame, base_preds: dict[str, np.ndarray]) -> dict[str, Any]:
    thresholds, qwk = tune_thresholds(y, scores["oof"], train_dist, lambd=lambd)
    preds = {
        "oof": scores_to_labels(scores["oof"], thresholds),
        "pub": scores_to_labels(scores["pub"], thresholds),
        "pri": scores_to_labels(scores["pri"], thresholds),
    }
    test_pred = np.concatenate([preds["pub"], preds["pri"]])
    summary = change_summary(y, base_preds["oof"], preds["oof"], base_preds["pub"], preds["pub"], base_preds["pri"], preds["pri"], raw_train)
    return {
        "name": name,
        "kind": kind,
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


def write_submission(label: str, result: dict[str, Any], raw: dict[str, pd.DataFrame], sample: pd.DataFrame) -> dict[str, Any]:
    out_dir = OUT / label
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"id": raw["train"]["id"], "Label": raw["train"]["Label"], "oof_score": result["scores"]["oof"], "oof_pred": result["preds"]["oof"]}).to_csv(out_dir / "oof_scores.csv", index=False)
    pd.DataFrame({"id": raw["public"]["id"], "score": result["scores"]["pub"], "pred": result["preds"]["pub"]}).to_csv(out_dir / "public_scores.csv", index=False)
    pd.DataFrame({"id": raw["private"]["id"], "score": result["scores"]["pri"], "pred": result["preds"]["pri"]}).to_csv(out_dir / "private_scores.csv", index=False)
    combo = pd.concat([
        pd.DataFrame({"id": raw["public"]["id"], "Label": result["preds"]["pub"]}),
        pd.DataFrame({"id": raw["private"]["id"], "Label": result["preds"]["pri"]}),
    ], ignore_index=True)
    sub = sample[["id"]].merge(combo, on="id", how="left")
    sub["Label"] = sub["Label"].astype(int)
    path = SUBMIT_DIR / f"next_step35a_{label}_submission.csv"
    sub.to_csv(path, index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)
    return {"label": label, "name": result["name"], "submission": str(path), "oof_qwk": result["oof_qwk"], "oof_lift_vs_step25": result["oof_lift_vs_step25"], "test_l1": result["test_l1"], "changed": result["public_changed"] + result["private_changed"]}


def main() -> None:
    raw = {
        "train": pd.read_csv(DATA_DIR / "train.csv"),
        "public": pd.read_csv(DATA_DIR / "public_test.csv"),
        "private": pd.read_csv(DATA_DIR / "private_test.csv"),
    }
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")
    y = raw["train"]["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()
    base_scores, base_preds = load_step25(raw)

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for name, directory in SCORE_DIRS.items():
        scores = load_score_family(name, directory, raw)
        if scores is not None:
            loaded[name] = align_scale(scores, base_scores)

    results = []
    result_by_name = {}
    for lambd in THRESHOLD_LAMBDAS:
        reference = evaluate(f"step25_reference_retuned_l{lambd}".replace(".", "p"), "reference", 0.0, lambd, base_scores, y, train_dist, raw["train"], base_preds)
        results.append(reference)
        result_by_name[reference["name"]] = reference
        for diag_name, diag_scores in loaded.items():
            for weight in BLEND_WEIGHTS:
                blended = {split: (1.0 - weight) * base_scores[split] + weight * diag_scores[split] for split in base_scores}
                name = f"step25_{diag_name}_blend_w{weight:.4f}_l{lambd}".replace(".", "p")
                result = evaluate(name, "label_blend", weight, lambd, blended, y, train_dist, raw["train"], base_preds)
                results.append(result)
                result_by_name[name] = result
            residual = {split: diag_scores[split] - base_scores[split] for split in base_scores}
            for weight in RESID_WEIGHTS:
                corrected = {split: base_scores[split] + weight * residual[split] for split in base_scores}
                name = f"step25_{diag_name}_resid_w{weight:.4f}_l{lambd}".replace(".", "p")
                result = evaluate(name, "residual_blend", weight, lambd, corrected, y, train_dist, raw["train"], base_preds)
                results.append(result)
                result_by_name[name] = result

    rows = []
    for result in results:
        rows.append({k: v for k, v in result.items() if k not in {"scores", "preds", "thresholds"}} | {"thresholds": ",".join(f"{x:.6f}" for x in result["thresholds"]), "pub_dist": str(result["pub_dist"]), "pri_dist": str(result["pri_dist"])})
    candidates = pd.DataFrame(rows).sort_values(["oof_lift_vs_step25", "test_l1", "changed"], ascending=[False, True, True])
    candidates.to_csv(OUT / "candidates.csv", index=False)

    safe = candidates[
        candidates["kind"].isin(["label_blend", "residual_blend"])
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
        picks.append(write_submission(label, result_by_name[row["name"]], raw, sample))

    summary = {
        "step25_oof_qwk": STEP25_OOF_QWK,
        "safe_l1_cap": SAFE_L1_CAP,
        "max_test_changed": MAX_TEST_CHANGED,
        "loaded_diagnostics": sorted(loaded),
        "safe_candidates": int(len(safe)),
        "picks": picks,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    top = safe.head(20) if len(safe) else candidates.head(20)
    show_cols = ["name", "kind", "weight", "threshold_lambda", "oof_qwk", "oof_lift_vs_step25", "test_l1", "changed", "improved", "worsened", "public_changed", "private_changed", "risky_promotions", "core_demotions"]
    lines = ["# Step35A existing diagnostic gate", ""]
    lines.append("This step reuses existing OOF/public/private diagnostic score artifacts as lightweight rubric/pairwise-style judges over Step25. It applies one generalized blend rule uniformly to train/public/private and only writes submissions for candidates passing safety gates.")
    lines.extend(["", "## Loaded diagnostics", "", "```json", json.dumps(sorted(loaded), indent=2), "```"])
    lines.extend(["", "## Top safe candidates" if len(safe) else "## Top candidates — no safe candidate passed filters", ""])
    lines.append(top[show_cols].to_markdown(index=False))
    lines.extend(["", "## Picks", "", "```json", json.dumps(picks, indent=2, ensure_ascii=False), "```", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== Step35A existing diagnostic gate ===")
    print(top[show_cols].head(15).to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
