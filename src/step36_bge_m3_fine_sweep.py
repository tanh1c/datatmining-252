from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
BGE_DIR = ROOT / "outputs" / "bge_m3_frozen_anchor"
STEP18B_DIR = ROOT / "outputs" / "step18b_targeted_feature_calibration" / "blend_ridge_a10_w0p08"
STEP25_DIR = ROOT / "outputs" / "bge_m3_frozen_anchor" / "best_safe"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
OUT = ROOT / "outputs" / "step36_bge_m3_fine_sweep"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

STEP18B_OOF_QWK = 0.660176335745315
STEP25_OOF_QWK = 0.6612605583792329
SAFE_L1_CAP = 0.147
MAX_DIFF_VS_STEP25 = 5
MODEL_NAMES = ["ridge_a3p0", "ridge_a10p0", "ridge_a30p0", "ridge_a100p0", "ridge_a300p0", "ridge_a1000p0", "huber"]
WEIGHTS = [0.010, 0.0125, 0.015, 0.0175, 0.020, 0.0225, 0.025, 0.0275, 0.030, 0.0325, 0.035, 0.0375, 0.040]
THRESHOLD_LAMBDAS = [0.5, 1.0, 2.0, 4.0]
FORBIDDEN_DEMOTE_TO_ONE_IDS = {2022}


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


def load_feature_split(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    ids = pd.read_csv(BGE_DIR / f"{split}_ids.csv")
    dense = np.load(BGE_DIR / f"{split}_dense.npy")
    diag_path = BGE_DIR / f"{split}_diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        ids = ids.merge(diag, on="id", how="left")
    return ids, dense


def make_matrix(ids: pd.DataFrame, dense: np.ndarray) -> np.ndarray:
    diag_cols = [c for c in ids.columns if c.startswith("bge_m3_dense_") or c.startswith("bge_m3_sparse_")]
    if diag_cols:
        diag = ids[diag_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
        return np.hstack([dense.astype(np.float32), diag])
    return dense.astype(np.float32)


def load_scores(split: str, directory: Path, prefix: str) -> pd.DataFrame:
    file_name = "oof_scores.csv" if split == "train" else f"{split}_scores.csv"
    df = pd.read_csv(directory / file_name)
    score_col = "oof_score" if "oof_score" in df.columns else "score"
    pred_col = "oof_pred" if "oof_pred" in df.columns else "pred"
    return df[["id", score_col, pred_col]].rename(columns={score_col: f"{prefix}_score", pred_col: f"{prefix}_pred"})


def model_factory(model_name: str):
    if model_name.startswith("ridge_a"):
        alpha = float(model_name.split("a", 1)[1].replace("p", "."))
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if model_name == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.05, max_iter=500))
    raise ValueError(model_name)


def cv_predict(model_name: str, x: np.ndarray, y: np.ndarray, pub_x: np.ndarray, pri_x: np.ndarray) -> dict[str, np.ndarray]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=252)
    oof = np.zeros(len(x), dtype=float)
    pub_preds = []
    pri_preds = []
    for tr_idx, va_idx in folds.split(x, y):
        model = model_factory(model_name)
        model.fit(x[tr_idx], y[tr_idx].astype(float))
        oof[va_idx] = model.predict(x[va_idx])
        pub_preds.append(model.predict(pub_x))
        pri_preds.append(model.predict(pri_x))
    return {"oof": oof, "pub": np.mean(pub_preds, axis=0), "pri": np.mean(pri_preds, axis=0)}


def context_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    title = out["title"].fillna("").str.lower()
    venue = out["venue"].fillna("").str.lower()
    out["risky_context"] = venue.str.contains("kr|cav|lics|lpar", regex=True) | title.str.contains("verification|planning|reason|logic|invariant|explain|abduction|argumentation", regex=True)
    out["coreish_context"] = venue.str.contains("iclp|lpnmr", regex=True) | title.str.contains("answer set|asp|stable model|clingo|dlv|grounding", regex=True)
    return out


def change_summary(y: np.ndarray, train: pd.DataFrame, base_pred: np.ndarray, new_pred: np.ndarray, pub_base: np.ndarray, pub_new: np.ndarray, pri_base: np.ndarray, pri_new: np.ndarray, public_ids: np.ndarray, private_ids: np.ndarray) -> dict[str, Any]:
    changed = base_pred != new_pred
    if changed.any():
        delta_abs_error = np.abs(base_pred[changed] - y[changed]) - np.abs(new_pred[changed] - y[changed])
        changed_train = train.loc[changed]
        risky_promotions = int(((new_pred[changed] > base_pred[changed]) & changed_train["risky_context"].to_numpy(bool)).sum())
        core_demotions = int(((new_pred[changed] < base_pred[changed]) & changed_train["coreish_context"].to_numpy(bool)).sum())
    else:
        delta_abs_error = np.array([])
        risky_promotions = 0
        core_demotions = 0
    pub_changed_mask = pub_base != pub_new
    pri_changed_mask = pri_base != pri_new
    forbidden = 0
    if pub_changed_mask.any():
        forbidden += int(sum((int(i) in FORBIDDEN_DEMOTE_TO_ONE_IDS) and int(n) == 1 for i, n in zip(public_ids[pub_changed_mask], pub_new[pub_changed_mask])))
    if pri_changed_mask.any():
        forbidden += int(sum((int(i) in FORBIDDEN_DEMOTE_TO_ONE_IDS) and int(n) == 1 for i, n in zip(private_ids[pri_changed_mask], pri_new[pri_changed_mask])))
    return {
        "changed": int(changed.sum()),
        "improved": int((delta_abs_error > 0).sum()) if len(delta_abs_error) else 0,
        "worsened": int((delta_abs_error < 0).sum()) if len(delta_abs_error) else 0,
        "mean_delta_abs_error": float(delta_abs_error.mean()) if len(delta_abs_error) else 0.0,
        "risky_promotions": risky_promotions,
        "core_demotions": core_demotions,
        "public_changed": int(pub_changed_mask.sum()),
        "private_changed": int(pri_changed_mask.sum()),
        "forbidden_changes": forbidden,
    }


def evaluate(name: str, model_name: str, weight: float, lambd: float, scores: dict[str, np.ndarray], y: np.ndarray, train_dist: np.ndarray, train: pd.DataFrame, step18_preds: dict[str, np.ndarray], step25_preds: dict[str, np.ndarray], ids: dict[str, np.ndarray]) -> dict[str, Any]:
    thresholds, qwk = tune_thresholds(y, scores["oof"], train_dist, lambd=lambd)
    preds = {"oof": scores_to_labels(scores["oof"], thresholds), "pub": scores_to_labels(scores["pub"], thresholds), "pri": scores_to_labels(scores["pri"], thresholds)}
    test_pred = np.concatenate([preds["pub"], preds["pri"]])
    vs18 = change_summary(y, train, step18_preds["oof"], preds["oof"], step18_preds["pub"], preds["pub"], step18_preds["pri"], preds["pri"], ids["pub"], ids["pri"])
    diff_vs25_pub = int((preds["pub"] != step25_preds["pub"]).sum())
    diff_vs25_pri = int((preds["pri"] != step25_preds["pri"]).sum())
    forbidden_vs25 = 0
    for split, key in [("pub", "pub"), ("pri", "pri")]:
        changed = preds[key] != step25_preds[key]
        forbidden_vs25 += int(sum((int(i) in FORBIDDEN_DEMOTE_TO_ONE_IDS) and int(n) == 1 for i, n in zip(ids[key][changed], preds[key][changed])))
    return {
        "name": name,
        "model": model_name,
        "weight": weight,
        "threshold_lambda": lambd,
        "oof_qwk": qwk,
        "oof_lift_vs_step18b": float(qwk - STEP18B_OOF_QWK),
        "oof_lift_vs_step25": float(qwk - STEP25_OOF_QWK),
        "test_l1": float(np.sum(np.abs(predicted_dist(test_pred) - train_dist))),
        "diff_vs_step25_public": diff_vs25_pub,
        "diff_vs_step25_private": diff_vs25_pri,
        "diff_vs_step25_total": diff_vs25_pub + diff_vs25_pri,
        "forbidden_vs_step25": forbidden_vs25,
        "thresholds": thresholds.tolist(),
        "scores": scores,
        "preds": preds,
        "pub_dist": {int(k): int(v) for k, v in pd.Series(preds["pub"]).value_counts().sort_index().items()},
        "pri_dist": {int(k): int(v) for k, v in pd.Series(preds["pri"]).value_counts().sort_index().items()},
        **vs18,
    }


def write_pick(label: str, result: dict[str, Any], raw: dict[str, pd.DataFrame], sample: pd.DataFrame, y: np.ndarray, step25_preds: dict[str, np.ndarray]) -> dict[str, Any]:
    out_dir = OUT / label
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"id": raw["train"]["id"], "Label": y, "oof_score": result["scores"]["oof"], "oof_pred": result["preds"]["oof"]}).to_csv(out_dir / "oof_scores.csv", index=False)
    pd.DataFrame({"id": raw["public"]["id"], "score": result["scores"]["pub"], "pred": result["preds"]["pub"]}).to_csv(out_dir / "public_scores.csv", index=False)
    pd.DataFrame({"id": raw["private"]["id"], "score": result["scores"]["pri"], "pred": result["preds"]["pri"]}).to_csv(out_dir / "private_scores.csv", index=False)
    for split, key in [("public", "pub"), ("private", "pri")]:
        changed = result["preds"][key] != step25_preds[key]
        raw[split].loc[changed, ["id", "title", "venue"]].assign(step25_pred=step25_preds[key][changed], step36_pred=result["preds"][key][changed]).to_csv(out_dir / f"changed_vs_step25_{split}.csv", index=False)
    combo = pd.concat([
        pd.DataFrame({"id": raw["public"]["id"], "Label": result["preds"]["pub"]}),
        pd.DataFrame({"id": raw["private"]["id"], "Label": result["preds"]["pri"]}),
    ], ignore_index=True)
    sub = sample[["id"]].merge(combo, on="id", how="left")
    sub["Label"] = sub["Label"].astype(int)
    path = SUBMIT_DIR / f"next_step36_{label}_submission.csv"
    sub.to_csv(path, index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)
    return {"label": label, "name": result["name"], "submission": str(path), "oof_qwk": result["oof_qwk"], "oof_lift_vs_step25": result["oof_lift_vs_step25"], "test_l1": result["test_l1"], "diff_vs_step25_total": result["diff_vs_step25_total"]}


def main() -> None:
    raw = {"train": pd.read_csv(DATA_DIR / "train.csv"), "public": pd.read_csv(DATA_DIR / "public_test.csv"), "private": pd.read_csv(DATA_DIR / "private_test.csv")}
    raw["train"] = context_flags(raw["train"])
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")
    y = raw["train"]["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    train_ids, train_dense = load_feature_split("train")
    pub_ids, pub_dense = load_feature_split("public")
    pri_ids, pri_dense = load_feature_split("private")
    x = make_matrix(train_ids, train_dense)
    pub_x = make_matrix(pub_ids, pub_dense)
    pri_x = make_matrix(pri_ids, pri_dense)

    step18_frames = {split: raw[split][["id"]].merge(load_scores(split, STEP18B_DIR, "step18b"), on="id", how="left") for split in ["train", "public", "private"]}
    step25_frames = {split: raw[split][["id"]].merge(load_scores(split, STEP25_DIR, "step25"), on="id", how="left") for split in ["train", "public", "private"]}
    step18_scores = {"oof": step18_frames["train"]["step18b_score"].to_numpy(float), "pub": step18_frames["public"]["step18b_score"].to_numpy(float), "pri": step18_frames["private"]["step18b_score"].to_numpy(float)}
    step18_preds = {"oof": step18_frames["train"]["step18b_pred"].astype(int).to_numpy(), "pub": step18_frames["public"]["step18b_pred"].astype(int).to_numpy(), "pri": step18_frames["private"]["step18b_pred"].astype(int).to_numpy()}
    step25_preds = {"oof": step25_frames["train"]["step25_pred"].astype(int).to_numpy(), "pub": step25_frames["public"]["step25_pred"].astype(int).to_numpy(), "pri": step25_frames["private"]["step25_pred"].astype(int).to_numpy()}
    ids = {"pub": raw["public"]["id"].astype(int).to_numpy(), "pri": raw["private"]["id"].astype(int).to_numpy()}

    model_scores = {model_name: cv_predict(model_name, x, y, pub_x, pri_x) for model_name in MODEL_NAMES}
    results = []
    result_by_name = {}
    for model_name, bge_scores in model_scores.items():
        for weight in WEIGHTS:
            for lambd in THRESHOLD_LAMBDAS:
                scores = {split: (1.0 - weight) * step18_scores[split] + weight * bge_scores[split] for split in step18_scores}
                name = f"step18b_bge_m3_{model_name}_w{weight:.4f}_l{lambd}".replace(".", "p")
                result = evaluate(name, model_name, weight, lambd, scores, y, train_dist, raw["train"], step18_preds, step25_preds, ids)
                results.append(result)
                result_by_name[name] = result

    rows = []
    for result in results:
        rows.append({k: v for k, v in result.items() if k not in {"scores", "preds", "thresholds"}} | {"thresholds": ",".join(f"{x:.6f}" for x in result["thresholds"]), "pub_dist": str(result["pub_dist"]), "pri_dist": str(result["pri_dist"])})
    candidates = pd.DataFrame(rows).sort_values(["oof_lift_vs_step25", "test_l1", "diff_vs_step25_total"], ascending=[False, True, True])
    candidates.to_csv(OUT / "candidates.csv", index=False)

    safe = candidates[
        (candidates["oof_lift_vs_step25"] > 0)
        & (candidates["test_l1"] <= SAFE_L1_CAP)
        & (candidates["improved"] > candidates["worsened"])
        & (candidates["risky_promotions"] <= 5)
        & (candidates["core_demotions"] <= 2)
        & (candidates["diff_vs_step25_total"] <= MAX_DIFF_VS_STEP25)
        & (candidates["forbidden_vs_step25"] == 0)
    ].sort_values(["oof_lift_vs_step25", "mean_delta_abs_error", "diff_vs_step25_total"], ascending=[False, False, True])
    safe.to_csv(OUT / "safe_candidates.csv", index=False)

    picks = []
    for label, (_, row) in zip(["best_safe", "second_safe"], safe.head(2).iterrows()):
        picks.append(write_pick(label, result_by_name[row["name"]], raw, sample, y, step25_preds))

    summary = {"step25_oof_qwk": STEP25_OOF_QWK, "safe_l1_cap": SAFE_L1_CAP, "max_diff_vs_step25": MAX_DIFF_VS_STEP25, "safe_candidates": int(len(safe)), "picks": picks}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    top = safe.head(20) if len(safe) else candidates.head(20)
    show_cols = ["name", "model", "weight", "threshold_lambda", "oof_qwk", "oof_lift_vs_step25", "test_l1", "changed", "improved", "worsened", "risky_promotions", "core_demotions", "public_changed", "private_changed", "diff_vs_step25_total", "forbidden_vs_step25"]
    lines = ["# Step36 BGE-M3 fine sweep around Step25", ""]
    lines.append("This step reruns the BGE-M3 frozen diagnostic stack with fine-grained weights around the validated Step25 region, then filters candidates by OOF lift, test distribution safety, and small diff versus the current Step25 public-best submission.")
    lines.extend(["", "## Top safe candidates" if len(safe) else "## Top candidates — no safe candidate passed filters", ""])
    lines.append(top[show_cols].to_markdown(index=False))
    lines.extend(["", "## Picks", "", "```json", json.dumps(picks, indent=2, ensure_ascii=False), "```", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("=== Step36 BGE-M3 fine sweep ===")
    print(top[show_cols].head(15).to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
