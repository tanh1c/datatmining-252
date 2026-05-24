from __future__ import annotations

import json
from pathlib import Path

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
OUT = ROOT / "outputs" / "bge_m3_frozen_anchor"
STEP18B_DIR = ROOT / "outputs" / "step18b_targeted_feature_calibration" / "blend_ridge_a10_w0p08"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

STEP18B_OOF_QWK = 0.660176335745315
SAFE_L1_CAP = 0.147
ALPHAS = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
BLEND_WEIGHTS = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12]


def scores_to_labels(scores: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def predicted_dist(labels: np.ndarray) -> np.ndarray:
    return pd.Series(labels).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, train_dist: np.ndarray, lambd: float = 0.5, seed: int = 42):
    def objective(raw: np.ndarray) -> float:
        thr = np.sort(raw)
        gap = np.min(np.diff(thr))
        gap_pen = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(scores, thr)
        qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
        dist_pen = float(np.sum(np.abs(predicted_dist(labels) - train_dist)))
        return -qwk + gap_pen + lambd * dist_pen

    res = differential_evolution(
        objective,
        [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed,
        maxiter=120,
        popsize=15,
        polish=True,
        updating="immediate",
        workers=1,
    )
    thr = np.sort(res.x)
    return thr, float(cohen_kappa_score(y_true, scores_to_labels(scores, thr), weights="quadratic"))


def require_features() -> None:
    missing = [
        path for path in [
            OUT / "train_dense.npy",
            OUT / "public_dense.npy",
            OUT / "private_dense.npy",
            OUT / "train_ids.csv",
            OUT / "public_ids.csv",
            OUT / "private_ids.csv",
        ] if not path.exists()
    ]
    if missing:
        lines = ["Missing BGE-M3 frozen feature files:"]
        lines.extend(f"- {path}" for path in missing)
        lines.append("Run notebooks/step25_bge_m3_frozen_anchor.ipynb first, then rerun this script.")
        raise FileNotFoundError("\n".join(lines))


def load_feature_split(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    ids = pd.read_csv(OUT / f"{split}_ids.csv")
    dense = np.load(OUT / f"{split}_dense.npy")
    diag_path = OUT / f"{split}_diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        ids = ids.merge(diag, on="id", how="left")
    return ids, dense


def load_step18b(split: str) -> pd.DataFrame:
    file_name = {"train": "oof_scores.csv", "public": "public_scores.csv", "private": "private_scores.csv"}[split]
    df = pd.read_csv(STEP18B_DIR / file_name)
    score_col = "oof_score" if "oof_score" in df.columns else "score"
    pred_col = "oof_pred" if "oof_pred" in df.columns else "pred"
    return df[["id", score_col, pred_col]].rename(columns={score_col: "step18b_score", pred_col: "step18b_pred"})


def make_matrix(ids: pd.DataFrame, dense: np.ndarray) -> np.ndarray:
    diag_cols = [c for c in ids.columns if c.startswith("bge_m3_dense_") or c.startswith("bge_m3_sparse_")]
    if diag_cols:
        diag = ids[diag_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
        return np.hstack([dense.astype(np.float32), diag])
    return dense.astype(np.float32)


def cv_predict(model_name: str, x: np.ndarray, y: np.ndarray, pub_x: np.ndarray, pri_x: np.ndarray) -> dict:
    if model_name.startswith("ridge_a"):
        alpha = float(model_name.split("a", 1)[1].replace("p", "."))
        factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    elif model_name == "huber":
        factory = lambda: make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.05, max_iter=500))
    else:
        raise ValueError(model_name)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=252)
    oof = np.zeros(len(x), dtype=float)
    pub_preds = []
    pri_preds = []
    for tr_idx, va_idx in folds.split(x, y):
        model = factory()
        model.fit(x[tr_idx], y[tr_idx].astype(float))
        oof[va_idx] = model.predict(x[va_idx])
        pub_preds.append(model.predict(pub_x))
        pri_preds.append(model.predict(pri_x))
    return {"oof": oof, "pub": np.mean(pub_preds, axis=0), "pri": np.mean(pri_preds, axis=0)}


def evaluate_scores(name: str, scores: dict[str, np.ndarray], y: np.ndarray, train_dist: np.ndarray) -> dict:
    thresholds, qwk = tune_thresholds(y, scores["oof"], train_dist)
    oof_pred = scores_to_labels(scores["oof"], thresholds)
    pub_pred = scores_to_labels(scores["pub"], thresholds)
    pri_pred = scores_to_labels(scores["pri"], thresholds)
    test_pred = np.concatenate([pub_pred, pri_pred])
    return {
        "name": name,
        "thresholds": thresholds.tolist(),
        "oof_qwk": qwk,
        "oof_lift": float(qwk - STEP18B_OOF_QWK),
        "oof_round_qwk": float(cohen_kappa_score(y, np.clip(np.round(scores["oof"]), 1, 5).astype(int), weights="quadratic")),
        "test_l1": float(np.sum(np.abs(predicted_dist(test_pred) - train_dist))),
        "scores": scores,
        "preds": {"oof": oof_pred, "pub": pub_pred, "pri": pri_pred},
        "pub_dist": {int(k): int(v) for k, v in pd.Series(pub_pred).value_counts().sort_index().items()},
        "pri_dist": {int(k): int(v) for k, v in pd.Series(pri_pred).value_counts().sort_index().items()},
    }


def risky_change_summary(train_meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray) -> dict:
    changed = base_pred != new_pred
    if not changed.any():
        return {"changed": 0, "improved": 0, "worsened": 0, "same": 0, "mean_delta_abs_error": 0.0, "risky_promotions": 0, "coreish_demotions_5_to_4": 0}
    delta = np.abs(base_pred[changed] - y[changed]) - np.abs(new_pred[changed] - y[changed])
    sub = train_meta.loc[changed, ["venue", "title"]].copy()
    old = base_pred[changed]
    new = new_pred[changed]
    venue = sub["venue"].fillna("").str.lower()
    title = sub["title"].fillna("").str.lower()
    risky = venue.str.contains("kr|cav|lics|lpar", regex=True) | title.str.contains("verification|planning|reason|logic|invariant|explain|abduction|argumentation", regex=True)
    coreish = venue.str.contains("iclp|lpnmr", regex=True) | title.str.contains("answer set|asp|stable model|clingo|dlv|grounding", regex=True)
    risky_promotions = sum((int(a), int(b)) in {(1, 2), (2, 3), (3, 4)} and bool(ctx) for a, b, ctx in zip(old, new, risky))
    coreish_demotions = sum(int(a) == 5 and int(b) == 4 and bool(ctx) for a, b, ctx in zip(old, new, coreish))
    return {
        "changed": int(changed.sum()),
        "improved": int((delta > 0).sum()),
        "worsened": int((delta < 0).sum()),
        "same": int((delta == 0).sum()),
        "mean_delta_abs_error": float(delta.mean()),
        "risky_promotions": int(risky_promotions),
        "coreish_demotions_5_to_4": int(coreish_demotions),
    }


def write_pick(label: str, result: dict, train_ids: pd.DataFrame, pub_ids: pd.DataFrame, pri_ids: pd.DataFrame, sample: pd.DataFrame, y: np.ndarray) -> dict:
    out_dir = OUT / label
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"id": train_ids["id"], "Label": y, "oof_score": result["scores"]["oof"], "oof_pred": result["preds"]["oof"]}).to_csv(out_dir / "oof_scores.csv", index=False)
    pd.DataFrame({"id": pub_ids["id"], "score": result["scores"]["pub"], "pred": result["preds"]["pub"]}).to_csv(out_dir / "public_scores.csv", index=False)
    pd.DataFrame({"id": pri_ids["id"], "score": result["scores"]["pri"], "pred": result["preds"]["pri"]}).to_csv(out_dir / "private_scores.csv", index=False)
    combo = pd.concat([
        pd.DataFrame({"id": pub_ids["id"], "Label": result["preds"]["pub"]}),
        pd.DataFrame({"id": pri_ids["id"], "Label": result["preds"]["pri"]}),
    ], ignore_index=True)
    sub = sample[["id"]].merge(combo, on="id", how="left")
    sub["Label"] = sub["Label"].astype(int)
    sub.to_csv(out_dir / "submission.csv", index=False)
    sub_path = SUBMIT_DIR / f"next_step25_bge_m3_{label}_submission.csv"
    sub.to_csv(sub_path, index=False)
    return {
        "label": label,
        "name": result["name"],
        "oof_qwk": result["oof_qwk"],
        "oof_lift": result["oof_lift"],
        "test_l1": result["test_l1"],
        "thresholds": result["thresholds"],
        "submission": str(sub_path),
    }


def main() -> None:
    require_features()
    train = pd.read_csv(DATA_DIR / "train.csv")
    public = pd.read_csv(DATA_DIR / "public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test.csv")
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")
    y = train["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    train_ids, train_dense = load_feature_split("train")
    pub_ids, pub_dense = load_feature_split("public")
    pri_ids, pri_dense = load_feature_split("private")
    for name, ids, raw in [("train", train_ids, train), ("public", pub_ids, public), ("private", pri_ids, private)]:
        if not ids["id"].equals(raw["id"]):
            raise ValueError(f"{name} ids are not aligned with raw data")

    x = make_matrix(train_ids, train_dense)
    pub_x = make_matrix(pub_ids, pub_dense)
    pri_x = make_matrix(pri_ids, pri_dense)
    print(f"feature matrix: train={x.shape} public={pub_x.shape} private={pri_x.shape}")

    step18_train = load_step18b("train")
    step18_pub = load_step18b("public")
    step18_pri = load_step18b("private")
    train_step = train[["id"]].merge(step18_train, on="id", how="left")
    pub_step = public[["id"]].merge(step18_pub, on="id", how="left")
    pri_step = private[["id"]].merge(step18_pri, on="id", how="left")
    step_scores = {
        "oof": train_step["step18b_score"].to_numpy(float),
        "pub": pub_step["step18b_score"].to_numpy(float),
        "pri": pri_step["step18b_score"].to_numpy(float),
    }
    step_pred = train_step["step18b_pred"].astype(int).to_numpy()

    model_names = [f"ridge_a{str(a).replace('.', 'p')}" for a in ALPHAS] + ["huber"]
    base_results = []
    model_scores = {}
    for model_name in model_names:
        scores = cv_predict(model_name, x, y, pub_x, pri_x)
        model_scores[model_name] = scores
        base_results.append(evaluate_scores(f"bge_m3_{model_name}", scores, y, train_dist))

    rows = []
    results_by_name = {}
    for result in base_results:
        change = risky_change_summary(train, y, step_pred, result["preds"]["oof"])
        pub_changed = int((pub_step["step18b_pred"].astype(int).to_numpy() != result["preds"]["pub"]).sum())
        pri_changed = int((pri_step["step18b_pred"].astype(int).to_numpy() != result["preds"]["pri"]).sum())
        rows.append({
            "name": result["name"],
            "kind": "standalone",
            "weight": 1.0,
            "oof_qwk": result["oof_qwk"],
            "oof_lift": result["oof_lift"],
            "oof_round_qwk": result["oof_round_qwk"],
            "test_l1": result["test_l1"],
            "public_changed": pub_changed,
            "private_changed": pri_changed,
            "thresholds": ",".join(f"{t:.6f}" for t in result["thresholds"]),
            "pub_dist": str(result["pub_dist"]),
            "pri_dist": str(result["pri_dist"]),
            **change,
        })
        results_by_name[result["name"]] = result

    for model_name, scores in model_scores.items():
        for w in BLEND_WEIGHTS:
            blend = {k: (1.0 - w) * step_scores[k] + w * scores[k] for k in step_scores}
            result = evaluate_scores(f"step18b_bge_m3_{model_name}_w{w:.2f}".replace(".", "p"), blend, y, train_dist)
            change = risky_change_summary(train, y, step_pred, result["preds"]["oof"])
            pub_changed = int((pub_step["step18b_pred"].astype(int).to_numpy() != result["preds"]["pub"]).sum())
            pri_changed = int((pri_step["step18b_pred"].astype(int).to_numpy() != result["preds"]["pri"]).sum())
            rows.append({
                "name": result["name"],
                "kind": "step18b_blend",
                "weight": w,
                "oof_qwk": result["oof_qwk"],
                "oof_lift": result["oof_lift"],
                "oof_round_qwk": result["oof_round_qwk"],
                "test_l1": result["test_l1"],
                "public_changed": pub_changed,
                "private_changed": pri_changed,
                "thresholds": ",".join(f"{t:.6f}" for t in result["thresholds"]),
                "pub_dist": str(result["pub_dist"]),
                "pri_dist": str(result["pri_dist"]),
                **change,
            })
            results_by_name[result["name"]] = result

    candidates = pd.DataFrame(rows).sort_values(["oof_lift", "test_l1"], ascending=[False, True])
    candidates.to_csv(OUT / "candidates.csv", index=False)
    corr_df = pd.DataFrame({"step18b": step_scores["oof"], **{name: scores["oof"] for name, scores in model_scores.items()}}).corr()
    corr_df.to_csv(OUT / "correlation_matrix.csv")

    safe = candidates[
        (candidates["kind"].eq("step18b_blend"))
        & (candidates["weight"] > 0)
        & (candidates["test_l1"] <= SAFE_L1_CAP)
        & (candidates["oof_lift"] > 0)
        & (candidates["changed"] <= 80)
        & (candidates["improved"] > candidates["worsened"])
        & (candidates["risky_promotions"] <= 5)
        & (candidates["coreish_demotions_5_to_4"] <= 2)
    ].sort_values(["oof_lift", "mean_delta_abs_error", "changed"], ascending=[False, False, True])

    top = safe.head(30) if len(safe) else candidates.head(30)
    top.to_csv(OUT / "top_candidates.csv", index=False)

    report = ["# Step25 BGE-M3 frozen anchor stack\n"]
    report.append("BGE-M3 dense plus rubric diagnostic features, trained with low-variance Ridge/Huber OOF models and small Step18b blend weights.\n")
    report.append("## Correlation matrix\n")
    report.append(corr_df.to_markdown())
    report.append("\n## Top candidates\n")
    cols = ["name", "kind", "weight", "oof_qwk", "oof_lift", "oof_round_qwk", "test_l1", "changed", "improved", "worsened", "risky_promotions", "coreish_demotions_5_to_4", "public_changed", "private_changed"]
    report.append(top[[c for c in cols if c in top.columns]].to_markdown(index=False))
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")

    picks = []
    if len(safe):
        for label, (_, row) in zip(["best_safe", "second_safe"], safe.head(2).iterrows()):
            picks.append(write_pick(label, results_by_name[row["name"]], train_ids, pub_ids, pri_ids, sample, y))
    (OUT / "picks_summary.json").write_text(json.dumps({"step18b_oof_qwk": STEP18B_OOF_QWK, "safe_l1_cap": SAFE_L1_CAP, "picks": picks}, indent=2), encoding="utf-8")

    print("=== Step25 BGE-M3 frozen anchor ===")
    print(top[[c for c in cols if c in top.columns]].head(15).to_string(index=False))
    print("\n=== Picks ===")
    print(json.dumps(picks, indent=2))


if __name__ == "__main__":
    main()
