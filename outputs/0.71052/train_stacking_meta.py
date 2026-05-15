"""Step 5 — Stacking the 4 public-validated anchors into a meta-model.

Anchors:
  ridge_5x5  : Ridge TF-IDF anchor (lexical, OOF QWK 0.5967, public 0.62820)
  specter_3s : SPECTER2 fine-tune 3 seeds (semantic, OOF 0.6389, public 0.69972)  <-- primary
  specter_5s : SPECTER2 fine-tune 5 seeds (semantic, OOF 0.6445, public 0.68737)
  specter_v2 : SPECTER2 fine-tune v3+384 (semantic, OOF 0.6446, public 0.68718)

Pipeline:
  1. Load OOF + test scores per anchor, align by id.
  2. Pearson correlation matrix to verify diversity.
  3. Train 4 candidate meta-models on OOF (5-fold CV inside meta to get
     unbiased meta OOF):
       - simple_avg   : mean of 4 scores
       - weighted_avg : weights from per-anchor OOF QWK
       - ridge_meta   : Ridge(alpha=1.0)
       - huber_meta   : HuberRegressor
       - constrained_blend : scipy.optimize, weights >=0, sum=1
  4. Distribution-constrained threshold tuning (lambda=0.5).
  5. Save artefacts under outputs/stacking/.

Output:
  outputs/stacking/<candidate>_oof.csv
  outputs/stacking/<candidate>_test_scores.csv
  outputs/stacking/<candidate>_submission.csv
  outputs/stacking/stacking_report.csv
  outputs/stacking/correlation_matrix.csv
  outputs/stacking/best_submission.csv (symlink of the best candidate)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "stacking"
OUT.mkdir(parents=True, exist_ok=True)
LABEL = "Label"


# ---------------------------------------------------------------------------
# Anchor loaders
# ---------------------------------------------------------------------------


@dataclass
class Anchor:
    name: str
    oof: pd.DataFrame   # columns: id, oof_score
    public: pd.DataFrame  # columns: id, score
    private: pd.DataFrame  # columns: id, score


def load_ridge_anchor() -> Anchor:
    base = ROOT / "outputs" / "ridge_threshold" / "reports"
    oof = pd.read_csv(base / "ridge_threshold_5fold_5seed_891e6ee4_oof.csv")
    public = pd.read_csv(base / "ridge_threshold_5fold_5seed_891e6ee4_public_scores.csv")
    private = pd.read_csv(base / "ridge_threshold_5fold_5seed_891e6ee4_private_scores.csv")
    oof = oof.rename(columns={"oof_score": "oof_score"})[["id", "oof_score"]]
    public = public.rename(columns={"score": "score"})[["id", "score"]]
    private = private.rename(columns={"score": "score"})[["id", "score"]]
    return Anchor("ridge_5x5", oof, public, private)


def load_specter_anchor(folder: str, name: str) -> Anchor:
    base = ROOT / "outputs" / folder
    oof = pd.read_csv(base / "oof_scores.csv")[["id", "oof_score"]]
    public = pd.read_csv(base / "public_scores.csv")[["id", "score"]]
    private = pd.read_csv(base / "private_scores.csv")[["id", "score"]]
    return Anchor(name, oof, public, private)


def merge_anchors(anchors: list[Anchor], train: pd.DataFrame,
                  public: pd.DataFrame, private: pd.DataFrame):
    # OOF is by train id
    train_meta = train[["id", LABEL]].copy()
    for a in anchors:
        col = a.oof.rename(columns={"oof_score": f"{a.name}_oof"})
        train_meta = train_meta.merge(col, on="id", how="left")
    public_meta = public[["id"]].copy()
    for a in anchors:
        col = a.public.rename(columns={"score": f"{a.name}_score"})
        public_meta = public_meta.merge(col, on="id", how="left")
    private_meta = private[["id"]].copy()
    for a in anchors:
        col = a.private.rename(columns={"score": f"{a.name}_score"})
        private_meta = private_meta.merge(col, on="id", how="left")
    return train_meta, public_meta, private_meta


# ---------------------------------------------------------------------------
# Threshold tuning (distribution-constrained, per L1+L6)
# ---------------------------------------------------------------------------


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def tune_thresholds_constrained(y_true: np.ndarray, oof_scores: np.ndarray,
                                 train_dist: np.ndarray, lambd: float = 0.5,
                                 seed: int = 42):
    def objective(raw):
        thr = np.sort(raw)
        gap = np.min(np.diff(thr))
        gap_pen = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(oof_scores, thr)
        qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
        pred_dist = pd.Series(labels).value_counts(normalize=True).reindex(
            [1, 2, 3, 4, 5], fill_value=0).to_numpy()
        dist_pen = float(np.sum(np.abs(pred_dist - train_dist)))
        return -qwk + gap_pen + lambd * dist_pen

    bounds = [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)]
    res = differential_evolution(objective, bounds, seed=seed, maxiter=120,
                                 popsize=15, polish=True, updating="immediate",
                                 workers=1)
    thr = np.sort(res.x)
    qwk = cohen_kappa_score(y_true, scores_to_labels(oof_scores, thr),
                            weights="quadratic")
    return thr, float(qwk)


# ---------------------------------------------------------------------------
# Meta-model candidates
# ---------------------------------------------------------------------------


def meta_simple_avg(X_train, y_train, X_test):
    return X_train.mean(axis=1), X_test.mean(axis=1)


def meta_weighted_by_qwk(X_train, y_train, X_test, weights):
    """Weights from per-anchor OOF QWK (computed externally)."""
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return X_train @ w, X_test @ w


def meta_ridge(X_train, y_train, X_test, alpha=1.0, n_folds=5, seed=42):
    """Train Ridge with K-fold CV to produce out-of-fold meta scores, then refit
    on all train data for test scoring."""
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train), dtype=np.float64)
    for tr, va in cv.split(X_train):
        m = Ridge(alpha=alpha)
        m.fit(X_train[tr], y_train[tr])
        oof[va] = m.predict(X_train[va])
    final = Ridge(alpha=alpha)
    final.fit(X_train, y_train)
    return np.clip(oof, 1.0, 5.0), np.clip(final.predict(X_test), 1.0, 5.0), final.coef_


def meta_huber(X_train, y_train, X_test, alpha=0.01, epsilon=1.5,
               n_folds=5, seed=42):
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train), dtype=np.float64)
    for tr, va in cv.split(X_train):
        m = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=2000)
        m.fit(X_train[tr], y_train[tr])
        oof[va] = m.predict(X_train[va])
    final = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=2000)
    final.fit(X_train, y_train)
    return np.clip(oof, 1.0, 5.0), np.clip(final.predict(X_test), 1.0, 5.0), final.coef_


def meta_constrained_blend(X_train, y_train, X_test, n_folds=5, seed=42):
    """Find weights >=0, sum=1 minimising MSE on OOF folds (proper stacking)."""
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n, k = X_train.shape

    def fit_weights(X, y):
        # Solve constrained least-squares: minimise ||Xw - y||^2 s.t. w>=0, sum w =1
        from scipy.optimize import minimize
        x0 = np.ones(k) / k
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bnds = [(0.0, 1.0)] * k
        res = minimize(lambda w: np.mean((X @ w - y) ** 2), x0,
                       bounds=bnds, constraints=cons, method="SLSQP")
        return res.x

    oof = np.zeros(n, dtype=np.float64)
    fold_weights = []
    for tr, va in cv.split(X_train):
        w = fit_weights(X_train[tr], y_train[tr])
        fold_weights.append(w)
        oof[va] = X_train[va] @ w
    # Final weights: average across folds (or refit on all train)
    final_w = fit_weights(X_train, y_train)
    return np.clip(oof, 1.0, 5.0), np.clip(X_test @ final_w, 1.0, 5.0), final_w


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    train = pd.read_csv(DATA / "train.csv")
    public = pd.read_csv(DATA / "public_test.csv")
    private = pd.read_csv(DATA / "private_test.csv")
    sample = pd.read_csv(DATA / "Test_Submission.csv")

    anchors = [
        load_ridge_anchor(),
        load_specter_anchor("0.69972", "specter_3s"),
        load_specter_anchor("0.68737", "specter_5s"),
        load_specter_anchor("0.68718", "specter_v2"),
    ]
    print("Loaded anchors:")
    for a in anchors:
        print(f"  {a.name:12s} oof={len(a.oof)} public={len(a.public)} private={len(a.private)}")

    train_meta, public_meta, private_meta = merge_anchors(anchors, train, public, private)
    feature_cols = [f"{a.name}_oof" for a in anchors]
    test_feature_cols = [f"{a.name}_score" for a in anchors]

    # Sanity: assert no NaNs in features
    for col in feature_cols:
        assert train_meta[col].notna().all(), f"NaN in {col}"
    for col in test_feature_cols:
        assert public_meta[col].notna().all(), f"NaN public {col}"
        assert private_meta[col].notna().all(), f"NaN private {col}"

    X_train = train_meta[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_meta[LABEL].astype(float).to_numpy()
    y_class = train_meta[LABEL].astype(int).to_numpy()
    X_public = public_meta[test_feature_cols].to_numpy(dtype=np.float64)
    X_private = private_meta[test_feature_cols].to_numpy(dtype=np.float64)
    train_dist = pd.Series(y_class).value_counts(normalize=True).reindex(
        [1, 2, 3, 4, 5], fill_value=0).to_numpy()

    # ----- Correlations -----
    corr = pd.DataFrame(X_train, columns=feature_cols).corr()
    corr.to_csv(OUT / "correlation_matrix.csv")
    print("\nOOF Pearson correlation:")
    print(corr.round(3).to_string())

    # ----- Per-anchor OOF QWK (used for weighting) -----
    per_anchor_qwk = {}
    for i, a in enumerate(anchors):
        scores = X_train[:, i]
        thr, qwk = tune_thresholds_constrained(y_class, scores, train_dist)
        per_anchor_qwk[a.name] = float(qwk)
        print(f"  per-anchor tuned QWK[{a.name}] = {qwk:.4f}")

    candidates: list[dict] = []

    def evaluate(name, oof_scores, test_public_scores, test_private_scores, extra=None):
        thr, oof_qwk = tune_thresholds_constrained(y_class, oof_scores, train_dist)
        oof_pred = scores_to_labels(oof_scores, thr)
        test_combined = np.concatenate([test_public_scores, test_private_scores])
        public_pred = scores_to_labels(test_public_scores, thr)
        private_pred = scores_to_labels(test_private_scores, thr)
        combined_dist = pd.Series(np.concatenate([public_pred, private_pred])).value_counts(
            normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()
        l1_test = float(np.sum(np.abs(combined_dist - train_dist)))
        oof_dist = pd.Series(oof_pred).value_counts(normalize=True).reindex(
            [1, 2, 3, 4, 5], fill_value=0).to_numpy()
        l1_oof = float(np.sum(np.abs(oof_dist - train_dist)))

        row = {
            "candidate": name,
            "oof_qwk": oof_qwk,
            "oof_mae": float(mean_absolute_error(y_class, oof_pred)),
            "oof_macro_f1": float(f1_score(y_class, oof_pred, average="macro")),
            "thresholds": [float(t) for t in thr],
            "oof_l1_dist": l1_oof,
            "test_combined_l1_dist": l1_test,
            "label_dist_combined": {int(k): int(v) for k, v in pd.Series(
                np.concatenate([public_pred, private_pred])).value_counts().sort_index().items()},
            "label_dist_public": {int(k): int(v) for k, v in pd.Series(public_pred).value_counts().sort_index().items()},
            "label_dist_private": {int(k): int(v) for k, v in pd.Series(private_pred).value_counts().sort_index().items()},
            "weights": extra,
        }
        candidates.append(row)
        # Save artefacts
        pd.DataFrame({"id": train_meta["id"], "Label": y_class,
                      "oof_score": oof_scores, "oof_pred": oof_pred}).to_csv(
            OUT / f"{name}_oof.csv", index=False)
        pd.DataFrame({"id": public_meta["id"], "score": test_public_scores,
                      "pred": public_pred}).to_csv(OUT / f"{name}_public.csv", index=False)
        pd.DataFrame({"id": private_meta["id"], "score": test_private_scores,
                      "pred": private_pred}).to_csv(OUT / f"{name}_private.csv", index=False)
        combo = pd.concat([
            pd.DataFrame({"id": public_meta["id"], "Label": public_pred}),
            pd.DataFrame({"id": private_meta["id"], "Label": private_pred}),
        ], ignore_index=True)
        submission = sample[["id"]].merge(combo, on="id", how="left")
        submission["Label"] = submission["Label"].astype(int)
        submission.to_csv(OUT / f"{name}_submission.csv", index=False)
        print(f"\n[{name}] OOF QWK={oof_qwk:.4f}  oof_L1={l1_oof:.4f}  test_L1={l1_test:.4f}")
        print(f"   thresholds={[round(t, 4) for t in thr]}")
        print(f"   combined_dist={row['label_dist_combined']}")
        if extra is not None:
            print(f"   weights/coef={extra}")

    # 1. Simple average
    oof, pub_s = meta_simple_avg(X_train, y_train, X_public)
    _, priv_s = meta_simple_avg(X_train, y_train, X_private)
    evaluate("simple_avg", oof, pub_s, priv_s, extra=[1/4]*4)

    # 2. Weighted by per-anchor OOF QWK
    weights = [per_anchor_qwk[a.name] for a in anchors]
    oof = X_train @ (np.array(weights) / sum(weights))
    pub_s = X_public @ (np.array(weights) / sum(weights))
    priv_s = X_private @ (np.array(weights) / sum(weights))
    evaluate("weighted_qwk", oof, pub_s, priv_s, extra=[float(w/sum(weights)) for w in weights])

    # 3. Ridge meta
    oof, _, ridge_coef = meta_ridge(X_train, y_train, X_public, alpha=1.0)
    _, pub_s, _ = meta_ridge(X_train, y_train, X_public, alpha=1.0)
    _, priv_s, _ = meta_ridge(X_train, y_train, X_private, alpha=1.0)
    evaluate("ridge_meta_a1", oof, pub_s, priv_s, extra=ridge_coef.tolist())

    # 4. Huber meta
    oof, _, huber_coef = meta_huber(X_train, y_train, X_public)
    _, pub_s, _ = meta_huber(X_train, y_train, X_public)
    _, priv_s, _ = meta_huber(X_train, y_train, X_private)
    evaluate("huber_meta", oof, pub_s, priv_s, extra=huber_coef.tolist())

    # 5. Constrained blend (weights >=0, sum=1)
    oof, _, blend_w = meta_constrained_blend(X_train, y_train, X_public)
    _, pub_s, _ = meta_constrained_blend(X_train, y_train, X_public)
    _, priv_s, _ = meta_constrained_blend(X_train, y_train, X_private)
    evaluate("constrained_blend", oof, pub_s, priv_s, extra=blend_w.tolist())

    # 6. 2-anchor stack: ridge + specter_3s only (drop redundant SPECTER2s)
    keep_idx = [i for i, a in enumerate(anchors) if a.name in {"ridge_5x5", "specter_3s"}]
    X_train_2 = X_train[:, keep_idx]
    X_public_2 = X_public[:, keep_idx]
    X_private_2 = X_private[:, keep_idx]
    for blend_name, w_specter in [("blend_2anchor_50_50", 0.5),
                                    ("blend_2anchor_60_specter", 0.6),
                                    ("blend_2anchor_70_specter", 0.7),
                                    ("blend_2anchor_80_specter", 0.8)]:
        w = np.array([1 - w_specter, w_specter])  # ridge, specter_3s order
        oof = X_train_2 @ w
        pub_s = X_public_2 @ w
        priv_s = X_private_2 @ w
        evaluate(blend_name, oof, pub_s, priv_s, extra=w.tolist())

    # 7. Bias toward specter_3s (50%) + meta-blend the other three
    bias = 0.50
    other_idx = [i for i, a in enumerate(anchors) if a.name != "specter_3s"]
    primary_idx = next(i for i, a in enumerate(anchors) if a.name == "specter_3s")
    X_train_o = X_train[:, other_idx]
    X_public_o = X_public[:, other_idx]
    X_private_o = X_private[:, other_idx]
    # simple average of the other 3 anchors as the residual
    other_oof = X_train_o.mean(axis=1)
    other_pub = X_public_o.mean(axis=1)
    other_priv = X_private_o.mean(axis=1)
    oof = bias * X_train[:, primary_idx] + (1 - bias) * other_oof
    pub_s = bias * X_public[:, primary_idx] + (1 - bias) * other_pub
    priv_s = bias * X_private[:, primary_idx] + (1 - bias) * other_priv
    evaluate("biased_specter3s_50_avg_other", oof, pub_s, priv_s,
             extra=[bias, 1 - bias])

    # 8. 70-30 bias (heavier on the public-best anchor)
    bias = 0.70
    oof = bias * X_train[:, primary_idx] + (1 - bias) * other_oof
    pub_s = bias * X_public[:, primary_idx] + (1 - bias) * other_pub
    priv_s = bias * X_private[:, primary_idx] + (1 - bias) * other_priv
    evaluate("biased_specter3s_70_avg_other", oof, pub_s, priv_s,
             extra=[bias, 1 - bias])

    # 9. Anchor-only baseline for sanity (= specter_3s alone, just to confirm same recipe)
    oof = X_train[:, primary_idx]
    pub_s = X_public[:, primary_idx]
    priv_s = X_private[:, primary_idx]
    evaluate("anchor_specter_3s_only", oof, pub_s, priv_s, extra=[1.0])

    # ----- Report -----
    rep = pd.DataFrame(candidates).sort_values("oof_qwk", ascending=False)
    rep.to_csv(OUT / "stacking_report.csv", index=False)
    print("\n=== Final ranking by OOF QWK ===")
    print(rep[["candidate", "oof_qwk", "oof_mae", "oof_l1_dist", "test_combined_l1_dist"]].round(4).to_string(index=False))

    # Best candidate by OOF QWK among those with reasonable test_L1 dist (<0.13)
    safe = rep[rep["test_combined_l1_dist"] < 0.13].sort_values("oof_qwk", ascending=False)
    if len(safe) == 0:
        safe = rep
    best_name = safe.iloc[0]["candidate"]
    print(f"\nBest safe candidate: {best_name}")
    import shutil as _sh
    _sh.copy(OUT / f"{best_name}_submission.csv", OUT / "best_submission.csv")
    (OUT / "best_candidate.txt").write_text(best_name + "\n")


if __name__ == "__main__":
    main()
