from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step17c_qwen14b_heavy_5anchor_stack"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR = ROOT / "outputs" / "submissions"
DATA_DIR = ROOT / "data" / "raw"

ANCHOR_FILES = {
    "qwen14b": ROOT / "outputs/qwen14b_lora_finetune_outputs",
    "e5": ROOT / "outputs/e5_large_finetune_outputs/e5_large_finetune",
    "scincl": ROOT / "outputs/scincl_finetune",
    "specter": ROOT / "outputs/specter2_finetune",
    "ridge": ROOT / "outputs/ridge_threshold/reports",
}

SAFE_L1_CAP = 0.147
E5_BEST_WEIGHTS = (0.00, 0.50, 0.30, 0.20, 0.00)  # qwen14b / e5 / scincl / specter / ridge


def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    public = pd.read_csv(DATA_DIR / "public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test.csv")
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")

    oof = train[["id", "Label"]].copy()
    pub = public[["id"]].copy()
    pri = private[["id"]].copy()
    file_map = {
        "qwen14b": ("oof_scores.csv", "public_scores.csv", "private_scores.csv"),
        "e5": ("oof_scores.csv", "public_scores.csv", "private_scores.csv"),
        "scincl": ("oof_scores.csv", "public_scores.csv", "private_scores.csv"),
        "specter": ("oof_scores.csv", "public_scores.csv", "private_scores.csv"),
        "ridge": (
            "ridge_threshold_5fold_5seed_891e6ee4_oof.csv",
            "ridge_threshold_5fold_5seed_891e6ee4_public_scores.csv",
            "ridge_threshold_5fold_5seed_891e6ee4_private_scores.csv",
        ),
    }
    for name, base in ANCHOR_FILES.items():
        oof_f, pub_f, pri_f = file_map[name]
        df = pd.read_csv(base / oof_f)
        oof = oof.merge(df[["id", "oof_score"]].rename(columns={"oof_score": name}), on="id", how="left")
        df = pd.read_csv(base / pub_f)
        col = "score" if "score" in df.columns else "oof_score"
        pub = pub.merge(df[["id", col]].rename(columns={col: name}), on="id", how="left")
        df = pd.read_csv(base / pri_f)
        col = "score" if "score" in df.columns else "oof_score"
        pri = pri.merge(df[["id", col]].rename(columns={col: name}), on="id", how="left")
    return oof, pub, pri, sample


def scores_to_labels(scores, thresholds):
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def predicted_dist(labels):
    return pd.Series(labels).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()


def tune_constrained(y_true, scores, train_dist, lambd=0.5, seed=42):
    def objective(raw):
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
    return thr, cohen_kappa_score(y_true, scores_to_labels(scores, thr), weights="quadratic")


def evaluate(weights, oof, pub, pri, y, train_dist):
    w_q, w_e, w_sc, w_sp, w_r = weights
    s_oof = (w_q * oof["qwen14b"] + w_e * oof["e5"] + w_sc * oof["scincl"] + w_sp * oof["specter"] + w_r * oof["ridge"]).to_numpy()
    s_pub = (w_q * pub["qwen14b"] + w_e * pub["e5"] + w_sc * pub["scincl"] + w_sp * pub["specter"] + w_r * pub["ridge"]).to_numpy()
    s_pri = (w_q * pri["qwen14b"] + w_e * pri["e5"] + w_sc * pri["scincl"] + w_sp * pri["specter"] + w_r * pri["ridge"]).to_numpy()
    thr, qwk = tune_constrained(y, s_oof, train_dist)
    oof_lab = scores_to_labels(s_oof, thr)
    pub_lab = scores_to_labels(s_pub, thr)
    pri_lab = scores_to_labels(s_pri, thr)
    test_lab = np.concatenate([pub_lab, pri_lab])
    return {
        "weights": weights,
        "thresholds": thr.tolist(),
        "oof_qwk": float(qwk),
        "oof_round_qwk": float(cohen_kappa_score(y, np.clip(np.round(s_oof), 1, 5).astype(int), weights="quadratic")),
        "test_l1": float(np.sum(np.abs(predicted_dist(test_lab) - train_dist))),
        "s_oof": s_oof,
        "s_pub": s_pub,
        "s_pri": s_pri,
        "oof_lab": oof_lab,
        "pub_lab": pub_lab,
        "pri_lab": pri_lab,
        "pub_dist": {int(k): int(v) for k, v in pd.Series(pub_lab).value_counts().sort_index().items()},
        "pri_dist": {int(k): int(v) for k, v in pd.Series(pri_lab).value_counts().sort_index().items()},
    }


def make_grid():
    cands = [E5_BEST_WEIGHTS]
    q_weights = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]
    base_shapes = [
        (0.50, 0.30, 0.20, 0.00),
        (0.55, 0.25, 0.20, 0.00),
        (0.45, 0.35, 0.20, 0.00),
        (0.50, 0.25, 0.20, 0.05),
        (0.45, 0.30, 0.20, 0.05),
        (0.40, 0.35, 0.20, 0.05),
        (0.45, 0.25, 0.25, 0.05),
        (0.40, 0.30, 0.25, 0.05),
        (0.50, 0.20, 0.20, 0.10),
        (0.40, 0.30, 0.20, 0.10),
    ]
    for q in q_weights:
        rest = 1.0 - q
        for e, sc, sp, r in base_shapes:
            cands.append((q, rest * e, rest * sc, rest * sp, rest * r))
    manual = [
        (0.20, 0.40, 0.25, 0.15, 0.00),
        (0.25, 0.35, 0.25, 0.15, 0.00),
        (0.30, 0.30, 0.25, 0.15, 0.00),
        (0.35, 0.30, 0.20, 0.15, 0.00),
        (0.40, 0.25, 0.20, 0.15, 0.00),
        (0.20, 0.35, 0.30, 0.10, 0.05),
        (0.25, 0.30, 0.30, 0.10, 0.05),
        (0.30, 0.30, 0.25, 0.10, 0.05),
    ]
    cands += manual
    seen = set()
    unique = []
    for c in cands:
        c = tuple(round(float(x), 4) for x in c)
        if abs(sum(c) - 1.0) > 1e-6:
            continue
        if c[-1] > 0.10 + 1e-6:
            continue
        if c in seen:
            continue
        seen.add(c)
        unique.append(c)
    return unique


def main():
    oof, pub, pri, sample = load()
    y = oof["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()
    corr = oof[["qwen14b", "e5", "scincl", "specter", "ridge"]].corr(method="pearson")
    corr.to_csv(OUT / "correlation_matrix.csv")
    print("=== Anchor correlation ===")
    print(corr.to_string())

    grid = make_grid()
    print(f"evaluating {len(grid)} candidates...")
    results = []
    for w in grid:
        r = evaluate(w, oof, pub, pri, y, train_dist)
        results.append(r)
        marker = "*" if r["test_l1"] <= SAFE_L1_CAP else " "
        print(f" {marker} q={w[0]:.2f} e5={w[1]:.2f} sc={w[2]:.2f} sp={w[3]:.2f} r={w[4]:.2f} OOF={r['oof_qwk']:.4f} round={r['oof_round_qwk']:.4f} test_L1={r['test_l1']:.4f}")

    rows = []
    for w, r in zip(grid, results):
        rows.append({
            "name": f"q{w[0]:.2f}_e5{w[1]:.2f}_sc{w[2]:.2f}_sp{w[3]:.2f}_r{w[4]:.2f}".replace(".", "p"),
            "w_qwen14b": w[0],
            "w_e5": w[1],
            "w_scincl": w[2],
            "w_specter": w[3],
            "w_ridge": w[4],
            "oof_qwk": r["oof_qwk"],
            "oof_round_qwk": r["oof_round_qwk"],
            "test_l1": r["test_l1"],
            "thresholds": ",".join(f"{t:.6f}" for t in r["thresholds"]),
            "pub_dist": str(r["pub_dist"]),
            "pri_dist": str(r["pri_dist"]),
        })
    df = pd.DataFrame(rows)
    df.sort_values(["test_l1", "oof_qwk"], ascending=[True, False]).to_csv(OUT / "candidates_by_l1.csv", index=False)
    df.sort_values(["oof_qwk", "test_l1"], ascending=[False, True]).to_csv(OUT / "candidates_by_oof.csv", index=False)
    df.to_csv(OUT / "candidates.csv", index=False)

    print("\n=== Safe candidates by OOF ===")
    safe = df[(df["test_l1"] <= SAFE_L1_CAP) & (df["w_qwen14b"] > 0)].sort_values("oof_qwk", ascending=False)
    print(safe.head(10)[["name", "oof_qwk", "oof_round_qwk", "test_l1"]].to_string(index=False))

    print("\n=== Top by OOF regardless of L1 ===")
    print(df.sort_values("oof_qwk", ascending=False).head(10)[["name", "oof_qwk", "oof_round_qwk", "test_l1"]].to_string(index=False))

    picks = []
    if len(safe) > 0:
        picks.append(("safe_high_oof", safe.iloc[0]))
    risky = df[(df["w_qwen14b"] >= 0.20)].sort_values("oof_qwk", ascending=False).head(1)
    if len(risky) > 0:
        picks.append(("risky_high_qwen", risky.iloc[0]))

    summary = []
    seen = set()
    for label, row in picks:
        w = tuple(round(float(row[c]), 4) for c in ["w_qwen14b", "w_e5", "w_scincl", "w_specter", "w_ridge"])
        if w in seen:
            continue
        seen.add(w)
        r = next(rr for ww, rr in zip(grid, results) if all(abs(a - b) < 1e-6 for a, b in zip(ww, w)))
        combo = pd.concat([
            pd.DataFrame({"id": pub["id"], "Label": r["pub_lab"]}),
            pd.DataFrame({"id": pri["id"], "Label": r["pri_lab"]}),
        ], ignore_index=True)
        sub = sample[["id"]].merge(combo, on="id", how="left")
        sub["Label"] = sub["Label"].astype(int)
        sub_path = SUBMIT_DIR / f"next_qwen14b_heavy_5anchor_{label}_submission.csv"
        sub.to_csv(sub_path, index=False)
        out_dir = OUT / label
        out_dir.mkdir(exist_ok=True)
        pd.DataFrame({"id": oof["id"], "Label": y, "oof_score": r["s_oof"], "oof_pred": r["oof_lab"]}).to_csv(out_dir / "oof_scores.csv", index=False)
        pd.DataFrame({"id": pub["id"], "score": r["s_pub"], "pred": r["pub_lab"]}).to_csv(out_dir / "public_scores.csv", index=False)
        pd.DataFrame({"id": pri["id"], "score": r["s_pri"], "pred": r["pri_lab"]}).to_csv(out_dir / "private_scores.csv", index=False)
        sub.to_csv(out_dir / "submission.csv", index=False)
        summary.append({
            "label": label,
            "weights": list(w),
            "oof_qwk": r["oof_qwk"],
            "oof_round_qwk": r["oof_round_qwk"],
            "test_l1": r["test_l1"],
            "submission": str(sub_path),
        })
    (OUT / "picks_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== Picks ===")
    for s in summary:
        w = s["weights"]
        print(f"{s['label']}: q={w[0]:.2f} e5={w[1]:.2f} sc={w[2]:.2f} sp={w[3]:.2f} r={w[4]:.2f} OOF={s['oof_qwk']:.4f} test_L1={s['test_l1']:.4f} -> {s['submission']}")


if __name__ == "__main__":
    main()
