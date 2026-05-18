"""Step 11 — Re-sweep around the public-confirmed `safest` (40/40/15/05).

Per L14: ridge weight 0.20 from L10 (3-anchor era) does NOT carry over.
This sweep only explores ridge in [0.00, 0.10] and prefers single-knob
neighbours of the public-validated 40/40/15/05.

Pick rule (per L7/L9, refined by L13/L14):
1. test_L1 must be <= 0.165 (the safest had 0.156).
2. Within that, rank by oof_qwk and prefer SINGLE-knob distance from safest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step11_resweep"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR = ROOT / "outputs" / "submissions"
DATA_DIR = ROOT / "data" / "raw"

ANCHOR_FILES = {
    "bge": ROOT / "outputs/bge_large_finetune_outputs/bge_large_finetune",
    "scincl": ROOT / "outputs/scincl_finetune",
    "specter": ROOT / "outputs/specter2_finetune",
    "ridge": ROOT / "outputs/ridge_threshold/reports",
}

PUBLIC_ANCHOR = (0.40, 0.40, 0.15, 0.05)  # safest -> 0.72374


def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    public = pd.read_csv(DATA_DIR / "public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test.csv")
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")

    oof = train[["id", "Label"]].copy()
    pub = public[["id"]].copy()
    pri = private[["id"]].copy()

    file_map = {
        "bge": ("oof_scores.csv", "public_scores.csv", "private_scores.csv"),
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


def tune_constrained(y_true, oof, train_dist, lambd=0.5, seed=42):
    def objective(raw):
        thr = np.sort(raw)
        gap = np.min(np.diff(thr))
        gap_pen = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(oof, thr)
        qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
        dist_pen = float(np.sum(np.abs(predicted_dist(labels) - train_dist)))
        return -qwk + gap_pen + lambd * dist_pen

    res = differential_evolution(
        objective, [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed, maxiter=120, popsize=15, polish=True, updating="immediate", workers=1,
    )
    thr = np.sort(res.x)
    return thr, cohen_kappa_score(y_true, scores_to_labels(oof, thr), weights="quadratic")


def evaluate(weights, oof, pub, pri, y, train_dist):
    w_bge, w_scincl, w_specter, w_ridge = weights
    s_oof = (w_bge * oof["bge"] + w_scincl * oof["scincl"]
             + w_specter * oof["specter"] + w_ridge * oof["ridge"]).to_numpy()
    s_pub = (w_bge * pub["bge"] + w_scincl * pub["scincl"]
             + w_specter * pub["specter"] + w_ridge * pub["ridge"]).to_numpy()
    s_pri = (w_bge * pri["bge"] + w_scincl * pri["scincl"]
             + w_specter * pri["specter"] + w_ridge * pri["ridge"]).to_numpy()
    thr, qwk = tune_constrained(y, s_oof, train_dist)
    oof_lab = scores_to_labels(s_oof, thr)
    pub_lab = scores_to_labels(s_pub, thr)
    pri_lab = scores_to_labels(s_pri, thr)
    test_combined = np.concatenate([pub_lab, pri_lab])
    test_l1 = float(np.sum(np.abs(predicted_dist(test_combined) - train_dist)))
    return {
        "weights": weights, "thresholds": thr.tolist(),
        "oof_qwk": float(qwk),
        "oof_round_qwk": float(cohen_kappa_score(
            y, np.clip(np.round(s_oof), 1, 5).astype(int), weights="quadratic")),
        "test_l1": test_l1,
        "knob_distance": sum(1 for a, b in zip(weights, PUBLIC_ANCHOR) if abs(a - b) > 0.005),
        "s_oof": s_oof, "s_pub": s_pub, "s_pri": s_pri,
        "oof_lab": oof_lab, "pub_lab": pub_lab, "pri_lab": pri_lab,
        "pub_dist": {int(k): int(v) for k, v in pd.Series(pub_lab).value_counts().sort_index().items()},
        "pri_dist": {int(k): int(v) for k, v in pd.Series(pri_lab).value_counts().sort_index().items()},
    }


def make_grid():
    cands = []
    # Single-knob neighbours of safest (40/40/15/05) — only one weight moves vs anchor
    # Move bge ±0.05
    cands += [(0.45, 0.40, 0.10, 0.05), (0.35, 0.40, 0.20, 0.05),
              (0.45, 0.40, 0.15, 0.00), (0.35, 0.45, 0.15, 0.05)]
    # Move scincl ±0.05
    cands += [(0.40, 0.45, 0.10, 0.05), (0.40, 0.35, 0.20, 0.05),
              (0.40, 0.45, 0.15, 0.00), (0.45, 0.35, 0.15, 0.05)]
    # Move specter ±0.05
    cands += [(0.40, 0.40, 0.10, 0.10), (0.40, 0.40, 0.20, 0.00),
              (0.45, 0.40, 0.10, 0.05), (0.40, 0.45, 0.10, 0.05)]
    # Move ridge — ridge can only go down (=0) or up to 0.10
    cands += [(0.40, 0.40, 0.20, 0.00),  # ridge=0
              (0.40, 0.40, 0.10, 0.10),  # ridge=0.10
              (0.40, 0.45, 0.10, 0.05),  # already covered
              (0.45, 0.45, 0.05, 0.05),  # bge+scincl ↑, specter ↓ to keep sum=1
              ]
    # Two-knob moves (still close to anchor)
    cands += [(0.45, 0.45, 0.10, 0.00), (0.35, 0.45, 0.15, 0.05),
              (0.45, 0.40, 0.05, 0.10), (0.40, 0.50, 0.05, 0.05),
              (0.50, 0.40, 0.10, 0.00), (0.50, 0.35, 0.10, 0.05),
              (0.40, 0.50, 0.10, 0.00)]
    # Higher BGE — riding the new strongest signal
    cands += [(0.50, 0.40, 0.05, 0.05), (0.55, 0.35, 0.05, 0.05),
              (0.50, 0.30, 0.15, 0.05)]
    # Equal BGE+scincl, no ridge
    cands += [(0.50, 0.50, 0.00, 0.00), (0.45, 0.45, 0.05, 0.05)]
    # Re-include the public anchor itself for sanity
    cands.append(PUBLIC_ANCHOR)

    seen = set(); unique = []
    for c in cands:
        c = tuple(round(x, 4) for x in c)
        if abs(sum(c) - 1.0) > 1e-6: continue
        if any(x < 0 or x > 1 for x in c): continue
        if c[3] > 0.10 + 1e-6: continue  # L14 hard cap on ridge
        if c in seen: continue
        seen.add(c); unique.append(c)
    return unique


def main():
    oof, pub, pri, sample = load()
    y = oof["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    grid = make_grid()
    print(f"evaluating {len(grid)} candidates (ridge <= 0.10)...")
    results = []
    for w in grid:
        r = evaluate(w, oof, pub, pri, y, train_dist)
        results.append(r)
        marker = "*" if w == PUBLIC_ANCHOR else " "
        print(f" {marker} bge={w[0]:.2f} scincl={w[1]:.2f} specter={w[2]:.2f} ridge={w[3]:.2f}  "
              f"OOF={r['oof_qwk']:.4f} test_L1={r['test_l1']:.4f} knobs={r['knob_distance']}")

    df = pd.DataFrame([{
        "name": f"bge{w[0]:.2f}_scincl{w[1]:.2f}_specter{w[2]:.2f}_ridge{w[3]:.2f}".replace(".", "p"),
        "w_bge": w[0], "w_scincl": w[1], "w_specter": w[2], "w_ridge": w[3],
        "oof_qwk": r["oof_qwk"], "oof_round_qwk": r["oof_round_qwk"],
        "test_l1": r["test_l1"], "knob_distance": r["knob_distance"],
        "thresholds": ",".join(f"{t:.6f}" for t in r["thresholds"]),
        "pub_dist": str(r["pub_dist"]), "pri_dist": str(r["pri_dist"]),
    } for w, r in zip(grid, results)])
    df.to_csv(OUT / "candidates.csv", index=False)

    print("\n=== Top 10 by OOF (test_L1 <= 0.165) ===")
    eligible = df[df["test_l1"] <= 0.165].sort_values("oof_qwk", ascending=False)
    print(eligible.head(10)[["name", "oof_qwk", "test_l1", "knob_distance"]].to_string(index=False))

    print("\n=== Top 5 single-knob neighbours of safest (knob_distance <= 1) ===")
    single = df[df["knob_distance"] <= 1].sort_values("oof_qwk", ascending=False)
    print(single.head(5)[["name", "oof_qwk", "test_l1", "knob_distance"]].to_string(index=False))

    # Picks: best single-knob (L14 priority), best 2-knob hedge, lowest test_L1
    pick_specs = []
    if len(single) > 0:
        pick_specs.append(("single_knob_v2",
                           tuple(single.iloc[0][["w_bge","w_scincl","w_specter","w_ridge"]])))
    pick_specs.append(("low_test_l1",
                       tuple(eligible.sort_values("test_l1").iloc[0][["w_bge","w_scincl","w_specter","w_ridge"]])))
    pick_specs.append(("high_oof_constrained",
                       tuple(eligible.iloc[0][["w_bge","w_scincl","w_specter","w_ridge"]])))

    summary = []
    for label, w in pick_specs:
        w = tuple(round(float(x), 4) for x in w)
        # find the matching result
        r = next(rr for ww, rr in zip(grid, results)
                 if all(abs(a - b) < 1e-6 for a, b in zip(ww, w)))
        combo = pd.concat([
            pd.DataFrame({"id": pub["id"], "Label": r["pub_lab"]}),
            pd.DataFrame({"id": pri["id"], "Label": r["pri_lab"]}),
        ], ignore_index=True)
        sub = sample[["id"]].merge(combo, on="id", how="left")
        sub["Label"] = sub["Label"].astype(int)
        sub_path = SUBMIT_DIR / f"next_bge_4anchor_{label}_submission.csv"
        sub.to_csv(sub_path, index=False)
        out_dir = OUT / label
        out_dir.mkdir(exist_ok=True)
        pd.DataFrame({"id": oof["id"], "Label": y, "oof_score": r["s_oof"], "oof_pred": r["oof_lab"]}).to_csv(out_dir / "oof_scores.csv", index=False)
        pd.DataFrame({"id": pub["id"], "score": r["s_pub"], "pred": r["pub_lab"]}).to_csv(out_dir / "public_scores.csv", index=False)
        pd.DataFrame({"id": pri["id"], "score": r["s_pri"], "pred": r["pri_lab"]}).to_csv(out_dir / "private_scores.csv", index=False)
        sub.to_csv(out_dir / "submission.csv", index=False)
        summary.append({
            "label": label, "weights": list(w),
            "oof_qwk": r["oof_qwk"], "test_l1": r["test_l1"],
            "knob_distance": r["knob_distance"],
            "thresholds": r["thresholds"],
            "pub_dist": r["pub_dist"], "pri_dist": r["pri_dist"],
            "submission": str(sub_path),
        })

    (OUT / "picks_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== submissions saved to {SUBMIT_DIR} ===")
    print(f"Public-validated anchor (safest, 0.72374): bge=0.40 scincl=0.40 specter=0.15 ridge=0.05  OOF=0.6567 test_L1=0.156")
    for s in summary:
        w = s["weights"]
        print(f"  [{s['label']:22s}] bge={w[0]:.2f} scincl={w[1]:.2f} specter={w[2]:.2f} ridge={w[3]:.2f}  "
              f"OOF={s['oof_qwk']:.4f} test_L1={s['test_l1']:.4f} knobs={s['knob_distance']}")


if __name__ == "__main__":
    main()
