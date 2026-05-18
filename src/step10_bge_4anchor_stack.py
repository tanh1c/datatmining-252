"""Stacking step (step 10): 4-anchor stack with BGE as new primary signal.

Sweeps weight combinations around the winning probe `bge 0.40 / scincl 0.30 /
specter 0.20 / ridge 0.10`. For each candidate:
- Apply distribution-constrained threshold tuner (lambda=0.5).
- Compute OOF QWK + test combined L1 distance to train distribution.
- Pick winner by L7: rank by test L1 first (must be inside acceptable band),
  then OOF QWK. Bias toward the public-validated 60/20/20 anchor's sister
  weights when in doubt.

Outputs (under outputs/step10_bge_stack/):
  - candidates.csv         : all weight combos with metrics
  - winner_oof.csv         : winner OOF scores
  - winner_public.csv      : winner public scores
  - winner_private.csv     : winner private scores
  - winner_submission.csv  : Kaggle submission
  - manifest.md            : human-readable summary
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step10_bge_stack"
OUT.mkdir(parents=True, exist_ok=True)

ANCHOR_FILES = {
    "bge": {
        "oof": ROOT / "outputs/bge_large_finetune_outputs/bge_large_finetune/oof_scores.csv",
        "public": ROOT / "outputs/bge_large_finetune_outputs/bge_large_finetune/public_scores.csv",
        "private": ROOT / "outputs/bge_large_finetune_outputs/bge_large_finetune/private_scores.csv",
    },
    "scincl": {
        "oof": ROOT / "outputs/scincl_finetune/oof_scores.csv",
        "public": ROOT / "outputs/scincl_finetune/public_scores.csv",
        "private": ROOT / "outputs/scincl_finetune/private_scores.csv",
    },
    "specter": {
        "oof": ROOT / "outputs/specter2_finetune/oof_scores.csv",
        "public": ROOT / "outputs/specter2_finetune/public_scores.csv",
        "private": ROOT / "outputs/specter2_finetune/private_scores.csv",
    },
    "ridge": {
        "oof": ROOT / "outputs/ridge_threshold/reports/ridge_threshold_5fold_5seed_891e6ee4_oof.csv",
        "public": ROOT / "outputs/ridge_threshold/reports/ridge_threshold_5fold_5seed_891e6ee4_public_scores.csv",
        "private": ROOT / "outputs/ridge_threshold/reports/ridge_threshold_5fold_5seed_891e6ee4_private_scores.csv",
    },
}

DATA_DIR = ROOT / "data" / "raw"


def load_scores():
    train = pd.read_csv(DATA_DIR / "train.csv")
    public = pd.read_csv(DATA_DIR / "public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test.csv")
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")

    oof = train[["id", "Label"]].copy()
    pub = public[["id"]].copy()
    pri = private[["id"]].copy()

    for name, paths in ANCHOR_FILES.items():
        # OOF
        df_oof = pd.read_csv(paths["oof"])
        if "oof_score" not in df_oof.columns:
            raise ValueError(f"{paths['oof']} missing oof_score")
        oof = oof.merge(df_oof[["id", "oof_score"]].rename(columns={"oof_score": name}), on="id", how="left")
        # public
        df_pub = pd.read_csv(paths["public"])
        score_col = "score" if "score" in df_pub.columns else "oof_score"
        pub = pub.merge(df_pub[["id", score_col]].rename(columns={score_col: name}), on="id", how="left")
        # private
        df_pri = pd.read_csv(paths["private"])
        score_col = "score" if "score" in df_pri.columns else "oof_score"
        pri = pri.merge(df_pri[["id", score_col]].rename(columns={score_col: name}), on="id", how="left")

    assert oof[list(ANCHOR_FILES)].notna().all().all(), "oof has NaN"
    assert pub[list(ANCHOR_FILES)].notna().all().all(), "public has NaN"
    assert pri[list(ANCHOR_FILES)].notna().all().all(), "private has NaN"
    return oof, pub, pri, sample


def scores_to_labels(scores, thresholds):
    return np.digitize(scores, np.sort(np.asarray(thresholds, dtype=float))) + 1


def predicted_dist(labels):
    return pd.Series(labels).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()


def tune_thresholds_constrained(y_true, oof, train_dist, lambd=0.5, seed=42):
    def objective(raw):
        thr = np.sort(raw)
        gap = np.min(np.diff(thr))
        gap_pen = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
        labels = scores_to_labels(oof, thr)
        qwk = cohen_kappa_score(y_true, labels, weights="quadratic")
        dist_pen = float(np.sum(np.abs(predicted_dist(labels) - train_dist)))
        return -qwk + gap_pen + lambd * dist_pen

    res = differential_evolution(
        objective,
        bounds=[(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)],
        seed=seed, maxiter=120, popsize=15, polish=True, updating="immediate", workers=1,
    )
    thr = np.sort(res.x)
    return thr, cohen_kappa_score(y_true, scores_to_labels(oof, thr), weights="quadratic")


def evaluate(weights, oof_df, pub_df, pri_df, y, train_dist):
    """Return dict with metrics + score arrays for a weight tuple."""
    w_bge, w_scincl, w_specter, w_ridge = weights
    s = sum(weights)
    assert abs(s - 1.0) < 1e-9, f"weights must sum to 1, got {s}"

    oof_score = (
        w_bge * oof_df["bge"].to_numpy()
        + w_scincl * oof_df["scincl"].to_numpy()
        + w_specter * oof_df["specter"].to_numpy()
        + w_ridge * oof_df["ridge"].to_numpy()
    )
    pub_score = (
        w_bge * pub_df["bge"].to_numpy()
        + w_scincl * pub_df["scincl"].to_numpy()
        + w_specter * pub_df["specter"].to_numpy()
        + w_ridge * pub_df["ridge"].to_numpy()
    )
    pri_score = (
        w_bge * pri_df["bge"].to_numpy()
        + w_scincl * pri_df["scincl"].to_numpy()
        + w_specter * pri_df["specter"].to_numpy()
        + w_ridge * pri_df["ridge"].to_numpy()
    )

    thresholds, oof_qwk = tune_thresholds_constrained(y, oof_score, train_dist)
    oof_labels = scores_to_labels(oof_score, thresholds)
    pub_labels = scores_to_labels(pub_score, thresholds)
    pri_labels = scores_to_labels(pri_score, thresholds)

    test_combined_labels = np.concatenate([pub_labels, pri_labels])
    test_dist = predicted_dist(test_combined_labels)
    test_l1 = float(np.sum(np.abs(test_dist - train_dist)))

    oof_dist = predicted_dist(oof_labels)
    oof_l1 = float(np.sum(np.abs(oof_dist - train_dist)))

    return {
        "weights": weights,
        "oof_qwk": float(oof_qwk),
        "oof_round_qwk": float(cohen_kappa_score(y, np.clip(np.round(oof_score), 1, 5).astype(int), weights="quadratic")),
        "oof_mae": float(mean_absolute_error(y, oof_labels)),
        "oof_l1": oof_l1,
        "test_l1": test_l1,
        "thresholds": thresholds.tolist(),
        "oof_score": oof_score,
        "pub_score": pub_score,
        "pri_score": pri_score,
        "oof_labels": oof_labels,
        "pub_labels": pub_labels,
        "pri_labels": pri_labels,
        "pub_dist": {int(k): int(v) for k, v in pd.Series(pub_labels).value_counts().sort_index().items()},
        "pri_dist": {int(k): int(v) for k, v in pd.Series(pri_labels).value_counts().sort_index().items()},
        "combined_dist": {int(k): int(v) for k, v in pd.Series(test_combined_labels).value_counts().sort_index().items()},
    }


def make_grid():
    """Single-knob sweeps around the winning probe (0.40, 0.30, 0.20, 0.10)."""
    candidates = []
    # The probe winner.
    candidates.append((0.40, 0.30, 0.20, 0.10))
    # Single-knob neighbours: vary one weight at a time, balance the others.
    # Move bge ±0.05
    candidates += [
        (0.35, 0.32, 0.22, 0.11),
        (0.45, 0.28, 0.18, 0.09),
        (0.50, 0.25, 0.17, 0.08),
        (0.30, 0.35, 0.22, 0.13),
    ]
    # Move scincl ±0.05 / ±0.10
    candidates += [
        (0.40, 0.25, 0.22, 0.13),
        (0.40, 0.35, 0.18, 0.07),
        (0.40, 0.20, 0.25, 0.15),
        (0.40, 0.40, 0.15, 0.05),
    ]
    # Move specter ±0.05
    candidates += [
        (0.40, 0.30, 0.15, 0.15),
        (0.40, 0.30, 0.25, 0.05),
    ]
    # Move ridge ±0.05 / ±0.10
    candidates += [
        (0.40, 0.30, 0.20, 0.10),  # baseline (already there but harmless)
        (0.40, 0.30, 0.25, 0.05),
        (0.40, 0.30, 0.15, 0.15),
        (0.40, 0.30, 0.10, 0.20),
        (0.40, 0.30, 0.30, 0.00),
    ]
    # 4-anchor variants from probe table C
    candidates += [
        (0.40, 0.20, 0.20, 0.20),
        (0.35, 0.30, 0.20, 0.15),
        (0.30, 0.30, 0.20, 0.20),
        (0.25, 0.25, 0.25, 0.25),
        (0.30, 0.40, 0.10, 0.20),
        (0.30, 0.30, 0.30, 0.10),
    ]
    # Reference: existing public-best stack (no bge)
    candidates.append((0.00, 0.60, 0.20, 0.20))
    # Add: keep ridge at 0.20 (proven public-safe per L10), sweep bge/scincl/specter
    for w_bge in [0.30, 0.35, 0.40, 0.45]:
        for w_scincl in [0.20, 0.25, 0.30, 0.35]:
            w_specter = 1.0 - 0.20 - w_bge - w_scincl
            if 0.05 <= w_specter <= 0.30:
                candidates.append((w_bge, w_scincl, round(w_specter, 4), 0.20))
    # Dedupe
    seen = set()
    unique = []
    for c in candidates:
        c_round = tuple(round(x, 4) for x in c)
        if abs(sum(c_round) - 1.0) > 1e-6:
            continue
        if c_round in seen:
            continue
        seen.add(c_round)
        unique.append(c_round)
    return unique


def main():
    oof, pub, pri, sample = load_scores()
    y = oof["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()
    print("train_dist =", dict(zip([1, 2, 3, 4, 5], train_dist.round(3).tolist())))

    grid = make_grid()
    print(f"\nEvaluating {len(grid)} weight combinations...")

    results = []
    for w in grid:
        r = evaluate(w, oof, pub, pri, y, train_dist)
        results.append(r)
        print(f"  bge={w[0]:.2f} scincl={w[1]:.2f} specter={w[2]:.2f} ridge={w[3]:.2f}  "
              f"OOF_QWK={r['oof_qwk']:.4f}  test_L1={r['test_l1']:.4f}  "
              f"pub_dist={r['pub_dist']} pri_dist={r['pri_dist']}")

    df = pd.DataFrame([
        {
            "name": f"bge_{w[0]:.2f}_scincl_{w[1]:.2f}_specter_{w[2]:.2f}_ridge_{w[3]:.2f}".replace("0.", "0p"),
            "w_bge": w[0], "w_scincl": w[1], "w_specter": w[2], "w_ridge": w[3],
            "oof_qwk": r["oof_qwk"],
            "oof_round_qwk": r["oof_round_qwk"],
            "oof_mae": r["oof_mae"],
            "oof_l1": r["oof_l1"],
            "test_l1": r["test_l1"],
            "thresholds": ",".join(f"{t:.6f}" for t in r["thresholds"]),
            "pub_dist": str(r["pub_dist"]),
            "pri_dist": str(r["pri_dist"]),
            "combined_dist": str(r["combined_dist"]),
        }
        for w, r in zip(grid, results)
    ]).sort_values("oof_qwk", ascending=False)
    df.to_csv(OUT / "candidates.csv", index=False)

    # Pick winner by L7 rule:
    # 1. test_l1 must be acceptable (< 0.20 hard cap from L9; prefer < 0.18)
    # 2. among those, rank by oof_qwk
    # 3. as tiebreaker, prefer the weights closest to public-validated 60/20/20 anchor's spirit
    eligible = df[df["test_l1"] <= 0.18].copy()
    if eligible.empty:
        eligible = df[df["test_l1"] <= 0.20].copy()
        print("\nWARNING: no candidate with test_l1 <= 0.18, relaxing to 0.20")
    eligible = eligible.sort_values(["oof_qwk", "test_l1"], ascending=[False, True])

    # For the actual submission, also include a "lowest-test-L1 within top-OOF-band" pick (per L7).
    top_oof_band = df["oof_qwk"].max() - 0.005
    safety_pool = df[df["oof_qwk"] >= top_oof_band].sort_values("test_l1")
    print(f"\nTop-OOF-band candidates (within 0.005 of max OOF):")
    print(safety_pool[["name", "oof_qwk", "test_l1", "combined_dist"]].head(10).to_string(index=False))

    print(f"\n=== Eligible candidates (test_l1 <= 0.18), top 5 by OOF: ===")
    print(eligible[["name", "oof_qwk", "test_l1", "combined_dist"]].head(5).to_string(index=False))

    # Winner = the best OOF among eligible (highest signal)
    winner_row = eligible.iloc[0]
    winner_name = winner_row["name"]
    winner_idx = df.index.get_loc(winner_row.name)
    # Find back the result dict by matching weights
    winner_w = (winner_row["w_bge"], winner_row["w_scincl"], winner_row["w_specter"], winner_row["w_ridge"])
    winner_result = next(r for w, r in zip(grid, results) if all(abs(a - b) < 1e-9 for a, b in zip(w, winner_w)))

    print(f"\nWINNER: {winner_name}")
    print(f"  weights: bge={winner_w[0]:.2f} scincl={winner_w[1]:.2f} specter={winner_w[2]:.2f} ridge={winner_w[3]:.2f}")
    print(f"  OOF QWK: {winner_result['oof_qwk']:.4f}")
    print(f"  test_L1: {winner_result['test_l1']:.4f}")
    print(f"  pub_dist: {winner_result['pub_dist']}")
    print(f"  pri_dist: {winner_result['pri_dist']}")

    # Save winner artefacts
    pd.DataFrame({
        "id": oof["id"], "Label": y,
        "oof_score": winner_result["oof_score"],
        "oof_pred": winner_result["oof_labels"],
    }).to_csv(OUT / "winner_oof.csv", index=False)
    pd.DataFrame({
        "id": pub["id"],
        "score": winner_result["pub_score"],
        "pred": winner_result["pub_labels"],
    }).to_csv(OUT / "winner_public.csv", index=False)
    pd.DataFrame({
        "id": pri["id"],
        "score": winner_result["pri_score"],
        "pred": winner_result["pri_labels"],
    }).to_csv(OUT / "winner_private.csv", index=False)

    combo = pd.concat([
        pd.DataFrame({"id": pub["id"], "Label": winner_result["pub_labels"]}),
        pd.DataFrame({"id": pri["id"], "Label": winner_result["pri_labels"]}),
    ], ignore_index=True)
    submission = sample[["id"]].merge(combo, on="id", how="left")
    submission["Label"] = submission["Label"].astype(int)
    submission.to_csv(OUT / "winner_submission.csv", index=False)
    print(f"\nsubmission rows = {len(submission)}")
    print(f"artefacts -> {OUT}")

    # Manifest
    manifest = [
        "# Step 10 — BGE 4-anchor stack (post-L11/L12 anchor candidate)",
        "",
        f"**Winner:** `{winner_name}`",
        "",
        f"- Weights: bge={winner_w[0]:.2f} scincl={winner_w[1]:.2f} specter={winner_w[2]:.2f} ridge={winner_w[3]:.2f}",
        f"- Constrained-tuned OOF QWK: **{winner_result['oof_qwk']:.4f}** "
        f"(prev best 60/20/20 = 0.6412)",
        f"- Test combined L1 distance: **{winner_result['test_l1']:.4f}** "
        f"(prev best 60/20/20 = 0.162)",
        f"- Thresholds: `{winner_result['thresholds']}`",
        f"- Public predicted dist: `{winner_result['pub_dist']}`",
        f"- Private predicted dist: `{winner_result['pri_dist']}`",
        f"- Combined predicted dist: `{winner_result['combined_dist']}`",
        "",
        "## Top 10 candidates by OOF QWK",
        "",
        df.head(10)[["name", "oof_qwk", "oof_round_qwk", "test_l1", "combined_dist"]].to_markdown(index=False),
        "",
        "## Top 10 candidates by lowest test L1 (per L7)",
        "",
        df.sort_values("test_l1").head(10)[["name", "oof_qwk", "oof_round_qwk", "test_l1", "combined_dist"]].to_markdown(index=False),
    ]
    (OUT / "manifest.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
