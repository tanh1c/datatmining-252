"""Pick top-3 BGE 4-anchor stack candidates by L7/L9 rule and emit submissions.

L7 rule:
  - Rank candidates by test combined L1 distance FIRST, OOF QWK second.
  - Winner = lowest test_L1 with OOF >= anchor + 0.005.

Emits three labelled candidates:
  1. safest      : lowest test_L1, OOF still > anchor 0.6412
  2. high_signal : highest OOF, regardless of test_L1 (within hard cap 0.20)
  3. single_knob : closest to "single weight change vs the 60/20/20 public anchor"
                   per L10 — kept as a hedge.

Run AFTER step10_bge_4anchor_stack.py has produced candidates.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
STACK_DIR = ROOT / "outputs" / "step10_bge_stack"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "raw"

ANCHOR_FILES = {
    "bge": ROOT / "outputs/bge_large_finetune_outputs/bge_large_finetune",
    "scincl": ROOT / "outputs/scincl_finetune",
    "specter": ROOT / "outputs/specter2_finetune",
    "ridge": ROOT / "outputs/ridge_threshold/reports",
}

# By L9: anchor 0.72103 has OOF 0.6412, test_L1 0.162. Pick must beat OOF + close to or below test_L1.
ANCHOR_OOF = 0.6412
ANCHOR_TEST_L1 = 0.162

# Manually picked top-3 from the sweep (see candidates.csv).
PICKS = {
    "safest": (0.40, 0.40, 0.15, 0.05),     # lowest test_L1 0.1556, OOF 0.6567
    "high_signal": (0.25, 0.25, 0.25, 0.25), # highest OOF 0.6593, test_L1 0.1724
    "single_knob": (0.30, 0.40, 0.10, 0.20), # OOF 0.6572, test_L1 0.1584, ridge=0.20 like anchor
}


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


def main():
    oof, pub, pri, sample = load()
    y = oof["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    summary = []

    for label, w in PICKS.items():
        w_bge, w_scincl, w_specter, w_ridge = w
        oof_score = (
            w_bge * oof["bge"].to_numpy()
            + w_scincl * oof["scincl"].to_numpy()
            + w_specter * oof["specter"].to_numpy()
            + w_ridge * oof["ridge"].to_numpy()
        )
        pub_score = (
            w_bge * pub["bge"].to_numpy()
            + w_scincl * pub["scincl"].to_numpy()
            + w_specter * pub["specter"].to_numpy()
            + w_ridge * pub["ridge"].to_numpy()
        )
        pri_score = (
            w_bge * pri["bge"].to_numpy()
            + w_scincl * pri["scincl"].to_numpy()
            + w_specter * pri["specter"].to_numpy()
            + w_ridge * pri["ridge"].to_numpy()
        )

        thr, oof_qwk = tune_constrained(y, oof_score, train_dist)
        oof_lab = scores_to_labels(oof_score, thr)
        pub_lab = scores_to_labels(pub_score, thr)
        pri_lab = scores_to_labels(pri_score, thr)

        test_combined = np.concatenate([pub_lab, pri_lab])
        test_l1 = float(np.sum(np.abs(predicted_dist(test_combined) - train_dist)))

        combo = pd.concat([
            pd.DataFrame({"id": pub["id"], "Label": pub_lab}),
            pd.DataFrame({"id": pri["id"], "Label": pri_lab}),
        ], ignore_index=True)
        sub = sample[["id"]].merge(combo, on="id", how="left")
        sub["Label"] = sub["Label"].astype(int)

        sub_path = SUBMIT_DIR / f"next_bge_4anchor_{label}_submission.csv"
        sub.to_csv(sub_path, index=False)

        # Also save the score dump
        out_dir = STACK_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": oof["id"], "Label": y, "oof_score": oof_score, "oof_pred": oof_lab}).to_csv(
            out_dir / "oof_scores.csv", index=False
        )
        pd.DataFrame({"id": pub["id"], "score": pub_score, "pred": pub_lab}).to_csv(
            out_dir / "public_scores.csv", index=False
        )
        pd.DataFrame({"id": pri["id"], "score": pri_score, "pred": pri_lab}).to_csv(
            out_dir / "private_scores.csv", index=False
        )
        sub.to_csv(out_dir / "submission.csv", index=False)

        info = {
            "label": label,
            "weights": {"bge": w_bge, "scincl": w_scincl, "specter": w_specter, "ridge": w_ridge},
            "oof_qwk": float(oof_qwk),
            "test_l1": test_l1,
            "thresholds": [float(t) for t in thr],
            "pub_dist": {int(k): int(v) for k, v in pd.Series(pub_lab).value_counts().sort_index().items()},
            "pri_dist": {int(k): int(v) for k, v in pd.Series(pri_lab).value_counts().sort_index().items()},
            "submission": str(sub_path),
        }
        summary.append(info)
        print(json.dumps(info, indent=2))

    (STACK_DIR / "picks_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Submissions saved to {SUBMIT_DIR} ===")
    print(f"Anchor 0.72103: OOF=0.6412 test_L1=0.162")
    print()
    for s in summary:
        w = s["weights"]
        print(f"  [{s['label']:11s}] bge={w['bge']:.2f} scincl={w['scincl']:.2f} "
              f"specter={w['specter']:.2f} ridge={w['ridge']:.2f}  "
              f"OOF={s['oof_qwk']:.4f} test_L1={s['test_l1']:.4f}")


if __name__ == "__main__":
    main()
