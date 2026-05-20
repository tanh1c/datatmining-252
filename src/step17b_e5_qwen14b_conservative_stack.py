from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
E5_DIR = ROOT / "outputs" / "step13_e5_stack" / "safest_e5"
QWEN_DIR = ROOT / "outputs" / "qwen14b_lora_finetune_outputs"
OUT = ROOT / "outputs" / "step17b_e5_qwen14b_conservative_stack"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

E5_PUBLIC_LB = 0.72394
E5_SAFE_L1 = 0.1422
SAFE_L1_CAP = 0.147
QWEN_WEIGHTS = [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]


def scores_to_labels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
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


def load_scores():
    train = pd.read_csv(DATA_DIR / "train.csv")
    public = pd.read_csv(DATA_DIR / "public_test.csv")
    private = pd.read_csv(DATA_DIR / "private_test.csv")
    sample = pd.read_csv(DATA_DIR / "Test_Submission.csv")

    oof = train[["id", "Label"]].copy()
    pub = public[["id"]].copy()
    pri = private[["id"]].copy()

    e5_oof = pd.read_csv(E5_DIR / "oof_scores.csv")[["id", "oof_score"]].rename(columns={"oof_score": "e5"})
    e5_pub = pd.read_csv(E5_DIR / "public_scores.csv")[["id", "score"]].rename(columns={"score": "e5"})
    e5_pri = pd.read_csv(E5_DIR / "private_scores.csv")[["id", "score"]].rename(columns={"score": "e5"})
    qwen_oof = pd.read_csv(QWEN_DIR / "oof_scores.csv")[["id", "oof_score"]].rename(columns={"oof_score": "qwen14b"})
    qwen_pub = pd.read_csv(QWEN_DIR / "public_scores.csv")[["id", "score"]].rename(columns={"score": "qwen14b"})
    qwen_pri = pd.read_csv(QWEN_DIR / "private_scores.csv")[["id", "score"]].rename(columns={"score": "qwen14b"})

    oof = oof.merge(e5_oof, on="id", how="left").merge(qwen_oof, on="id", how="left")
    pub = pub.merge(e5_pub, on="id", how="left").merge(qwen_pub, on="id", how="left")
    pri = pri.merge(e5_pri, on="id", how="left").merge(qwen_pri, on="id", how="left")
    if oof[["e5", "qwen14b"]].isna().any().any() or pub[["e5", "qwen14b"]].isna().any().any() or pri[["e5", "qwen14b"]].isna().any().any():
        raise ValueError("Missing scores after merging E5 and Qwen14B anchors.")
    return oof, pub, pri, sample


def evaluate(qwen_weight: float, oof: pd.DataFrame, pub: pd.DataFrame, pri: pd.DataFrame, y: np.ndarray, train_dist: np.ndarray):
    e5_weight = 1.0 - qwen_weight
    s_oof = e5_weight * oof["e5"].to_numpy() + qwen_weight * oof["qwen14b"].to_numpy()
    s_pub = e5_weight * pub["e5"].to_numpy() + qwen_weight * pub["qwen14b"].to_numpy()
    s_pri = e5_weight * pri["e5"].to_numpy() + qwen_weight * pri["qwen14b"].to_numpy()
    thresholds, qwk = tune_thresholds(y, s_oof, train_dist)
    oof_lab = scores_to_labels(s_oof, thresholds)
    pub_lab = scores_to_labels(s_pub, thresholds)
    pri_lab = scores_to_labels(s_pri, thresholds)
    test_lab = np.concatenate([pub_lab, pri_lab])
    return {
        "qwen_weight": qwen_weight,
        "e5_weight": e5_weight,
        "thresholds": thresholds.tolist(),
        "oof_qwk": float(qwk),
        "oof_round_qwk": float(cohen_kappa_score(y, np.clip(np.round(s_oof), 1, 5).astype(int), weights="quadratic")),
        "test_l1": float(np.sum(np.abs(predicted_dist(test_lab) - train_dist))),
        "oof_mae": float(mean_absolute_error(y, oof_lab)),
        "s_oof": s_oof,
        "s_pub": s_pub,
        "s_pri": s_pri,
        "oof_lab": oof_lab,
        "pub_lab": pub_lab,
        "pri_lab": pri_lab,
        "pub_dist": {int(k): int(v) for k, v in pd.Series(pub_lab).value_counts().sort_index().items()},
        "pri_dist": {int(k): int(v) for k, v in pd.Series(pri_lab).value_counts().sort_index().items()},
    }


def main() -> None:
    oof, pub, pri, sample = load_scores()
    y = oof["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    corr = oof[["e5", "qwen14b"]].corr(method="pearson")
    corr.to_csv(OUT / "correlation_matrix.csv")
    print("=== Anchor correlation ===")
    print(corr.to_string())

    results = []
    for w in QWEN_WEIGHTS:
        r = evaluate(w, oof, pub, pri, y, train_dist)
        results.append(r)
        marker = "*" if r["test_l1"] <= SAFE_L1_CAP else " "
        print(f" {marker} qwen14b={w:.2f} e5={1-w:.2f} OOF={r['oof_qwk']:.4f} round={r['oof_round_qwk']:.4f} test_L1={r['test_l1']:.4f}")

    rows = []
    for r in results:
        rows.append({
            "name": f"e5_qwen14b_w{r['qwen_weight']:.2f}".replace(".", "p"),
            "w_e5": r["e5_weight"],
            "w_qwen14b": r["qwen_weight"],
            "oof_qwk": r["oof_qwk"],
            "oof_round_qwk": r["oof_round_qwk"],
            "test_l1": r["test_l1"],
            "oof_mae": r["oof_mae"],
            "thresholds": ",".join(f"{t:.6f}" for t in r["thresholds"]),
            "pub_dist": str(r["pub_dist"]),
            "pri_dist": str(r["pri_dist"]),
        })
    df = pd.DataFrame(rows)
    df.sort_values(["test_l1", "oof_qwk"], ascending=[True, False]).to_csv(OUT / "candidates.csv", index=False)

    e5_qwk = next(r for r in results if r["qwen_weight"] == 0.0)["oof_qwk"]
    eligible = pd.DataFrame(rows)
    eligible = eligible[(eligible["w_qwen14b"] > 0) & (eligible["test_l1"] <= SAFE_L1_CAP) & (eligible["oof_qwk"] > e5_qwk)]
    if len(eligible) == 0:
        print("\nNo Qwen14B blend improved E5 while staying inside safe L1 cap; saving lowest-L1 Qwen hedge only.")
        eligible = pd.DataFrame(rows)
        eligible = eligible[eligible["w_qwen14b"] > 0].sort_values(["test_l1", "oof_qwk"], ascending=[True, False]).head(1)
    else:
        eligible = eligible.sort_values(["oof_qwk", "test_l1"], ascending=[False, True]).head(3)

    summaries = []
    for _, row in eligible.iterrows():
        r = next(item for item in results if abs(item["qwen_weight"] - row["w_qwen14b"]) < 1e-9)
        label = f"qwen14b_w{r['qwen_weight']:.2f}".replace(".", "p")
        combo = pd.concat([
            pd.DataFrame({"id": pub["id"], "Label": r["pub_lab"]}),
            pd.DataFrame({"id": pri["id"], "Label": r["pri_lab"]}),
        ], ignore_index=True)
        sub = sample[["id"]].merge(combo, on="id", how="left")
        sub["Label"] = sub["Label"].astype(int)
        sub_path = SUBMIT_DIR / f"next_e5_qwen14b_conservative_{label}_submission.csv"
        sub.to_csv(sub_path, index=False)
        out_dir = OUT / label
        out_dir.mkdir(exist_ok=True)
        pd.DataFrame({"id": oof["id"], "Label": y, "oof_score": r["s_oof"], "oof_pred": r["oof_lab"]}).to_csv(out_dir / "oof_scores.csv", index=False)
        pd.DataFrame({"id": pub["id"], "score": r["s_pub"], "pred": r["pub_lab"]}).to_csv(out_dir / "public_scores.csv", index=False)
        pd.DataFrame({"id": pri["id"], "score": r["s_pri"], "pred": r["pri_lab"]}).to_csv(out_dir / "private_scores.csv", index=False)
        sub.to_csv(out_dir / "submission.csv", index=False)
        summaries.append({
            "label": label,
            "weights": {"e5": r["e5_weight"], "qwen14b": r["qwen_weight"]},
            "oof_qwk": r["oof_qwk"],
            "oof_round_qwk": r["oof_round_qwk"],
            "test_l1": r["test_l1"],
            "thresholds": r["thresholds"],
            "submission": str(sub_path),
        })

    (OUT / "picks_summary.json").write_text(json.dumps({
        "e5_public_lb": E5_PUBLIC_LB,
        "e5_safe_l1": E5_SAFE_L1,
        "safe_l1_cap": SAFE_L1_CAP,
        "correlation": corr.to_dict(),
        "picks": summaries,
    }, indent=2, default=str))
    print("\n=== Picks ===")
    for s in summaries:
        print(f"{s['label']}: OOF={s['oof_qwk']:.4f} round={s['oof_round_qwk']:.4f} test_L1={s['test_l1']:.4f} -> {s['submission']}")


if __name__ == "__main__":
    main()
