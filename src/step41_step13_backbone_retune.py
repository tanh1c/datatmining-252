from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP13_SCRIPT = ROOT / "src" / "step13_e5_anchor_stack.py"
STEP18B_SCRIPT = ROOT / "src" / "step18b_targeted_feature_calibration.py"
STEP36_SCRIPT = ROOT / "src" / "step36_bge_m3_fine_sweep.py"
OUT = ROOT / "outputs" / "step41_step13_backbone_retune"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
STEP39_BEST_SUBMISSION = SUBMIT_DIR / "next_step39_best_backbone_oof_submission.csv"
STEP36_BEST_SUBMISSION = SUBMIT_DIR / "next_step36_risky_small_diff_ridge_a3_w0p0275_l4_submission.csv"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

STEP39_BEST_OOF_QWK = 0.6625861066095553
STEP36_BEST_OOF_QWK = 0.6620979854294228
STEP25_OOF_QWK = 0.6612605583792329
BGE_WEIGHT = 0.0275
BGE_MODEL_NAME = "ridge_a3p0"
THRESHOLD_LAMBDAS = [4.0, 6.0]
STEP39_CALIBRATOR_WEIGHT = 0.10
PROBE_LABELS = ["best_backbone_oof", "closest_backbone_probe", "balanced_backbone_probe"]
FORCED_EXPORTS = {
    "high_e5_scincl_ridge0": (0.55, 0.35, 0.10, 0.00),
    "higher_e5_ridge0": (0.60, 0.30, 0.10, 0.00),
    "balanced_high_e5_ridge0": (0.55, 0.30, 0.15, 0.00),
    "specter25_from_e5_ridge0": (0.45, 0.30, 0.25, 0.00),
    "specter25_from_scincl_ridge0": (0.50, 0.25, 0.25, 0.00),
    "specter30_balanced_ridge0": (0.40, 0.30, 0.30, 0.00),
    "specter30_from_scincl_ridge0": (0.50, 0.20, 0.30, 0.00),
}


EXTRA_WEIGHT_GRID = [
    (0.42, 0.38, 0.15, 0.05),
    (0.43, 0.37, 0.15, 0.05),
    (0.44, 0.36, 0.15, 0.05),
    (0.46, 0.34, 0.15, 0.05),
    (0.47, 0.33, 0.15, 0.05),
    (0.48, 0.32, 0.15, 0.05),
    (0.49, 0.31, 0.15, 0.05),
    (0.50, 0.32, 0.13, 0.05),
    (0.50, 0.33, 0.12, 0.05),
    (0.50, 0.34, 0.11, 0.05),
    (0.48, 0.35, 0.12, 0.05),
    (0.48, 0.34, 0.13, 0.05),
    (0.46, 0.37, 0.12, 0.05),
    (0.46, 0.36, 0.13, 0.05),
    (0.45, 0.37, 0.13, 0.05),
    (0.45, 0.36, 0.14, 0.05),
    (0.45, 0.38, 0.12, 0.05),
    (0.45, 0.35, 0.15, 0.05),
    (0.45, 0.35, 0.14, 0.06),
    (0.45, 0.35, 0.13, 0.07),
    (0.45, 0.35, 0.12, 0.08),
    (0.45, 0.35, 0.11, 0.09),
    (0.45, 0.35, 0.10, 0.10),
    (0.50, 0.30, 0.14, 0.06),
    (0.50, 0.30, 0.13, 0.07),
    (0.50, 0.30, 0.12, 0.08),
    (0.50, 0.30, 0.11, 0.09),
    *FORCED_EXPORTS.values(),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_submission_preds(path: Path, raw: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    sub = pd.read_csv(path)
    return {
        "pub": raw["public"][["id"]].merge(sub, on="id", how="left")["Label"].astype(int),
        "pri": raw["private"][["id"]].merge(sub, on="id", how="left")["Label"].astype(int),
    }


def unique_weight_grid(s13) -> list[tuple[float, float, float, float]]:
    seen = set()
    weights = []
    for candidate in list(s13.make_grid()) + EXTRA_WEIGHT_GRID:
        candidate = tuple(round(float(x), 4) for x in candidate)
        if abs(sum(candidate) - 1.0) > 1e-6:
            continue
        if candidate[3] > 0.10 + 1e-6:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        weights.append(candidate)
    return weights


def blend_step13_scores(weights: tuple[float, float, float, float], oof: pd.DataFrame, pub: pd.DataFrame, pri: pd.DataFrame) -> dict[str, np.ndarray]:
    w_e5, w_scincl, w_specter, w_ridge = weights
    return {
        "oof": (w_e5 * oof["e5"] + w_scincl * oof["scincl"] + w_specter * oof["specter"] + w_ridge * oof["ridge"]).to_numpy(float),
        "pub": (w_e5 * pub["e5"] + w_scincl * pub["scincl"] + w_specter * pub["specter"] + w_ridge * pub["ridge"]).to_numpy(float),
        "pri": (w_e5 * pri["e5"] + w_scincl * pri["scincl"] + w_specter * pri["specter"] + w_ridge * pri["ridge"]).to_numpy(float),
    }


def write_pick(label: str, result: dict[str, Any], raw: dict[str, pd.DataFrame], sample: pd.DataFrame, y: np.ndarray, step25_preds: dict[str, Any], step36_preds: dict[str, pd.Series], step39_preds: dict[str, pd.Series]) -> dict[str, Any]:
    out_dir = OUT / label
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"id": raw["train"]["id"], "Label": y, "oof_score": result["scores"]["oof"], "oof_pred": result["preds"]["oof"]}).to_csv(out_dir / "oof_scores.csv", index=False)
    pd.DataFrame({"id": raw["public"]["id"], "score": result["scores"]["pub"], "pred": result["preds"]["pub"]}).to_csv(out_dir / "public_scores.csv", index=False)
    pd.DataFrame({"id": raw["private"]["id"], "score": result["scores"]["pri"], "pred": result["preds"]["pri"]}).to_csv(out_dir / "private_scores.csv", index=False)

    for split, key in [("public", "pub"), ("private", "pri")]:
        step36_base = step36_preds[key].to_numpy(int)
        step39_base = step39_preds[key].to_numpy(int)
        changed_25 = result["preds"][key] != step25_preds[key]
        changed_36 = result["preds"][key] != step36_base
        changed_39 = result["preds"][key] != step39_base
        raw[split].loc[changed_25, ["id", "title", "venue"]].assign(step25_pred=step25_preds[key][changed_25], step41_pred=result["preds"][key][changed_25]).to_csv(out_dir / f"changed_vs_step25_{split}.csv", index=False)
        raw[split].loc[changed_36, ["id", "title", "venue"]].assign(step36_pred=step36_base[changed_36], step41_pred=result["preds"][key][changed_36]).to_csv(out_dir / f"changed_vs_step36_{split}.csv", index=False)
        raw[split].loc[changed_39, ["id", "title", "venue"]].assign(step39_pred=step39_base[changed_39], step41_pred=result["preds"][key][changed_39]).to_csv(out_dir / f"changed_vs_step39_{split}.csv", index=False)

    combo = pd.concat([
        pd.DataFrame({"id": raw["public"]["id"], "Label": result["preds"]["pub"]}),
        pd.DataFrame({"id": raw["private"]["id"], "Label": result["preds"]["pri"]}),
    ], ignore_index=True)
    sub = sample[["id"]].merge(combo, on="id", how="left")
    sub["Label"] = sub["Label"].astype(int)
    path = SUBMIT_DIR / f"next_step41_{label}_submission.csv"
    sub.to_csv(path, index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)
    return {
        "label": label,
        "name": result["name"],
        "submission": str(path),
        "w_e5": result["w_e5"],
        "w_scincl": result["w_scincl"],
        "w_specter": result["w_specter"],
        "w_ridge": result["w_ridge"],
        "threshold_lambda": result["threshold_lambda"],
        "oof_qwk": result["oof_qwk"],
        "oof_lift_vs_step39": result["oof_qwk"] - STEP39_BEST_OOF_QWK,
        "oof_lift_vs_step36": result["oof_qwk"] - STEP36_BEST_OOF_QWK,
        "oof_lift_vs_step25": result["oof_lift_vs_step25"],
        "test_l1": result["test_l1"],
        "diff_vs_step39_total": result["diff_vs_step39_total"],
        "diff_vs_step36_total": result["diff_vs_step36_total"],
        "diff_vs_step25_total": result["diff_vs_step25_total"],
    }


def add_unique_pick(chosen_rows: list[pd.Series], row: pd.Series) -> None:
    if row["name"] not in {chosen["name"] for chosen in chosen_rows}:
        chosen_rows.append(row)


def main() -> None:
    s13 = load_module("step13", STEP13_SCRIPT)
    s18 = load_module("step18b", STEP18B_SCRIPT)
    s36 = load_module("step36", STEP36_SCRIPT)

    step13_oof, step13_pub, step13_pri, sample = s13.load()
    train, public, private, _ = s18.load_data()
    cols = s18.feature_columns(train)
    for df in [public, private]:
        s18.feature_columns(df)
    y = train["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    x = train[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    pub_x = public[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    pri_x = private[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    meta_oof, meta_pub, meta_pri = s18.cv_predict("ridge_a10", x, y.astype(float), pub_x, pri_x)
    ridge_a10_scores = {"oof": meta_oof, "pub": meta_pub, "pri": meta_pri}

    raw = {"train": pd.read_csv(s36.DATA_DIR / "train.csv"), "public": pd.read_csv(s36.DATA_DIR / "public_test.csv"), "private": pd.read_csv(s36.DATA_DIR / "private_test.csv")}
    raw["train"] = s36.context_flags(raw["train"])
    train_ids, train_dense = s36.load_feature_split("train")
    pub_ids, pub_dense = s36.load_feature_split("public")
    pri_ids, pri_dense = s36.load_feature_split("private")
    bge_x = s36.make_matrix(train_ids, train_dense)
    bge_pub_x = s36.make_matrix(pub_ids, pub_dense)
    bge_pri_x = s36.make_matrix(pri_ids, pri_dense)
    bge_scores = s36.cv_predict(BGE_MODEL_NAME, bge_x, y, bge_pub_x, bge_pri_x)

    step18_frames = {split: raw[split][["id"]].merge(s36.load_scores(split, s36.STEP18B_DIR, "step18b"), on="id", how="left") for split in ["train", "public", "private"]}
    step25_frames = {split: raw[split][["id"]].merge(s36.load_scores(split, s36.STEP25_DIR, "step25"), on="id", how="left") for split in ["train", "public", "private"]}
    step18_preds = {"oof": step18_frames["train"]["step18b_pred"].astype(int).to_numpy(), "pub": step18_frames["public"]["step18b_pred"].astype(int).to_numpy(), "pri": step18_frames["private"]["step18b_pred"].astype(int).to_numpy()}
    step25_preds = {"oof": step25_frames["train"]["step25_pred"].astype(int).to_numpy(), "pub": step25_frames["public"]["step25_pred"].astype(int).to_numpy(), "pri": step25_frames["private"]["step25_pred"].astype(int).to_numpy()}
    step36_preds = load_submission_preds(STEP36_BEST_SUBMISSION, raw)
    step39_preds = load_submission_preds(STEP39_BEST_SUBMISSION, raw)
    ids = {"pub": raw["public"]["id"].astype(int).to_numpy(), "pri": raw["private"]["id"].astype(int).to_numpy()}

    results = []
    result_by_name = {}
    for weights in unique_weight_grid(s13):
        step13_scores = blend_step13_scores(weights, step13_oof, step13_pub, step13_pri)
        step39_style_backbone = {split: (1.0 - STEP39_CALIBRATOR_WEIGHT) * step13_scores[split] + STEP39_CALIBRATOR_WEIGHT * ridge_a10_scores[split] for split in step13_scores}
        final_scores = {split: (1.0 - BGE_WEIGHT) * step39_style_backbone[split] + BGE_WEIGHT * bge_scores[split] for split in step39_style_backbone}
        for lambd in THRESHOLD_LAMBDAS:
            name = f"step13retune_e5{weights[0]:.2f}_scincl{weights[1]:.2f}_specter{weights[2]:.2f}_ridge{weights[3]:.2f}_bge{BGE_WEIGHT:.4f}_l{lambd}".replace(".", "p")
            result = s36.evaluate(name, BGE_MODEL_NAME, BGE_WEIGHT, lambd, final_scores, y, train_dist, raw["train"], step18_preds, step25_preds, ids)
            result["w_e5"], result["w_scincl"], result["w_specter"], result["w_ridge"] = weights
            result["threshold_lambda"] = lambd
            result["diff_vs_step36_public"] = int((result["preds"]["pub"] != step36_preds["pub"].to_numpy(int)).sum())
            result["diff_vs_step36_private"] = int((result["preds"]["pri"] != step36_preds["pri"].to_numpy(int)).sum())
            result["diff_vs_step36_total"] = result["diff_vs_step36_public"] + result["diff_vs_step36_private"]
            result["diff_vs_step39_public"] = int((result["preds"]["pub"] != step39_preds["pub"].to_numpy(int)).sum())
            result["diff_vs_step39_private"] = int((result["preds"]["pri"] != step39_preds["pri"].to_numpy(int)).sum())
            result["diff_vs_step39_total"] = result["diff_vs_step39_public"] + result["diff_vs_step39_private"]
            result["oof_lift_vs_step36"] = float(result["oof_qwk"] - STEP36_BEST_OOF_QWK)
            result["oof_lift_vs_step39"] = float(result["oof_qwk"] - STEP39_BEST_OOF_QWK)
            results.append(result)
            result_by_name[name] = result

    rows = []
    for result in results:
        rows.append({k: v for k, v in result.items() if k not in {"scores", "preds", "thresholds"}} | {"thresholds": ",".join(f"{x:.6f}" for x in result["thresholds"]), "pub_dist": str(result["pub_dist"]), "pri_dist": str(result["pri_dist"])})
    candidates = pd.DataFrame(rows).sort_values(["oof_lift_vs_step39", "diff_vs_step39_total", "diff_vs_step36_total"], ascending=[False, True, True])
    candidates.to_csv(OUT / "candidates.csv", index=False)

    probe_pool = candidates[
        (candidates["forbidden_vs_step25"] == 0)
        & (candidates["oof_lift_vs_step25"] > 0)
        & (candidates["diff_vs_step39_total"].between(1, 6))
    ].copy()
    probe_pool["rank_score"] = probe_pool["oof_lift_vs_step39"] - 0.00014 * probe_pool["diff_vs_step39_total"] - 0.00006 * probe_pool["diff_vs_step36_total"] - 0.00003 * probe_pool["risky_promotions"]
    probe_pool = probe_pool.sort_values(["rank_score", "oof_lift_vs_step39"], ascending=[False, False])
    probe_pool.to_csv(OUT / "probe_candidates.csv", index=False)

    chosen_rows = []
    if len(probe_pool):
        best_oof = probe_pool.sort_values(["oof_lift_vs_step39", "diff_vs_step39_total"], ascending=[False, True]).iloc[0]
        closest = probe_pool.sort_values(["diff_vs_step39_total", "oof_lift_vs_step39"], ascending=[True, False]).iloc[0]
        balanced = probe_pool.iloc[0]
        for row in [best_oof, closest, balanced]:
            add_unique_pick(chosen_rows, row)
    picks = []
    for label, row in zip(PROBE_LABELS, chosen_rows):
        picks.append(write_pick(label, result_by_name[row["name"]], raw, sample, y, step25_preds, step36_preds, step39_preds))

    forced_picks = []
    for label, weights in FORCED_EXPORTS.items():
        forced_matches = candidates[
            (candidates["w_e5"].round(4) == weights[0])
            & (candidates["w_scincl"].round(4) == weights[1])
            & (candidates["w_specter"].round(4) == weights[2])
            & (candidates["w_ridge"].round(4) == weights[3])
            & (candidates["threshold_lambda"] == 4.0)
        ]
        if len(forced_matches):
            forced_picks.append(write_pick(label, result_by_name[forced_matches.iloc[0]["name"]], raw, sample, y, step25_preds, step36_preds, step39_preds))

    show_cols = ["name", "w_e5", "w_scincl", "w_specter", "w_ridge", "threshold_lambda", "oof_qwk", "oof_lift_vs_step39", "oof_lift_vs_step36", "oof_lift_vs_step25", "test_l1", "diff_vs_step39_total", "diff_vs_step36_total", "diff_vs_step25_total", "risky_promotions", "core_demotions", "public_changed", "private_changed", "forbidden_vs_step25"]
    top = candidates.head(20)
    lines = ["# Step41 Step13 backbone retune", ""]
    lines.append("This step retunes the Step13 E5/SciNCL/SPECTER/Ridge backbone while keeping the Step39-style Ridge(alpha=10) calibrator share and fixed BGE-M3 diagnostic at `w=0.0275`.")
    lines.extend(["", "## Top candidates", "", top[show_cols].to_markdown(index=False), "", "## Probe candidates", "", probe_pool[show_cols + ["rank_score"]].head(20).to_markdown(index=False) if len(probe_pool) else "No probe candidates passed filters.", "", "## Picks", "", "```json", json.dumps(picks, indent=2, ensure_ascii=False), "```", "", "## Forced exports", "", "```json", json.dumps(forced_picks, indent=2, ensure_ascii=False), "```", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"step39_best_oof_qwk": STEP39_BEST_OOF_QWK, "step36_best_oof_qwk": STEP36_BEST_OOF_QWK, "step25_oof_qwk": STEP25_OOF_QWK, "probe_candidates": int(len(probe_pool)), "picks": picks, "forced_picks": forced_picks}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Step41 Step13 backbone retune ===")
    print(top[show_cols].head(15).to_string(index=False))
    print("\n=== Probe candidates ===")
    print(probe_pool[show_cols + ["rank_score"]].head(10).to_string(index=False) if len(probe_pool) else "none")
    print("\n=== Summary ===")
    print(json.dumps({"probe_candidates": int(len(probe_pool)), "picks": picks, "forced_picks": forced_picks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
