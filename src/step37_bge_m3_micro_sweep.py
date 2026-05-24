from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP36_SCRIPT = ROOT / "src" / "step36_bge_m3_fine_sweep.py"
OUT = ROOT / "outputs" / "step37_bge_m3_micro_sweep"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
STEP36_BEST_SUBMISSION = SUBMIT_DIR / "next_step36_risky_small_diff_ridge_a3_w0p0275_l4_submission.csv"
OUT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

STEP36_BEST_OOF_QWK = 0.6620979854294228
WEIGHTS = [0.0260, 0.0265, 0.0270, 0.02725, 0.0275, 0.02775, 0.0280, 0.0285, 0.0290]
THRESHOLD_LAMBDAS = [2.0, 4.0, 6.0, 8.0]
MODEL_NAME = "ridge_a3p0"
PROBE_LABELS = ["best_micro", "close_oof_micro", "closest_changed_micro"]


def load_step36_module():
    spec = importlib.util.spec_from_file_location("step36", STEP36_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Step36 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_step36_best_preds(raw: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    sub = pd.read_csv(STEP36_BEST_SUBMISSION)
    return {
        "pub": raw["public"][["id"]].merge(sub, on="id", how="left")["Label"].astype(int),
        "pri": raw["private"][["id"]].merge(sub, on="id", how="left")["Label"].astype(int),
    }


def write_pick(label: str, result: dict[str, Any], raw: dict[str, pd.DataFrame], sample: pd.DataFrame, y, step25_preds: dict[str, Any], step36_preds: dict[str, pd.Series]) -> dict[str, Any]:
    out_dir = OUT / label
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"id": raw["train"]["id"], "Label": y, "oof_score": result["scores"]["oof"], "oof_pred": result["preds"]["oof"]}).to_csv(out_dir / "oof_scores.csv", index=False)
    pd.DataFrame({"id": raw["public"]["id"], "score": result["scores"]["pub"], "pred": result["preds"]["pub"]}).to_csv(out_dir / "public_scores.csv", index=False)
    pd.DataFrame({"id": raw["private"]["id"], "score": result["scores"]["pri"], "pred": result["preds"]["pri"]}).to_csv(out_dir / "private_scores.csv", index=False)

    for split, key in [("public", "pub"), ("private", "pri")]:
        step36_base = step36_preds[key].to_numpy(int)
        changed_25 = result["preds"][key] != step25_preds[key]
        changed_36 = result["preds"][key] != step36_base
        raw[split].loc[changed_25, ["id", "title", "venue"]].assign(step25_pred=step25_preds[key][changed_25], step37_pred=result["preds"][key][changed_25]).to_csv(out_dir / f"changed_vs_step25_{split}.csv", index=False)
        raw[split].loc[changed_36, ["id", "title", "venue"]].assign(step36_pred=step36_base[changed_36], step37_pred=result["preds"][key][changed_36]).to_csv(out_dir / f"changed_vs_step36_{split}.csv", index=False)

    combo = pd.concat([
        pd.DataFrame({"id": raw["public"]["id"], "Label": result["preds"]["pub"]}),
        pd.DataFrame({"id": raw["private"]["id"], "Label": result["preds"]["pri"]}),
    ], ignore_index=True)
    sub = sample[["id"]].merge(combo, on="id", how="left")
    sub["Label"] = sub["Label"].astype(int)
    path = SUBMIT_DIR / f"next_step37_{label}_submission.csv"
    sub.to_csv(path, index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)
    return {
        "label": label,
        "name": result["name"],
        "submission": str(path),
        "oof_qwk": result["oof_qwk"],
        "oof_lift_vs_step25": result["oof_lift_vs_step25"],
        "oof_lift_vs_step36": result["oof_qwk"] - STEP36_BEST_OOF_QWK,
        "test_l1": result["test_l1"],
        "diff_vs_step25_total": result["diff_vs_step25_total"],
        "diff_vs_step36_total": result["diff_vs_step36_total"],
    }


def main() -> None:
    s36 = load_step36_module()
    raw = {"train": pd.read_csv(s36.DATA_DIR / "train.csv"), "public": pd.read_csv(s36.DATA_DIR / "public_test.csv"), "private": pd.read_csv(s36.DATA_DIR / "private_test.csv")}
    raw["train"] = s36.context_flags(raw["train"])
    sample = pd.read_csv(s36.DATA_DIR / "Test_Submission.csv")
    y = raw["train"]["Label"].astype(int).to_numpy()
    train_dist = pd.Series(y).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()

    train_ids, train_dense = s36.load_feature_split("train")
    pub_ids, pub_dense = s36.load_feature_split("public")
    pri_ids, pri_dense = s36.load_feature_split("private")
    x = s36.make_matrix(train_ids, train_dense)
    pub_x = s36.make_matrix(pub_ids, pub_dense)
    pri_x = s36.make_matrix(pri_ids, pri_dense)

    step18_frames = {split: raw[split][["id"]].merge(s36.load_scores(split, s36.STEP18B_DIR, "step18b"), on="id", how="left") for split in ["train", "public", "private"]}
    step25_frames = {split: raw[split][["id"]].merge(s36.load_scores(split, s36.STEP25_DIR, "step25"), on="id", how="left") for split in ["train", "public", "private"]}
    step18_scores = {"oof": step18_frames["train"]["step18b_score"].to_numpy(float), "pub": step18_frames["public"]["step18b_score"].to_numpy(float), "pri": step18_frames["private"]["step18b_score"].to_numpy(float)}
    step18_preds = {"oof": step18_frames["train"]["step18b_pred"].astype(int).to_numpy(), "pub": step18_frames["public"]["step18b_pred"].astype(int).to_numpy(), "pri": step18_frames["private"]["step18b_pred"].astype(int).to_numpy()}
    step25_preds = {"oof": step25_frames["train"]["step25_pred"].astype(int).to_numpy(), "pub": step25_frames["public"]["step25_pred"].astype(int).to_numpy(), "pri": step25_frames["private"]["step25_pred"].astype(int).to_numpy()}
    step36_preds = load_step36_best_preds(raw)
    ids = {"pub": raw["public"]["id"].astype(int).to_numpy(), "pri": raw["private"]["id"].astype(int).to_numpy()}

    model_scores = s36.cv_predict(MODEL_NAME, x, y, pub_x, pri_x)
    results = []
    result_by_name = {}
    for weight in WEIGHTS:
        for lambd in THRESHOLD_LAMBDAS:
            scores = {split: (1.0 - weight) * step18_scores[split] + weight * model_scores[split] for split in step18_scores}
            name = f"step18b_bge_m3_{MODEL_NAME}_w{weight:.5f}_l{lambd}".replace(".", "p")
            result = s36.evaluate(name, MODEL_NAME, weight, lambd, scores, y, train_dist, raw["train"], step18_preds, step25_preds, ids)
            result["diff_vs_step36_public"] = int((result["preds"]["pub"] != step36_preds["pub"].to_numpy(int)).sum())
            result["diff_vs_step36_private"] = int((result["preds"]["pri"] != step36_preds["pri"].to_numpy(int)).sum())
            result["diff_vs_step36_total"] = result["diff_vs_step36_public"] + result["diff_vs_step36_private"]
            result["oof_lift_vs_step36"] = float(result["oof_qwk"] - STEP36_BEST_OOF_QWK)
            results.append(result)
            result_by_name[name] = result

    rows = []
    for result in results:
        rows.append({k: v for k, v in result.items() if k not in {"scores", "preds", "thresholds"}} | {"thresholds": ",".join(f"{x:.6f}" for x in result["thresholds"]), "pub_dist": str(result["pub_dist"]), "pri_dist": str(result["pri_dist"])})
    candidates = pd.DataFrame(rows).sort_values(["oof_lift_vs_step36", "diff_vs_step36_total", "diff_vs_step25_total"], ascending=[False, True, True])
    candidates.to_csv(OUT / "candidates.csv", index=False)

    probe_pool = candidates[
        (candidates["forbidden_vs_step25"] == 0)
        & (candidates["diff_vs_step25_total"] <= 9)
        & (candidates["diff_vs_step36_total"].between(1, 6))
        & (candidates["oof_lift_vs_step25"] > 0)
    ].copy()
    probe_pool["rank_score"] = probe_pool["oof_lift_vs_step36"] - 0.00012 * probe_pool["diff_vs_step36_total"] - 0.00005 * probe_pool["diff_vs_step25_total"]
    probe_pool = probe_pool.sort_values(["rank_score", "oof_lift_vs_step36"], ascending=[False, False])
    probe_pool.to_csv(OUT / "probe_candidates.csv", index=False)

    chosen_rows = []
    if len(probe_pool):
        chosen_rows.append(probe_pool.iloc[0])
        best_oof = probe_pool.sort_values(["oof_lift_vs_step36", "diff_vs_step36_total"], ascending=[False, True]).iloc[0]
        closest = probe_pool.sort_values(["diff_vs_step36_total", "oof_lift_vs_step25"], ascending=[True, False]).iloc[0]
        for row in [best_oof, closest]:
            if row["name"] not in {r["name"] for r in chosen_rows}:
                chosen_rows.append(row)
    picks = []
    for label, row in zip(PROBE_LABELS, chosen_rows):
        picks.append(write_pick(label, result_by_name[row["name"]], raw, sample, y, step25_preds, step36_preds))

    show_cols = ["name", "weight", "threshold_lambda", "oof_qwk", "oof_lift_vs_step25", "oof_lift_vs_step36", "test_l1", "diff_vs_step25_total", "diff_vs_step36_total", "risky_promotions", "core_demotions", "public_changed", "private_changed", "forbidden_vs_step25"]
    top = candidates.head(20)
    lines = ["# Step37 BGE-M3 micro sweep around Step36", ""]
    lines.append("This step sweeps tiny `ridge_a3p0` weights around the Step36 public-best `w=0.0275` and compares candidates against both Step25 and Step36 best.")
    lines.extend(["", "## Top candidates", "", top[show_cols].to_markdown(index=False), "", "## Probe candidates", "", probe_pool[show_cols].head(20).to_markdown(index=False) if len(probe_pool) else "No probe candidates passed filters.", "", "## Picks", "", "```json", json.dumps(picks, indent=2, ensure_ascii=False), "```", ""])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"step36_best_oof_qwk": STEP36_BEST_OOF_QWK, "probe_candidates": int(len(probe_pool)), "picks": picks}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Step37 BGE-M3 micro sweep ===")
    print(top[show_cols].head(15).to_string(index=False))
    print("\n=== Probe candidates ===")
    print(probe_pool[show_cols].head(10).to_string(index=False) if len(probe_pool) else "none")
    print("\n=== Summary ===")
    print(json.dumps({"probe_candidates": int(len(probe_pool)), "picks": picks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
