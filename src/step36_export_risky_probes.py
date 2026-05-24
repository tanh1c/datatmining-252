from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP36_SCRIPT = ROOT / "src" / "step36_bge_m3_fine_sweep.py"
OUT = ROOT / "outputs" / "step36_bge_m3_fine_sweep"
SUBMIT_DIR = ROOT / "outputs" / "submissions"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

PROBES = [
    ("risky_high_oof_ridge_a3_w0p030_l4", "step18b_bge_m3_ridge_a3p0_w0p0300_l4p0"),
    ("risky_small_diff_ridge_a3_w0p0275_l4", "step18b_bge_m3_ridge_a3p0_w0p0275_l4p0"),
    ("risky_high_oof_ridge_a3_w0p0375_l4", "step18b_bge_m3_ridge_a3p0_w0p0375_l4p0"),
]


def load_step36_module():
    spec = importlib.util.spec_from_file_location("step36", STEP36_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Step36 script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    s36 = load_step36_module()
    candidates = pd.read_csv(OUT / "candidates.csv")
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
    ids = {"pub": raw["public"]["id"].astype(int).to_numpy(), "pri": raw["private"]["id"].astype(int).to_numpy()}

    needed_models = sorted(set(candidates[candidates["name"].isin([name for _, name in PROBES])]["model"]))
    model_scores = {model: s36.cv_predict(model, x, y, pub_x, pri_x) for model in needed_models}

    picks = []
    for label, name in PROBES:
        row = candidates[candidates["name"].eq(name)]
        if row.empty:
            raise ValueError(f"Missing candidate: {name}")
        row = row.iloc[0]
        scores = {split: (1.0 - float(row["weight"])) * step18_scores[split] + float(row["weight"]) * model_scores[row["model"]][split] for split in step18_scores}
        result = s36.evaluate(name, row["model"], float(row["weight"]), float(row["threshold_lambda"]), scores, y, train_dist, raw["train"], step18_preds, step25_preds, ids)
        pick = s36.write_pick(label, result, raw, sample, y, step25_preds)
        pick.update({"probe_note": "risky_public_probe_not_best_safe"})
        picks.append(pick)

    (OUT / "risky_probe_summary.json").write_text(json.dumps({"probes": picks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"probes": picks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
