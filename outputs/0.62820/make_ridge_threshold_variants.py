from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error

from src.train_randomforest_best import LABEL_COLUMN, load_data, resolve_paths
from src.train_ridge_threshold_ensemble import adjacent_accuracy, scores_to_labels
from src.train_tfidf_linear_baselines import write_submission


BASE_THRESHOLDS = np.array(
    [2.1385873707649368, 2.5065516160733186, 2.880155795505213, 3.14222286850873]
)
EXTERNAL_PUBLIC_THRESHOLDS = np.array(
    [2.0713333333333335, 2.441033735387485, 2.8673333333333337, 3.132666666666667]
)


def variants() -> dict[str, np.ndarray]:
    return {
        "ridge5x5_external_thresholds": EXTERNAL_PUBLIC_THRESHOLDS,
        "ridge5x5_label4_more_small": BASE_THRESHOLDS + np.array([0.0, 0.0, -0.04, 0.04]),
        "ridge5x5_label4_more_mid": BASE_THRESHOLDS + np.array([0.0, 0.0, -0.07, 0.05]),
        "ridge5x5_label4_more_50": BASE_THRESHOLDS + np.array([0.0, 0.0, -0.08, 0.06]),
        "ridge5x5_label4_more_52": BASE_THRESHOLDS + np.array([0.0, 0.0, -0.09, 0.06]),
        "ridge5x5_label4_more_55": BASE_THRESHOLDS + np.array([0.0, 0.0, -0.10, 0.07]),
        "ridge5x5_label4_more_60": BASE_THRESHOLDS + np.array([0.0, 0.0, -0.12, 0.08]),
        "ridge5x5_label4_less_small": BASE_THRESHOLDS + np.array([0.0, 0.0, 0.04, -0.03]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create threshold-only variants from saved Ridge 5x5 scores.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/ridge_threshold_variants/submissions")
    parser.add_argument("--report-dir", default="outputs/ridge_threshold_variants/reports")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--public-test-file", default="public_test.csv")
    parser.add_argument("--private-test-file", default="private_test.csv")
    parser.add_argument("--sample-file", default="Test_Submission.csv")
    parser.add_argument("--experiment", default="ridge_threshold_5fold_5seed_891e6ee4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train_df, public_df, private_df, sample_df = load_data(paths)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    if not output_dir.is_absolute():
        output_dir = paths.project_root / output_dir
    if not report_dir.is_absolute():
        report_dir = paths.project_root / report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    ridge_report_dir = paths.project_root / "outputs" / "ridge_threshold" / "reports"
    oof = pd.read_csv(ridge_report_dir / f"{args.experiment}_oof.csv")
    public_scores = pd.read_csv(ridge_report_dir / f"{args.experiment}_public_scores.csv")
    private_scores = pd.read_csv(ridge_report_dir / f"{args.experiment}_private_scores.csv")
    y_true = train_df[LABEL_COLUMN].astype(int).to_numpy()

    rows = []
    for name, thresholds in variants().items():
        thresholds = np.sort(thresholds)
        oof_pred = scores_to_labels(oof["oof_score"].to_numpy(), thresholds)
        public_pred = scores_to_labels(public_scores["score"].to_numpy(), thresholds)
        private_pred = scores_to_labels(private_scores["score"].to_numpy(), thresholds)
        public_path, private_path, combined_path = write_submission(
            name,
            public_df,
            private_df,
            public_pred,
            private_pred,
            sample_df,
            output_dir,
        )
        rows.append(
            {
                "name": name,
                "thresholds": ",".join(f"{value:.12f}" for value in thresholds),
                "oof_qwk": cohen_kappa_score(y_true, oof_pred, weights="quadratic"),
                "mae": mean_absolute_error(y_true, oof_pred),
                "adjacent_accuracy": adjacent_accuracy(y_true, oof_pred),
                "macro_f1": f1_score(y_true, oof_pred, average="macro"),
                "label_distribution": dict(
                    pd.concat([pd.Series(public_pred), pd.Series(private_pred)])
                    .value_counts()
                    .sort_index()
                ),
                "combined_submission": combined_path,
            }
        )
        print(f"{name}: oof_qwk={rows[-1]['oof_qwk']:.6f}, dist={rows[-1]['label_distribution']}")

    report = pd.DataFrame(rows).sort_values("oof_qwk", ascending=False)
    report_path = report_dir / "ridge_threshold_variants_report.csv"
    report.to_csv(report_path, index=False)

    best = report.iloc[0]
    best_target = paths.project_root / "outputs" / "submissions" / "next_ridge_threshold_variant_submission.csv"
    best_target.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(best["combined_submission"]).to_csv(best_target, index=False)
    print(f"Best variant by OOF: {best['name']}")
    print(f"Best copy: {best_target}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
