#!/usr/bin/env python3
"""Train, validate, bias-correct, and export the NCA Cognitive Age model."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from nca_utils import (
    apply_age_bias_correction,
    dump_json,
    ensure_directory,
    fit_age_bias,
    grouped_oof_predictions,
    load_config,
    nested_model_benchmark,
    paired_mae_improvement,
    participant_groups,
    prepare_cognitive_features,
    read_table,
    regression_metrics,
    require_columns,
    set_global_seed,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared normative cognitive table")
    parser.add_argument("--config", default="config.yaml", help="Workflow YAML configuration")
    parser.add_argument("--output-dir", required=True, help="Directory for models and results")
    return parser.parse_args()


def _fixed_svr_pipeline(settings: dict) -> Pipeline:
    final = settings["final_svr"]
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="linear",
                    C=float(final["C"]),
                    epsilon=float(final["epsilon"]),
                    tol=float(final["tolerance"]),
                ),
            ),
        ]
    )


def run(input_path: str, config_path: str, output_dir: str) -> dict:
    config = load_config(config_path)
    project = config["project"]
    settings = config["cognitive_age"]
    seed = int(project["random_seed"])
    set_global_seed(seed)
    output = ensure_directory(output_dir)

    frame = read_table(input_path, project["delimiter"])
    age_column = str(project["age_column"])
    id_column = str(project["id_column"])
    require_columns(frame, [id_column, age_column], "Cognitive Age input")
    if frame[id_column].isna().any() or frame[age_column].isna().any():
        raise ValueError("Cognitive Age input contains missing identifiers or chronological ages")

    X = prepare_cognitive_features(frame, config)
    feature_columns = list(X.columns)
    y = pd.to_numeric(frame[age_column], errors="raise").to_numpy(dtype=float)
    groups = participant_groups(frame, config)

    benchmark_predictions, benchmark_folds, benchmark_summary = nested_model_benchmark(
        X,
        y,
        groups,
        settings["candidate_models"],
        settings["grids"],
        int(settings["outer_folds"]),
        int(settings["inner_folds"]),
        seed,
        int(project["n_jobs"]),
    )
    write_table(benchmark_predictions, output / "cognitive_benchmark_oof.csv", project["delimiter"])
    write_table(benchmark_folds, output / "cognitive_benchmark_folds.csv", project["delimiter"])
    write_table(benchmark_summary, output / "cognitive_benchmark_summary.csv", project["delimiter"])

    retained_pipeline = _fixed_svr_pipeline(settings)
    boxcox_oof_raw, oof_folds = grouped_oof_predictions(
        retained_pipeline,
        X,
        y,
        groups,
        int(settings["final_bias_folds"]),
        seed + 801,
    )
    correction = fit_age_bias(boxcox_oof_raw, y)
    boxcox_oof_corrected = apply_age_bias_correction(boxcox_oof_raw, y, correction)

    X_zscore = X.copy()
    fluency_column = settings["fluency_column"]
    X_zscore["fluency_bc"] = pd.to_numeric(frame[fluency_column], errors="raise").to_numpy(
        dtype=float
    )
    zscore_oof_raw, _ = grouped_oof_predictions(
        retained_pipeline,
        X_zscore,
        y,
        groups,
        int(settings["final_bias_folds"]),
        seed + 802,
    )

    oof = pd.DataFrame(
        {
            id_column: frame[id_column].to_numpy(),
            "outer_fold": oof_folds,
            "chronological_age": y,
            "cognitive_age_raw": boxcox_oof_raw,
            "cognitive_age": boxcox_oof_corrected,
            "cognitive_gap": boxcox_oof_corrected - y,
            "cognitive_age_zscore_only_raw": zscore_oof_raw,
        }
    )
    write_table(oof, output / "cognitive_age_oof.csv", project["delimiter"])

    scaler = StandardScaler().fit(X)
    final = settings["final_svr"]
    model = SVR(
        kernel="linear",
        C=float(final["C"]),
        epsilon=float(final["epsilon"]),
        tol=float(final["tolerance"]),
    ).fit(scaler.transform(X), y)
    artifact = {
        "model": model,
        "scaler": scaler,
        "features": feature_columns,
        "lambda_bc": float(settings["boxcox_lambda"]),
        "bias_correction": correction,
        "metadata": {
            "component": "cognitive_age",
            "random_seed": seed,
            "n_participants": int(len(np.unique(groups))),
            "n_rows": int(len(frame)),
            "features": feature_columns,
            "C": float(final["C"]),
            "epsilon": float(final["epsilon"]),
            "tolerance": float(final["tolerance"]),
        },
    }
    model_path = output / "nca_cognitive_pipeline.joblib"
    joblib.dump(artifact, model_path)

    transformation_ablation = paired_mae_improvement(
        y,
        boxcox_oof_raw,
        zscore_oof_raw,
        int(config["evaluation"]["bootstrap_iterations"]),
        seed + 803,
        float(config["evaluation"]["confidence_level"]),
    )
    report = {
        "component": "cognitive_age",
        "input_rows": int(len(frame)),
        "unique_participants": int(len(np.unique(groups))),
        "features": feature_columns,
        "retained_model": settings["retained_model"],
        "retained_parameters": final,
        "boxcox_lambda": float(settings["boxcox_lambda"]),
        "bias_correction": correction,
        "oof_raw_metrics": regression_metrics(y, boxcox_oof_raw),
        "oof_bias_corrected_metrics": regression_metrics(y, boxcox_oof_corrected),
        "zscore_only_oof_metrics": regression_metrics(y, zscore_oof_raw),
        "boxcox_vs_zscore_ablation": transformation_ablation,
        "exported_model": str(model_path.name),
    }
    dump_json(report, output / "cognitive_training_report.json")
    return report


def main() -> None:
    args = parse_args()
    report = run(args.input, args.config, args.output_dir)
    print(f"Cognitive Age workflow completed: {report['exported_model']}")


if __name__ == "__main__":
    main()
