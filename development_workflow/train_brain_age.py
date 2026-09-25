#!/usr/bin/env python3
"""Train, validate, bias-correct, and export the NCA Brain Age model."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nca_utils import (
    apply_age_bias_correction,
    dump_json,
    ensure_directory,
    fit_age_bias,
    group_kfold,
    grouped_oof_predictions,
    heldout_group_split,
    load_config,
    make_regression_pipeline,
    nested_model_benchmark,
    paired_mae_improvement,
    participant_groups,
    prepare_brain_features,
    read_table,
    regression_metrics,
    require_columns,
    set_global_seed,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared normative morphometry table")
    parser.add_argument("--config", default="config.yaml", help="Workflow YAML configuration")
    parser.add_argument("--output-dir", required=True, help="Directory for models and results")
    parser.add_argument(
        "--raw-tiv-input",
        help="Optional aligned table of raw TIV-adjusted morphometry for the normalization ablation",
    )
    return parser.parse_args()


def run(
    input_path: str,
    config_path: str,
    output_dir: str,
    raw_tiv_input: str | None = None,
) -> dict:
    config = load_config(config_path)
    project = config["project"]
    settings = config["brain_age"]
    seed = int(project["random_seed"])
    set_global_seed(seed)
    output = ensure_directory(output_dir)

    frame = read_table(input_path, project["delimiter"])
    age_column = str(project["age_column"])
    id_column = str(project["id_column"])
    require_columns(frame, [id_column, age_column], "Brain Age input")
    if frame[id_column].isna().any() or frame[age_column].isna().any():
        raise ValueError("Brain Age input contains missing identifiers or chronological ages")

    X, feature_columns = prepare_brain_features(frame, config)
    y = pd.to_numeric(frame[age_column], errors="raise").to_numpy(dtype=float)
    groups = participant_groups(frame, config)
    development_index, heldout_index = heldout_group_split(
        groups,
        float(settings["development_fraction"]),
        seed,
    )

    X_development = X.iloc[development_index].reset_index(drop=True)
    y_development = y[development_index]
    groups_development = groups[development_index]
    benchmark_predictions, benchmark_folds, benchmark_summary = nested_model_benchmark(
        X_development,
        y_development,
        groups_development,
        settings["candidate_models"],
        settings["grids"],
        int(settings["outer_folds"]),
        int(settings["inner_folds"]),
        seed,
        int(project["n_jobs"]),
    )
    write_table(benchmark_predictions, output / "brain_benchmark_oof.csv", project["delimiter"])
    write_table(benchmark_folds, output / "brain_benchmark_folds.csv", project["delimiter"])
    write_table(benchmark_summary, output / "brain_benchmark_summary.csv", project["delimiter"])

    retained = str(settings["retained_model"])
    retained_pipeline = make_regression_pipeline(retained, seed)
    inner = group_kfold(int(settings["inner_folds"]), groups_development, seed + 701)
    search = GridSearchCV(
        retained_pipeline,
        param_grid=dict(settings["grids"][retained]),
        scoring="neg_mean_absolute_error",
        cv=list(inner.split(X_development, y_development, groups_development)),
        n_jobs=int(project["n_jobs"]),
        refit=True,
        error_score="raise",
    )
    search.fit(X_development, y_development)

    development_oof, development_folds = grouped_oof_predictions(
        clone(search.best_estimator_),
        X_development,
        y_development,
        groups_development,
        int(settings["outer_folds"]),
        seed + 702,
    )
    development_correction = fit_age_bias(development_oof, y_development)
    heldout_raw = search.best_estimator_.predict(X.iloc[heldout_index])
    heldout_corrected = apply_age_bias_correction(
        heldout_raw,
        y[heldout_index],
        development_correction,
    )
    heldout = pd.DataFrame(
        {
            id_column: frame.iloc[heldout_index][id_column].to_numpy(),
            "chronological_age": y[heldout_index],
            "brain_age_raw": heldout_raw,
            "brain_age": heldout_corrected,
            "brain_gap": heldout_corrected - y[heldout_index],
        }
    )
    write_table(heldout, output / "brain_heldout_predictions.csv", project["delimiter"])

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    final_model = RidgeCV(
        alphas=tuple(float(value) for value in settings["final_ridge_alphas"]),
        cv=int(settings["final_ridge_cv"]),
        scoring="neg_mean_squared_error",
    ).fit(X_scaled, y)
    selected_alpha = float(final_model.alpha_)
    bias_estimator = Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=selected_alpha))]
    )
    full_oof_raw, full_oof_folds = grouped_oof_predictions(
        bias_estimator,
        X,
        y,
        groups,
        int(settings["final_bias_folds"]),
        seed + 703,
    )
    final_correction = fit_age_bias(full_oof_raw, y)
    full_oof_corrected = apply_age_bias_correction(full_oof_raw, y, final_correction)
    full_oof = pd.DataFrame(
        {
            id_column: frame[id_column].to_numpy(),
            "outer_fold": full_oof_folds,
            "chronological_age": y,
            "brain_age_raw": full_oof_raw,
            "brain_age": full_oof_corrected,
            "brain_gap": full_oof_corrected - y,
        }
    )
    normalization_ablation = None
    if raw_tiv_input:
        raw_frame = read_table(raw_tiv_input, project["delimiter"])
        require_columns(raw_frame, [id_column, age_column], "raw TIV comparator")
        if frame[id_column].duplicated().any() or raw_frame[id_column].duplicated().any():
            raise ValueError(
                "The optional normalization ablation requires one row per participant in both tables"
            )
        raw_frame = raw_frame.set_index(id_column).reindex(frame[id_column]).reset_index()
        if raw_frame[id_column].isna().any():
            raise ValueError("Raw TIV comparator does not contain every normative participant")
        raw_age = pd.to_numeric(raw_frame[age_column], errors="raise").to_numpy(dtype=float)
        if not np.allclose(raw_age, y):
            raise ValueError("Chronological ages differ between NOMIS and raw TIV comparator tables")
        X_raw_tiv, _ = prepare_brain_features(raw_frame, config)
        raw_tiv_oof, _ = grouped_oof_predictions(
            bias_estimator,
            X_raw_tiv,
            y,
            groups,
            int(settings["final_bias_folds"]),
            seed + 704,
        )
        full_oof["brain_age_raw_tiv_prediction"] = raw_tiv_oof
        normalization_ablation = paired_mae_improvement(
            y,
            full_oof_raw,
            raw_tiv_oof,
            int(config["evaluation"]["bootstrap_iterations"]),
            seed + 705,
            float(config["evaluation"]["confidence_level"]),
        )
    write_table(full_oof, output / "brain_age_oof.csv", project["delimiter"])

    artifact = {
        "model": final_model,
        "scaler": scaler,
        "features": feature_columns,
        "beheshti_correction": final_correction,
        "metadata": {
            "component": "brain_age",
            "random_seed": seed,
            "n_participants": int(len(np.unique(groups))),
            "n_rows": int(len(frame)),
            "n_features": int(len(feature_columns)),
            "selected_alpha": float(final_model.alpha_),
            "candidate_alphas": [float(value) for value in settings["final_ridge_alphas"]],
        },
    }
    model_path = output / "nca_brain_pipeline.joblib"
    joblib.dump(artifact, model_path)

    report = {
        "component": "brain_age",
        "input_rows": int(len(frame)),
        "unique_participants": int(len(np.unique(groups))),
        "development_rows": int(len(development_index)),
        "heldout_rows": int(len(heldout_index)),
        "features": int(len(feature_columns)),
        "retained_model": retained,
        "development_best_parameters": search.best_params_,
        "heldout_raw_metrics": regression_metrics(y[heldout_index], heldout_raw),
        "heldout_bias_corrected_metrics": regression_metrics(
            y[heldout_index], heldout_corrected
        ),
        "development_bias_correction": development_correction,
        "final_bias_correction": final_correction,
        "full_oof_raw_metrics": regression_metrics(y, full_oof_raw),
        "full_oof_bias_corrected_metrics": regression_metrics(y, full_oof_corrected),
        "nomis_vs_raw_tiv_ablation": normalization_ablation,
        "exported_model": str(model_path.name),
    }
    dump_json(report, output / "brain_training_report.json")
    return report


def main() -> None:
    args = parse_args()
    report = run(args.input, args.config, args.output_dir, args.raw_tiv_input)
    print(f"Brain Age workflow completed: {report['exported_model']}")


if __name__ == "__main__":
    main()
