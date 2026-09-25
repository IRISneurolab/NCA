#!/usr/bin/env python3
"""Fit and evaluate the seven NCA fusion architectures with nested CV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nca_utils import (
    diagnostic_labels,
    dump_json,
    ensure_directory,
    load_artifact,
    load_config,
    participant_groups,
    predict_brain_age,
    predict_cognitive_age,
    read_table,
    require_columns,
    safe_auc,
    set_global_seed,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Participant-disjoint evaluation table")
    parser.add_argument("--config", default="config.yaml", help="Workflow YAML configuration")
    parser.add_argument("--output-dir", required=True, help="Directory for fusion outputs")
    parser.add_argument("--brain-model", help="Brain Age joblib; optional when brain_age is present")
    parser.add_argument(
        "--cognitive-model", help="Cognitive Age joblib; optional when cognitive_age is present"
    )
    return parser.parse_args()


def _positive_moca_weights(ba: np.ndarray, ca: np.ndarray, moca: np.ndarray) -> np.ndarray:
    X = np.column_stack([ba, ca]).astype(float)
    X = StandardScaler().fit_transform(X)
    target = -StandardScaler().fit_transform(np.asarray(moca).reshape(-1, 1)).ravel()
    coefficients = LinearRegression(positive=True).fit(X, target).coef_
    total = float(np.sum(coefficients))
    if not np.isfinite(total) or total <= 0:
        return np.asarray([0.5, 0.5], dtype=float)
    return coefficients / total


@dataclass
class FusionModel:
    strategy: str
    settings: dict[str, Any]

    def fit(
        self,
        brain_age: np.ndarray,
        cognitive_age: np.ndarray,
        chronological_age: np.ndarray,
        moca: np.ndarray,
    ) -> "FusionModel":
        self.brain_age_mean_ = float(np.mean(brain_age))
        self.brain_age_sd_ = float(np.std(brain_age, ddof=0)) or 1.0
        self.cognitive_age_mean_ = float(np.mean(cognitive_age))
        self.cognitive_age_sd_ = float(np.std(cognitive_age, ddof=0)) or 1.0
        self.weights_ = _positive_moca_weights(brain_age, cognitive_age, moca)

        if self.strategy == "exponential":
            raw = self._exponential_raw(brain_age, cognitive_age)
            self.calibrator_ = LinearRegression().fit(raw.reshape(-1, 1), chronological_age)
        elif self.strategy == "logarithmic":
            raw = self._logarithmic_raw(brain_age, cognitive_age)
            self.calibrator_ = LinearRegression().fit(raw.reshape(-1, 1), chronological_age)
        elif self.strategy == "linear_regression":
            self.estimator_ = LinearRegression().fit(
                np.column_stack([brain_age, cognitive_age]), chronological_age
            )
        elif self.strategy == "penalized_splines":
            self.estimator_ = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "splines",
                        SplineTransformer(
                            n_knots=int(self.settings["spline_knots"]),
                            degree=int(self.settings["spline_degree"]),
                            include_bias=False,
                        ),
                    ),
                    (
                        "ridge",
                        RidgeCV(
                            alphas=tuple(
                                float(value)
                                for value in self.settings["spline_ridge_alphas"]
                            )
                        ),
                    ),
                ]
            ).fit(np.column_stack([brain_age, cognitive_age]), chronological_age)
        return self

    def _standardized(self, ba: np.ndarray, ca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            (np.asarray(ba, dtype=float) - self.brain_age_mean_) / self.brain_age_sd_,
            (np.asarray(ca, dtype=float) - self.cognitive_age_mean_) / self.cognitive_age_sd_,
        )

    def _exponential_raw(self, ba: np.ndarray, ca: np.ndarray) -> np.ndarray:
        z_ba, z_ca = self._standardized(ba, ca)
        linear = self.weights_[0] * z_ba + self.weights_[1] * z_ca
        return np.exp(np.clip(linear, -5.0, 5.0))

    @staticmethod
    def _logarithmic_raw(ba: np.ndarray, ca: np.ndarray) -> np.ndarray:
        summed = np.asarray(ba, dtype=float) + np.asarray(ca, dtype=float)
        return np.log(np.clip(summed, 1e-6, None))

    def predict(self, brain_age: np.ndarray, cognitive_age: np.ndarray) -> np.ndarray:
        ba = np.asarray(brain_age, dtype=float)
        ca = np.asarray(cognitive_age, dtype=float)
        if self.strategy == "simple_mean":
            return 0.5 * ba + 0.5 * ca
        if self.strategy == "weighted_mean":
            return self.weights_[0] * ba + self.weights_[1] * ca
        if self.strategy == "euclidean":
            return np.sqrt((np.square(ba) + np.square(ca)) / 2.0)
        if self.strategy == "exponential":
            raw = self._exponential_raw(ba, ca)
            return self.calibrator_.predict(raw.reshape(-1, 1))
        if self.strategy == "logarithmic":
            raw = self._logarithmic_raw(ba, ca)
            return self.calibrator_.predict(raw.reshape(-1, 1))
        if self.strategy in {"linear_regression", "penalized_splines"}:
            return self.estimator_.predict(np.column_stack([ba, ca]))
        raise KeyError(f"Unknown fusion strategy: {self.strategy}")


def _stratified_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    for label in np.unique(labels):
        group_count = len(np.unique(groups[labels == label]))
        if group_count < n_splits:
            raise ValueError(
                f"Class {label} has {group_count} participant groups; {n_splits} folds requested"
            )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(labels), dtype=float)
    return list(splitter.split(dummy, labels, groups))


def _score(predicted_age: np.ndarray, chronological_age: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gap":
        return np.asarray(predicted_age) - np.asarray(chronological_age)
    if mode == "age":
        return np.asarray(predicted_age)
    raise ValueError("fusion.diagnostic_score must be 'age' or 'gap'")


def _youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return float(np.median(scores))
    index_candidates = np.flatnonzero(finite)
    local = np.argmax((true_positive - false_positive)[finite])
    return float(thresholds[index_candidates[local]])


def _inner_predictions(
    frame: pd.DataFrame,
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    settings: dict[str, Any],
    age_column: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    local_labels = labels[indices]
    local_groups = groups[indices]
    splits = _stratified_splits(
        local_labels,
        local_groups,
        int(settings["inner_folds"]),
        seed,
    )
    predictions = {
        strategy: np.full(len(indices), np.nan, dtype=float)
        for strategy in settings["strategies"]
    }
    for train_local, validation_local in splits:
        train_index = indices[train_local]
        validation_index = indices[validation_local]
        for strategy in settings["strategies"]:
            model = FusionModel(strategy, settings).fit(
                frame.iloc[train_index][settings["brain_age_column"]].to_numpy(dtype=float),
                frame.iloc[train_index][settings["cognitive_age_column"]].to_numpy(dtype=float),
                frame.iloc[train_index][age_column].to_numpy(dtype=float),
                frame.iloc[train_index][settings["moca_column"]].to_numpy(dtype=float),
            )
            predictions[strategy][validation_local] = model.predict(
                frame.iloc[validation_index][settings["brain_age_column"]].to_numpy(dtype=float),
                frame.iloc[validation_index][settings["cognitive_age_column"]].to_numpy(dtype=float),
            )
    for strategy, values in predictions.items():
        if np.isnan(values).any():
            raise RuntimeError(f"Incomplete inner-fold predictions for {strategy}")
    return predictions, splits


def _bootstrap_weights(
    brain_age: np.ndarray,
    cognitive_age: np.ndarray,
    moca: np.ndarray,
    labels: np.ndarray,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    records = []
    for iteration in range(1, iterations + 1):
        sampled = np.concatenate(
            [rng.choice(index, size=len(index), replace=True) for index in class_indices]
        )
        weights = _positive_moca_weights(
            brain_age[sampled], cognitive_age[sampled], moca[sampled]
        )
        records.append(
            {
                "iteration": iteration,
                "brain_age_weight": float(weights[0]),
                "cognitive_age_weight": float(weights[1]),
            }
        )
    return pd.DataFrame(records)


def run(
    input_path: str,
    config_path: str,
    output_dir: str,
    brain_model_path: str | None = None,
    cognitive_model_path: str | None = None,
) -> dict:
    config = load_config(config_path)
    project = config["project"]
    settings = config["fusion"]
    seed = int(project["random_seed"])
    set_global_seed(seed)
    output = ensure_directory(output_dir)
    frame = read_table(input_path, project["delimiter"])

    age_column = str(project["age_column"])
    id_column = str(project["id_column"])
    required = [
        id_column,
        age_column,
        settings["moca_column"],
        settings["diagnosis_column"],
    ]
    require_columns(frame, required, "NCA fusion input")
    if settings["brain_age_column"] not in frame.columns:
        if not brain_model_path:
            raise ValueError("Provide --brain-model or a brain_age column")
        frame[settings["brain_age_column"]] = predict_brain_age(
            frame, load_artifact(brain_model_path), config
        )
    if settings["cognitive_age_column"] not in frame.columns:
        if not cognitive_model_path:
            raise ValueError("Provide --cognitive-model or a cognitive_age column")
        frame[settings["cognitive_age_column"]] = predict_cognitive_age(
            frame, load_artifact(cognitive_model_path), config
        )

    required.extend([settings["brain_age_column"], settings["cognitive_age_column"]])
    require_columns(frame, required, "NCA fusion input")
    if frame[required].isna().any().any():
        missing_counts = frame[required].isna().sum()
        raise ValueError(f"NCA fusion input contains missing values: {missing_counts[missing_counts > 0].to_dict()}")

    labels = diagnostic_labels(frame[settings["diagnosis_column"]], config)
    groups = participant_groups(frame, config)
    outer_splits = _stratified_splits(
        labels,
        groups,
        int(settings["outer_folds"]),
        seed + 901,
    )
    age = frame[age_column].to_numpy(dtype=float)
    outer_predictions = {
        strategy: np.full(len(frame), np.nan, dtype=float)
        for strategy in settings["strategies"]
    }
    outer_fold = np.full(len(frame), -1, dtype=int)
    outer_threshold = np.full(len(frame), np.nan, dtype=float)
    outer_classification = np.full(len(frame), -1, dtype=int)
    selected_by_inner = np.full(len(frame), "", dtype=object)
    fold_records: list[dict[str, Any]] = []
    retained_strategy = str(settings["retained_strategy"])
    diagnostic_mode = str(settings["diagnostic_score"])

    for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        inner_predictions, _ = _inner_predictions(
            frame,
            train_index,
            labels,
            groups,
            settings,
            age_column,
            seed + 1000 + fold,
        )
        inner_aucs = {
            strategy: safe_auc(
                labels[train_index],
                _score(values, age[train_index], diagnostic_mode),
            )
            for strategy, values in inner_predictions.items()
        }
        selected_strategy = max(
            inner_aucs,
            key=lambda key: -np.inf if np.isnan(inner_aucs[key]) else inner_aucs[key],
        )
        retained_inner_scores = _score(
            inner_predictions[retained_strategy], age[train_index], diagnostic_mode
        )
        threshold = _youden_threshold(labels[train_index], retained_inner_scores)

        fitted_models: dict[str, FusionModel] = {}
        for strategy in settings["strategies"]:
            model = FusionModel(strategy, settings).fit(
                frame.iloc[train_index][settings["brain_age_column"]].to_numpy(dtype=float),
                frame.iloc[train_index][settings["cognitive_age_column"]].to_numpy(dtype=float),
                age[train_index],
                frame.iloc[train_index][settings["moca_column"]].to_numpy(dtype=float),
            )
            fitted_models[strategy] = model
            outer_predictions[strategy][test_index] = model.predict(
                frame.iloc[test_index][settings["brain_age_column"]].to_numpy(dtype=float),
                frame.iloc[test_index][settings["cognitive_age_column"]].to_numpy(dtype=float),
            )

        test_scores = _score(
            outer_predictions[retained_strategy][test_index], age[test_index], diagnostic_mode
        )
        outer_fold[test_index] = fold
        outer_threshold[test_index] = threshold
        outer_classification[test_index] = (test_scores >= threshold).astype(int)
        selected_by_inner[test_index] = selected_strategy
        weights = fitted_models["weighted_mean"].weights_
        fold_records.append(
            {
                "outer_fold": fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "selected_strategy": selected_strategy,
                "retained_strategy": retained_strategy,
                "brain_age_weight": float(weights[0]),
                "cognitive_age_weight": float(weights[1]),
                "youden_threshold": threshold,
                **{f"inner_auc_{name}": value for name, value in inner_aucs.items()},
            }
        )

    if any(np.isnan(values).any() for values in outer_predictions.values()):
        raise RuntimeError("One or more fusion strategies have incomplete outer-fold predictions")

    result_columns = [id_column, age_column, settings["diagnosis_column"]]
    for optional in ["sex", "education", "language", settings["moca_column"]]:
        if optional in frame.columns and optional not in result_columns:
            result_columns.append(optional)
    results = frame.loc[:, result_columns].copy()
    results["binary_label"] = labels
    results["outer_fold"] = outer_fold
    results["brain_age"] = frame[settings["brain_age_column"]].to_numpy(dtype=float)
    results["cognitive_age"] = frame[settings["cognitive_age_column"]].to_numpy(dtype=float)
    results["brain_gap"] = results["brain_age"] - age
    results["cognitive_gap"] = results["cognitive_age"] - age
    for strategy, values in outer_predictions.items():
        results[f"nca_{strategy}"] = values
        results[f"nca_gap_{strategy}"] = values - age
    results["nca_index"] = outer_predictions[retained_strategy]
    results["nca_gap"] = results["nca_index"] - age
    results["research_threshold"] = outer_threshold
    results["predicted_label"] = outer_classification
    results["inner_selected_strategy"] = selected_by_inner
    write_table(results, output / "nca_outer_fold_predictions.csv", project["delimiter"])

    fold_summary = pd.DataFrame(fold_records)
    write_table(fold_summary, output / "nca_fold_summary.csv", project["delimiter"])
    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "auc": safe_auc(labels, _score(values, age, diagnostic_mode)),
                "mean_absolute_age_error": float(np.mean(np.abs(values - age))),
            }
            for strategy, values in outer_predictions.items()
        ]
    ).sort_values("auc", ascending=False)
    write_table(strategy_summary, output / "nca_strategy_summary.csv", project["delimiter"])

    weight_bootstrap = _bootstrap_weights(
        results["brain_age"].to_numpy(dtype=float),
        results["cognitive_age"].to_numpy(dtype=float),
        frame[settings["moca_column"]].to_numpy(dtype=float),
        labels,
        int(settings["weight_bootstrap_iterations"]),
        seed + 902,
    )
    write_table(weight_bootstrap, output / "nca_weight_bootstrap.csv", project["delimiter"])
    pooled_weights = _positive_moca_weights(
        results["brain_age"].to_numpy(dtype=float),
        results["cognitive_age"].to_numpy(dtype=float),
        frame[settings["moca_column"]].to_numpy(dtype=float),
    )
    report = {
        "component": "nca_fusion",
        "input_rows": int(len(frame)),
        "unique_participants": int(len(np.unique(groups))),
        "outer_folds": int(settings["outer_folds"]),
        "inner_folds": int(settings["inner_folds"]),
        "retained_strategy": retained_strategy,
        "diagnostic_score": diagnostic_mode,
        "pooled_descriptive_weights": {
            "brain_age": float(pooled_weights[0]),
            "cognitive_age": float(pooled_weights[1]),
        },
        "bootstrap_95_ci": {
            "brain_age": [
                float(weight_bootstrap["brain_age_weight"].quantile(0.025)),
                float(weight_bootstrap["brain_age_weight"].quantile(0.975)),
            ],
            "cognitive_age": [
                float(weight_bootstrap["cognitive_age_weight"].quantile(0.025)),
                float(weight_bootstrap["cognitive_age_weight"].quantile(0.975)),
            ],
        },
        "strategy_auc": dict(zip(strategy_summary["strategy"], strategy_summary["auc"])),
        "random_seed": seed,
    }
    dump_json(report, output / "nca_fusion_report.json")
    return report


def main() -> None:
    args = parse_args()
    report = run(
        args.input,
        args.config,
        args.output_dir,
        args.brain_model,
        args.cognitive_model,
    )
    print(
        "NCA fusion completed: "
        f"{report['outer_folds']} outer folds, retained strategy {report['retained_strategy']}"
    )


if __name__ == "__main__":
    main()

