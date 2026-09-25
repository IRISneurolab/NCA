#!/usr/bin/env python3
"""Evaluate participant-disjoint NCA out-of-fold predictions."""

from __future__ import annotations

import argparse
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from nca_utils import (
    dump_json,
    ensure_directory,
    load_config,
    read_table,
    require_columns,
    set_global_seed,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="nca_outer_fold_predictions.csv")
    parser.add_argument("--config", default="config.yaml", help="Workflow YAML configuration")
    parser.add_argument("--output-dir", required=True, help="Directory for statistical outputs")
    return parser.parse_args()


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    count = len(values)
    midranks = np.zeros(count, dtype=float)
    start = 0
    while start < count:
        end = start
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    result = np.empty(count, dtype=float)
    result[order] = midranks
    return result


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    classifiers, examples = predictions.shape
    negative_count = examples - positive_count
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    tx = np.empty((classifiers, positive_count), dtype=float)
    ty = np.empty((classifiers, negative_count), dtype=float)
    tz = np.empty((classifiers, examples), dtype=float)
    for index in range(classifiers):
        tx[index] = _compute_midrank(positive[index])
        ty[index] = _compute_midrank(negative[index])
        tz[index] = _compute_midrank(predictions[index])
    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / positive_count + sy / negative_count
    return aucs, np.atleast_2d(covariance)


def delong_test(
    labels: np.ndarray, first_scores: np.ndarray, second_scores: np.ndarray
) -> tuple[float, float, float, float]:
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("DeLong inference requires both binary classes")
    order = np.argsort(-labels)
    predictions = np.vstack([first_scores, second_scores])[:, order]
    aucs, covariance = _fast_delong(predictions, int(labels.sum()))
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0 or not np.isfinite(variance):
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    z_value = float((aucs[0] - aucs[1]) / math.sqrt(variance))
    p_value = float(2.0 * norm.sf(abs(z_value)))
    return float(aucs[0]), float(aucs[1]), z_value, p_value


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted.tolist()
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _stratified_sample_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in (np.flatnonzero(labels == value) for value in np.unique(labels))
        ]
    )


def bootstrap_interval(
    labels: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = _stratified_sample_indices(labels, rng)
        try:
            estimate = float(statistic(sampled))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence_level
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def _age_score(frame: pd.DataFrame, column: str, mode: str, age_column: str) -> np.ndarray:
    values = frame[column].to_numpy(dtype=float)
    return values - frame[age_column].to_numpy(dtype=float) if mode == "gap" else values


def _operating_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    true_positive = np.sum((labels == 1) & (predicted == 1))
    false_negative = np.sum((labels == 1) & (predicted == 0))
    true_negative = np.sum((labels == 0) & (predicted == 0))
    false_positive = np.sum((labels == 0) & (predicted == 1))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
    }


def _calibration(predicted_age: np.ndarray, chronological_age: np.ndarray) -> tuple[float, float]:
    model = LinearRegression().fit(
        np.asarray(chronological_age).reshape(-1, 1), np.asarray(predicted_age)
    )
    return float(model.intercept_), float(model.coef_[0])


def _subgroup_masks(frame: pd.DataFrame, config: dict) -> list[tuple[str, str, np.ndarray]]:
    masks: list[tuple[str, str, np.ndarray]] = []
    settings = config["evaluation"]
    for column in settings.get("subgroup_columns", []):
        if column in frame.columns:
            for value in sorted(frame[column].dropna().unique(), key=str):
                masks.append((column, str(value), frame[column].to_numpy() == value))
    age_column = config["project"]["age_column"]
    age = frame[age_column].to_numpy(dtype=float)
    for group in settings.get("age_groups", []):
        lower = float(group["lower"])
        upper = group.get("upper")
        mask = age >= lower
        if upper is not None:
            mask &= age < float(upper)
        masks.append(("age", str(group["label"]), mask))
    if "education" in frame.columns:
        education = frame["education"].to_numpy(dtype=float)
        for group in settings.get("education_groups", []):
            lower = float(group["lower"])
            upper = group.get("upper")
            mask = education >= lower
            if upper is not None:
                mask &= education < float(upper)
            masks.append(("education", str(group["label"]), mask))
    return masks


def run(input_path: str, config_path: str, output_dir: str) -> dict:
    config = load_config(config_path)
    project = config["project"]
    fusion = config["fusion"]
    settings = config["evaluation"]
    seed = int(project["random_seed"])
    set_global_seed(seed)
    output = ensure_directory(output_dir)
    frame = read_table(input_path, project["delimiter"])
    age_column = str(project["age_column"])
    id_column = str(project["id_column"])
    required = [
        id_column,
        age_column,
        "binary_label",
        "predicted_label",
        "brain_age",
        "cognitive_age",
        "nca_index",
    ]
    require_columns(frame, required, "OOF evaluation input")
    if frame[id_column].duplicated().any():
        raise ValueError(
            "OOF evaluation requires one row per participant so that each bootstrap draw is "
            "participant-level. Aggregate repeated visits or provide one prespecified visit per participant."
        )
    labels = frame["binary_label"].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("OOF evaluation requires binary labels coded 0 and 1")
    iterations = int(settings["bootstrap_iterations"])
    confidence = float(settings["confidence_level"])
    diagnostic_mode = str(fusion["diagnostic_score"])

    age_models = {
        "BA": "brain_age",
        "CA": "cognitive_age",
        "NCA": "nca_index",
    }
    auc_records = []
    for offset, (name, column) in enumerate(age_models.items()):
        scores = _age_score(frame, column, diagnostic_mode, age_column)
        estimate = float(roc_auc_score(labels, scores))
        interval = bootstrap_interval(
            labels,
            lambda index, s=scores: roc_auc_score(labels[index], s[index]),
            iterations,
            confidence,
            seed + 1100 + offset,
        )
        auc_records.append(
            {
                "analysis": name,
                "auc": estimate,
                "ci_lower": interval[0],
                "ci_upper": interval[1],
                "score_scale": diagnostic_mode,
            }
        )
    auc_table = pd.DataFrame(auc_records)
    write_table(auc_table, output / "auc_summary.csv", project["delimiter"])

    predicted_labels = frame["predicted_label"].to_numpy(dtype=int)
    operating = _operating_metrics(labels, predicted_labels)
    operating_records = []
    for offset, metric in enumerate(["sensitivity", "specificity", "balanced_accuracy"]):
        interval = bootstrap_interval(
            labels,
            lambda index, key=metric: _operating_metrics(
                labels[index], predicted_labels[index]
            )[key],
            iterations,
            confidence,
            seed + 1200 + offset,
        )
        operating_records.append(
            {
                "metric": metric,
                "estimate": operating[metric],
                "ci_lower": interval[0],
                "ci_upper": interval[1],
            }
        )
    operating_table = pd.DataFrame(operating_records)
    write_table(operating_table, output / "operating_point.csv", project["delimiter"])

    calibration_records = []
    age = frame[age_column].to_numpy(dtype=float)
    for model_offset, (name, column) in enumerate(age_models.items()):
        predicted = frame[column].to_numpy(dtype=float)
        intercept, slope = _calibration(predicted, age)
        intercept_ci = bootstrap_interval(
            labels,
            lambda index, p=predicted: _calibration(p[index], age[index])[0],
            iterations,
            confidence,
            seed + 1300 + model_offset * 2,
        )
        slope_ci = bootstrap_interval(
            labels,
            lambda index, p=predicted: _calibration(p[index], age[index])[1],
            iterations,
            confidence,
            seed + 1301 + model_offset * 2,
        )
        calibration_records.append(
            {
                "analysis": name,
                "intercept": intercept,
                "intercept_ci_lower": intercept_ci[0],
                "intercept_ci_upper": intercept_ci[1],
                "slope": slope,
                "slope_ci_lower": slope_ci[0],
                "slope_ci_upper": slope_ci[1],
            }
        )
    calibration_table = pd.DataFrame(calibration_records)
    write_table(calibration_table, output / "continuous_age_calibration.csv", project["delimiter"])

    retained_column = f"nca_{fusion['retained_strategy']}"
    require_columns(frame, [retained_column], "fusion comparison")
    retained_scores = _age_score(frame, retained_column, diagnostic_mode, age_column)
    comparison_columns = [
        f"nca_{strategy}"
        for strategy in fusion["strategies"]
        if strategy != fusion["retained_strategy"]
    ]
    comparison_records = []
    for column in comparison_columns:
        require_columns(frame, [column], "fusion comparison")
        alternative_scores = _age_score(frame, column, diagnostic_mode, age_column)
        retained_auc, alternative_auc, z_value, p_value = delong_test(
            labels, retained_scores, alternative_scores
        )
        comparison_records.append(
            {
                "retained_strategy": fusion["retained_strategy"],
                "alternative_strategy": column.removeprefix("nca_"),
                "retained_auc": retained_auc,
                "alternative_auc": alternative_auc,
                "auc_difference": retained_auc - alternative_auc,
                "delong_z": z_value,
                "raw_p": p_value,
            }
        )
    adjusted = holm_adjust([record["raw_p"] for record in comparison_records])
    for record, adjusted_p in zip(comparison_records, adjusted):
        record["holm_adjusted_p"] = adjusted_p
    comparison_table = pd.DataFrame(comparison_records)
    write_table(comparison_table, output / "fusion_delong_comparisons.csv", project["delimiter"])

    subgroup_records = []
    for group_index, (variable, level, mask) in enumerate(_subgroup_masks(frame, config)):
        subgroup_labels = labels[mask]
        if mask.sum() < 20 or len(np.unique(subgroup_labels)) < 2:
            continue
        for model_index, (name, column) in enumerate(age_models.items()):
            subgroup_frame = frame.loc[mask]
            subgroup_scores = _age_score(
                subgroup_frame, column, diagnostic_mode, age_column
            )
            estimate = float(roc_auc_score(subgroup_labels, subgroup_scores))
            interval = bootstrap_interval(
                subgroup_labels,
                lambda index, s=subgroup_scores, lab=subgroup_labels: roc_auc_score(
                    lab[index], s[index]
                ),
                iterations,
                confidence,
                seed + 1400 + group_index * 10 + model_index,
            )
            subgroup_records.append(
                {
                    "subgroup_variable": variable,
                    "subgroup": level,
                    "n": int(mask.sum()),
                    "analysis": name,
                    "auc": estimate,
                    "ci_lower": interval[0],
                    "ci_upper": interval[1],
                }
            )
    subgroup_table = pd.DataFrame(subgroup_records)
    write_table(subgroup_table, output / "subgroup_auc.csv", project["delimiter"])

    report = {
        "n_rows": int(len(frame)),
        "n_controls": int(np.sum(labels == 0)),
        "n_impaired": int(np.sum(labels == 1)),
        "bootstrap_iterations": iterations,
        "confidence_level": confidence,
        "diagnostic_score": diagnostic_mode,
        "auc": {row["analysis"]: row["auc"] for row in auc_records},
        "operating_point": operating,
        "probability_calibration": {
            "assessed": False,
            "reason": (
                "The analysis uses continuous age scores and does not fit a mapping to "
                "diagnostic probabilities."
            ),
        },
        "multiple_testing": {
            "fusion_family": "Holm correction across the six retained-versus-alternative comparisons"
        },
    }
    dump_json(report, output / "evaluation_report.json")
    return report


def main() -> None:
    args = parse_args()
    report = run(args.input, args.config, args.output_dir)
    print(
        "OOF evaluation completed: "
        f"N={report['n_rows']}, NCA AUC={report['auc']['NCA']:.3f}"
    )


if __name__ == "__main__":
    main()
