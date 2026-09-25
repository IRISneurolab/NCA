"""Shared utilities for the NCA development and evaluation workflow."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import boxcox as boxcox_fixed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_table(path: str | Path, delimiter: str = ";") -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=delimiter)


def write_table(frame: pd.DataFrame, path: str | Path, delimiter: str = ";") -> None:
    frame.to_csv(path, sep=delimiter, index=False)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{context}: missing required columns: {missing}")


def _token(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value)).casefold()
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value)).casefold()
    return str(value).strip().casefold()


def encode_binary(
    values: pd.Series,
    positive_values: Sequence[Any],
    negative_values: Sequence[Any],
    name: str,
) -> pd.Series:
    positives = {_token(value) for value in positive_values}
    negatives = {_token(value) for value in negative_values}
    encoded = []
    unknown = set()
    for value in values:
        token = _token(value)
        if token in positives:
            encoded.append(1.0)
        elif token in negatives:
            encoded.append(0.0)
        else:
            unknown.add(str(value))
            encoded.append(np.nan)
    if unknown:
        raise ValueError(
            f"{name}: unrecognized categories {sorted(unknown)}. "
            "Update the corresponding values in config.yaml."
        )
    return pd.Series(encoded, index=values.index, name=name, dtype=float)


def numeric_frame(frame: pd.DataFrame, columns: Sequence[str], context: str) -> pd.DataFrame:
    converted = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    bad = converted.columns[converted.isna().any()].tolist()
    if bad:
        raise ValueError(f"{context}: missing or non-numeric values in columns: {bad}")
    return converted.astype(float)


def participant_groups(frame: pd.DataFrame, config: Mapping[str, Any]) -> np.ndarray:
    id_column = str(config["project"]["id_column"])
    group_column = "participant_id" if "participant_id" in frame.columns else id_column
    require_columns(frame, [group_column], "participant grouping")
    groups = frame[group_column].astype(str).to_numpy()
    if pd.isna(groups).any():
        raise ValueError(f"Participant grouping column {group_column!r} contains missing values")
    return groups


def group_kfold(n_splits: int, groups: np.ndarray, seed: int) -> GroupKFold:
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Requested {n_splits} folds but only {len(unique_groups)} participant groups are available"
        )
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def heldout_group_split(
    groups: np.ndarray, development_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=development_fraction,
        random_state=seed,
    )
    dummy = np.zeros(len(groups), dtype=float)
    train_index, test_index = next(splitter.split(dummy, groups=groups))
    return train_index, test_index


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(observed) < 2:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "pearson_r": correlation,
        "r2": float(r2_score(observed, predicted)),
    }


def paired_mae_improvement(
    chronological_age: np.ndarray,
    retained_predictions: np.ndarray,
    alternative_predictions: np.ndarray,
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float | list[float]]:
    """Bootstrap the paired MAE improvement of retained over alternative predictions."""
    age = np.asarray(chronological_age, dtype=float)
    retained_error = np.abs(np.asarray(retained_predictions, dtype=float) - age)
    alternative_error = np.abs(np.asarray(alternative_predictions, dtype=float) - age)
    observed = float(np.mean(alternative_error - retained_error))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(age), size=len(age))
        estimates[iteration] = float(
            np.mean(alternative_error[sampled] - retained_error[sampled])
        )
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower_tail = (np.sum(estimates <= 0.0) + 1.0) / (iterations + 1.0)
    upper_tail = (np.sum(estimates >= 0.0) + 1.0) / (iterations + 1.0)
    p_value = float(min(1.0, 2.0 * min(lower_tail, upper_tail)))
    return {
        "mae_improvement_years": observed,
        "confidence_interval": [float(lower), float(upper)],
        "bootstrap_two_sided_p": p_value,
        "bootstrap_iterations": int(iterations),
    }


def fit_age_bias(predicted_age: np.ndarray, chronological_age: np.ndarray) -> dict[str, float]:
    predicted_age = np.asarray(predicted_age, dtype=float)
    chronological_age = np.asarray(chronological_age, dtype=float)
    error = predicted_age - chronological_age
    model = LinearRegression().fit(chronological_age.reshape(-1, 1), error)
    return {"alpha": float(model.coef_[0]), "beta": float(model.intercept_)}


def apply_age_bias_correction(
    predicted_age: np.ndarray,
    chronological_age: np.ndarray,
    correction: Mapping[str, float],
) -> np.ndarray:
    return np.asarray(predicted_age, dtype=float) - (
        float(correction["alpha"]) * np.asarray(chronological_age, dtype=float)
        + float(correction["beta"])
    )


def make_regression_pipeline(model_name: str, seed: int) -> Pipeline:
    if model_name == "ridge":
        estimator = Ridge()
    elif model_name == "svr_linear":
        estimator = SVR(kernel="linear", tol=0.001)
    elif model_name == "svr_rbf":
        estimator = SVR(kernel="rbf", tol=0.001)
    elif model_name == "lasso":
        estimator = Lasso(max_iter=50_000, random_state=seed)
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=1)
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=seed)
    else:
        raise KeyError(f"Unknown model family: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def dump_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(serializable(data), stream, indent=2, ensure_ascii=False, allow_nan=False)


def nested_model_benchmark(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    grids: Mapping[str, Mapping[str, Sequence[Any]]],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run participant-grouped nested CV for each candidate age model."""
    y = np.asarray(y, dtype=float)
    outer = group_kfold(outer_folds, groups, seed)
    prediction_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    for model_index, model_name in enumerate(model_names):
        for fold, (train_index, test_index) in enumerate(
            outer.split(X, y, groups), start=1
        ):
            X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
            X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            groups_train = groups[train_index]
            inner = group_kfold(inner_folds, groups_train, seed + model_index * 100 + fold)
            inner_splits = list(inner.split(X_train, y_train, groups_train))
            search = GridSearchCV(
                make_regression_pipeline(model_name, seed + fold),
                param_grid=dict(grids[model_name]),
                scoring="neg_mean_absolute_error",
                cv=inner_splits,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            predicted = search.predict(X_test)
            metrics = regression_metrics(y_test, predicted)
            fold_records.append(
                {
                    "model": model_name,
                    "outer_fold": fold,
                    **metrics,
                    "best_parameters": json.dumps(serializable(search.best_params_), sort_keys=True),
                }
            )
            for row_index, observed, estimate in zip(test_index, y_test, predicted):
                prediction_records.append(
                    {
                        "row_index": int(row_index),
                        "model": model_name,
                        "outer_fold": fold,
                        "chronological_age": float(observed),
                        "predicted_age": float(estimate),
                    }
                )

    folds = pd.DataFrame(fold_records)
    summary = (
        folds.groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_sd=("pearson_r", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
        )
        .sort_values("mae_mean")
        .reset_index(drop=True)
    )
    return pd.DataFrame(prediction_records), folds, summary


def grouped_oof_predictions(
    estimator: Pipeline,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = group_kfold(n_splits, groups, seed)
    predictions = np.full(len(y), np.nan, dtype=float)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        fitted = clone(estimator).fit(
            X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index],
            y[train_index],
        )
        X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
        predictions[test_index] = fitted.predict(X_test)
        folds[test_index] = fold
    if np.isnan(predictions).any() or (folds < 1).any():
        raise RuntimeError("Failed to generate complete out-of-fold predictions")
    return predictions, folds


def prepare_brain_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    settings = config["brain_age"]
    if feature_columns is None:
        feature_file = settings.get("feature_columns_file")
        if feature_file:
            config_path = Path(str(config["_config_path"]))
            path = Path(feature_file)
            if not path.is_absolute():
                path = config_path.parent / path
            feature_columns = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            reserved = set(settings.get("reserved_columns", []))
            feature_columns = [column for column in frame.columns if column not in reserved]
    feature_columns = list(feature_columns)
    require_columns(frame, feature_columns, "brain-age features")
    prepared = frame.loc[:, feature_columns].copy()
    sex_column = str(settings["sex_column"])
    if sex_column in prepared.columns and not pd.api.types.is_numeric_dtype(prepared[sex_column]):
        prepared[sex_column] = encode_binary(
            prepared[sex_column],
            settings["sex_positive_values"],
            settings["sex_negative_values"],
            sex_column,
        )
    prepared = numeric_frame(prepared, feature_columns, "brain-age features")
    return prepared, feature_columns


def prepare_cognitive_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    boxcox_lambda: float | None = None,
) -> pd.DataFrame:
    settings = config["cognitive_age"]
    columns = [
        settings["fluency_column"],
        settings["education_column"],
        settings["sex_column"],
        settings["language_column"],
    ]
    require_columns(frame, columns, "cognitive-age features")
    fluency = pd.to_numeric(frame[settings["fluency_column"]], errors="coerce")
    education = pd.to_numeric(frame[settings["education_column"]], errors="coerce")
    if fluency.isna().any() or education.isna().any():
        raise ValueError("Cognitive features contain missing or non-numeric fluency/education values")
    shift = float(settings["boxcox_shift"])
    shifted = fluency.to_numpy(dtype=float) + shift
    if np.any(shifted <= 0):
        raise ValueError("Box-Cox requires fluency + boxcox_shift to be strictly positive")
    lam = float(settings["boxcox_lambda"] if boxcox_lambda is None else boxcox_lambda)
    sex = encode_binary(
        frame[settings["sex_column"]],
        settings["sex_positive_values"],
        settings["sex_negative_values"],
        "sex_M",
    )
    language = encode_binary(
        frame[settings["language_column"]],
        settings["french_values"],
        settings["non_french_values"],
        "language_french",
    )
    return pd.DataFrame(
        {
            "fluency_bc": boxcox_fixed(shifted, lam),
            "education": education.to_numpy(dtype=float),
            "sex_M": sex.to_numpy(dtype=float),
            "language_french": language.to_numpy(dtype=float),
        },
        index=frame.index,
    )


def predict_brain_age(
    frame: pd.DataFrame,
    artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    X, _ = prepare_brain_features(frame, config, artifact["features"])
    raw = artifact["model"].predict(artifact["scaler"].transform(X))
    age_column = config["project"]["age_column"]
    require_columns(frame, [age_column], "brain-age bias correction")
    correction = artifact.get("beheshti_correction", artifact.get("bias_correction"))
    return apply_age_bias_correction(raw, frame[age_column].to_numpy(dtype=float), correction)


def predict_cognitive_age(
    frame: pd.DataFrame,
    artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    X = prepare_cognitive_features(frame, config, artifact["lambda_bc"])
    X = X.loc[:, artifact["features"]]
    raw = artifact["model"].predict(artifact["scaler"].transform(X))
    age_column = config["project"]["age_column"]
    require_columns(frame, [age_column], "cognitive-age bias correction")
    correction = artifact.get("bias_correction", artifact.get("beheshti_correction"))
    return apply_age_bias_correction(raw, frame[age_column].to_numpy(dtype=float), correction)


def load_artifact(path: str | Path) -> Mapping[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, Mapping):
        raise TypeError(f"Expected a mapping in model artifact: {path}")
    return artifact


def diagnostic_labels(values: pd.Series, config: Mapping[str, Any]) -> np.ndarray:
    settings = config["fusion"]
    controls = {_token(value) for value in settings["control_labels"]}
    impaired = {_token(value) for value in settings["impaired_labels"]}
    labels = []
    unknown = set()
    for value in values:
        token = _token(value)
        if token in controls:
            labels.append(0)
        elif token in impaired:
            labels.append(1)
        else:
            unknown.add(str(value))
            labels.append(-1)
    if unknown:
        raise ValueError(
            f"Unrecognized diagnosis labels {sorted(unknown)}. Update config.yaml."
        )
    return np.asarray(labels, dtype=int)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))
