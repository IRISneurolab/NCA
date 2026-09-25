#!/usr/bin/env python3
"""Functional smoke test using synthetic, non-identifying data."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_oof import run as evaluate_oof  # noqa: E402
from fit_nca_fusion import run as fit_nca_fusion  # noqa: E402
from train_brain_age import run as train_brain_age  # noqa: E402
from train_cognitive_age import run as train_cognitive_age  # noqa: E402


def _test_config(destination: Path) -> Path:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["project"]["random_seed"] = 123
    config["project"]["n_jobs"] = 1
    brain = config["brain_age"]
    brain["candidate_models"] = ["ridge"]
    brain["grids"]["ridge"] = {"model__alpha": [0.1, 1.0]}
    brain["outer_folds"] = 3
    brain["inner_folds"] = 2
    brain["final_bias_folds"] = 3
    brain["final_ridge_alphas"] = [0.1, 1.0]
    brain["final_ridge_cv"] = 3
    cognitive = config["cognitive_age"]
    cognitive["candidate_models"] = ["svr_linear"]
    cognitive["grids"]["svr_linear"] = {
        "model__C": [1.0, 10.0],
        "model__epsilon": [0.1],
    }
    cognitive["outer_folds"] = 3
    cognitive["inner_folds"] = 2
    cognitive["final_bias_folds"] = 3
    cognitive["final_svr"] = {"C": 10.0, "epsilon": 0.1, "tolerance": 0.001}
    config["fusion"]["outer_folds"] = 3
    config["fusion"]["inner_folds"] = 2
    config["fusion"]["weight_bootstrap_iterations"] = 20
    config["evaluation"]["bootstrap_iterations"] = 20
    path = destination / "test_config.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return path


def _synthetic_tables(destination: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(123)
    n = 72
    age = rng.uniform(40, 90, n)
    sex = np.where(rng.random(n) > 0.5, "M", "F")
    brain = pd.DataFrame({"id": [f"B{i:03d}" for i in range(n)], "chron_age": age, "sex": sex})
    for feature in range(8):
        brain[f"mri_{feature:03d}"] = (
            0.05 * (feature + 1) * age + rng.normal(0, 1.5, n)
        )

    fluency = np.clip(42 - 0.28 * age + rng.normal(0, 4, n), 1, None)
    cognitive = pd.DataFrame(
        {
            "id": [f"C{i:03d}" for i in range(n)],
            "chron_age": age,
            "fluency": fluency,
            "education": rng.integers(8, 21, n),
            "sex": sex,
            "language": np.where(rng.random(n) > 0.35, "french", "english"),
        }
    )

    diagnosis = np.where(np.arange(n) % 2 == 0, "CON", "MCI")
    impairment = (diagnosis == "MCI").astype(float)
    fusion = pd.DataFrame(
        {
            "id": [f"N{i:03d}" for i in range(n)],
            "chron_age": age,
            "moca": 29 - 4.0 * impairment - 0.03 * (age - 60) + rng.normal(0, 1, n),
            "diagnosis": diagnosis,
            "sex": sex,
            "education": rng.integers(8, 21, n),
            "language": np.where(rng.random(n) > 0.35, "french", "english"),
            "fluency": np.clip(
                42 - 0.28 * age - 4.0 * impairment + rng.normal(0, 4, n), 1, None
            ),
        }
    )
    for feature in range(8):
        fusion[f"mri_{feature:03d}"] = (
            0.05 * (feature + 1) * (age + 4.5 * impairment) + rng.normal(0, 1.5, n)
        )
    paths = (
        destination / "brain.csv",
        destination / "cognitive.csv",
        destination / "fusion.csv",
    )
    for frame, path in zip([brain, cognitive, fusion], paths):
        frame.to_csv(path, sep=";", index=False)
    return paths


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nca_workflow_test_") as temporary:
        work = Path(temporary)
        config = _test_config(work)
        brain_data, cognitive_data, fusion_data = _synthetic_tables(work)
        brain_output = work / "brain_output"
        cognitive_output = work / "cognitive_output"
        fusion_output = work / "fusion_output"
        evaluation_output = work / "evaluation_output"

        train_brain_age(
            str(brain_data), str(config), str(brain_output), str(brain_data)
        )
        train_cognitive_age(str(cognitive_data), str(config), str(cognitive_output))
        fit_nca_fusion(
            str(fusion_data),
            str(config),
            str(fusion_output),
            str(brain_output / "nca_brain_pipeline.joblib"),
            str(cognitive_output / "nca_cognitive_pipeline.joblib"),
        )
        report = evaluate_oof(
            str(fusion_output / "nca_outer_fold_predictions.csv"),
            str(config),
            str(evaluation_output),
        )

        expected = [
            brain_output / "nca_brain_pipeline.joblib",
            cognitive_output / "nca_cognitive_pipeline.joblib",
            fusion_output / "nca_outer_fold_predictions.csv",
            evaluation_output / "fusion_delong_comparisons.csv",
            evaluation_output / "continuous_age_calibration.csv",
        ]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise AssertionError(f"Smoke test did not create expected files: {missing}")
        if not 0.0 <= float(report["auc"]["NCA"]) <= 1.0:
            raise AssertionError("NCA AUC is outside [0, 1]")
        print("NCA workflow smoke test passed")


if __name__ == "__main__":
    main()
