#!/usr/bin/env python3
"""Run the complete NCA development, fusion, and OOF evaluation workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-data", required=True, help="NOMIS-compatible Brain Age table")
    parser.add_argument(
        "--brain-raw-tiv-data",
        help="Optional raw TIV-adjusted morphometry table for the normalization ablation",
    )
    parser.add_argument("--cognitive-data", required=True, help="COGNIS Cognitive Age table")
    parser.add_argument("--fusion-data", required=True, help="NCA evaluation table")
    parser.add_argument("--config", default="config.yaml", help="Workflow YAML configuration")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    return parser.parse_args()


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output_dir).resolve()
    brain_output = output / "brain_age"
    cognitive_output = output / "cognitive_age"
    fusion_output = output / "fusion"
    evaluation_output = output / "evaluation"
    for directory in [brain_output, cognitive_output, fusion_output, evaluation_output]:
        directory.mkdir(parents=True, exist_ok=True)

    config = str(Path(args.config).resolve())
    commands = [
        [
            sys.executable,
            str(root / "train_brain_age.py"),
            "--input",
            str(Path(args.brain_data).resolve()),
            "--config",
            config,
            "--output-dir",
            str(brain_output),
        ],
        [
            sys.executable,
            str(root / "train_cognitive_age.py"),
            "--input",
            str(Path(args.cognitive_data).resolve()),
            "--config",
            config,
            "--output-dir",
            str(cognitive_output),
        ],
        [
            sys.executable,
            str(root / "fit_nca_fusion.py"),
            "--input",
            str(Path(args.fusion_data).resolve()),
            "--config",
            config,
            "--output-dir",
            str(fusion_output),
            "--brain-model",
            str(brain_output / "nca_brain_pipeline.joblib"),
            "--cognitive-model",
            str(cognitive_output / "nca_cognitive_pipeline.joblib"),
        ],
        [
            sys.executable,
            str(root / "evaluate_oof.py"),
            "--input",
            str(fusion_output / "nca_outer_fold_predictions.csv"),
            "--config",
            config,
            "--output-dir",
            str(evaluation_output),
        ],
    ]
    if args.brain_raw_tiv_data:
        commands[0].extend(
            ["--raw-tiv-input", str(Path(args.brain_raw_tiv_data).resolve())]
        )
    for command in commands:
        _run(command, root)

    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "config": config,
        "inputs": {
            "brain_data": str(Path(args.brain_data).resolve()),
            "brain_raw_tiv_data": (
                str(Path(args.brain_raw_tiv_data).resolve())
                if args.brain_raw_tiv_data
                else None
            ),
            "cognitive_data": str(Path(args.cognitive_data).resolve()),
            "fusion_data": str(Path(args.fusion_data).resolve()),
        },
        "outputs": {
            "brain_age": str(brain_output),
            "cognitive_age": str(cognitive_output),
            "fusion": str(fusion_output),
            "evaluation": str(evaluation_output),
        },
        "commands": commands,
    }
    with (output / "workflow_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print(f"NCA workflow completed. Outputs: {output}")


if __name__ == "__main__":
    main()
