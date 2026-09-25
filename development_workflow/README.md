# NCA model-development and evaluation workflow

This directory contains the shareable training, nested cross-validation, multimodal-fusion, and statistical-evaluation workflow for the NeuroCognitive Age (NCA) framework. It operates on de-identified tabular data and contains no participant identifiers, institutional paths, credentials, or cohort-specific access logic.

## Scope

The workflow covers:

1. Brain Age (BA) model benchmarking, participant-grouped nested cross-validation, age-bias correction, and model export.
2. Cognitive Age (CA) transformation, model benchmarking, participant-grouped nested cross-validation, age-bias correction, and model export.
3. Participant-disjoint NCA fusion with ten-fold outer and inner loops.
4. Comparison of seven prespecified fusion architectures.
5. Generation of out-of-fold predictions, fold-specific research thresholds, confidence intervals, continuous age-calibration estimates, DeLong tests, Holm-adjusted p values, and subgroup AUC summaries.

Raw T1-weighted images and controlled cohort data are not distributed with this code. MRI inputs must first be processed with FreeSurfer 6.0 and converted to the NOMIS-compatible feature representation used by the study. FreeSurfer and NOMIS remain subject to their respective installation, access, and licensing terms.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Random seed, cross-validation design, model grids, retained parameters, fusion settings, and bootstrap settings |
| `nca_utils.py` | Shared input validation, encoding, cross-validation, bias-correction, prediction, and file utilities |
| `train_brain_age.py` | BA benchmarking, held-out evaluation, out-of-fold bias correction, and joblib export |
| `train_cognitive_age.py` | CA transformation, benchmarking, ablation output, out-of-fold bias correction, and joblib export |
| `fit_nca_fusion.py` | Nested fusion analysis, MoCA-anchored weights, seven-strategy comparison, and outer-fold predictions |
| `evaluate_oof.py` | Bootstrap confidence intervals, calibration, operating-point metrics, DeLong tests, Holm correction, and subgroup reporting |
| `run_workflow.py` | End-to-end command-line orchestration |
| `requirements-development.txt` | Versioned Python dependencies |
| `tests/smoke_test.py` | Synthetic-data functional test |

## Software environment

- Python 3.9 or later
- scikit-learn 1.6.1
- NumPy 1.24 or later
- pandas 2.0 or later
- SciPy 1.10 or later
- joblib 1.3 or later
- PyYAML 6.0 or later

Install the dependencies in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-development.txt
```

## Input tables

All example commands assume semicolon-delimited CSV files. The delimiter can be changed in `config.yaml`.

### Brain Age normative table

Required columns:

- `id`: participant identifier;
- `chron_age`: chronological age in years;
- `sex`: binary sex variable using a coding declared in `config.yaml`;
- the prepared NOMIS-compatible morphometric variables.

When `brain_age.feature_columns_file` is `null`, all columns not listed under `brain_age.reserved_columns` are used as BA predictors. For strict column control, supply a text file containing one predictor name per line and set `feature_columns_file` accordingly.

Repeated observations may be identified with an optional `participant_id` column. All observations belonging to the same participant are then kept in the same cross-validation fold.

### Cognitive Age normative table

Required columns:

- `id`;
- `chron_age`;
- `fluency`: semantic verbal-fluency score;
- `education`: years of education;
- `sex`;
- `language`.

Category mappings are explicit in `config.yaml`. Unrecognized values stop execution rather than being silently assigned to a reference category.

### NCA fusion and evaluation table

Required columns:

- `id`;
- `chron_age`;
- `moca`;
- `diagnosis`;
- variables required by the BA and CA models.

Alternatively, the table may directly contain `brain_age` and `cognitive_age`. Diagnostic labels and control/impaired mappings are declared in `config.yaml`. Optional `sex` and `education` columns enable subgroup summaries.

## Model-development settings

The default configuration fixes `random_seed: 2026` and uses participant-grouped splits throughout.

The BA and CA multiverse evaluates Ridge, linear SVR, radial-basis-function SVR, Lasso, random forest, and gradient boosting. Every hyperparameter grid is listed explicitly in `config.yaml`. The retained BA estimator is RidgeCV with candidate alphas 0.1, 1.0, and 10.0. The retained CA estimator is linear SVR with `C = 500`, `epsilon = 0.5`, and `tolerance = 0.001`. Semantic fluency is transformed using Box–Cox `lambda = 0.633` before standardization.

Age-bias correction is estimated from participant-disjoint out-of-fold predictions. For each component, the prediction error is regressed on chronological age:

```text
error = predicted_age - chronological_age
error = alpha * chronological_age + beta
corrected_age = predicted_age - (alpha * chronological_age + beta)
```

The fitted coefficients are stored in the exported joblib artifacts.

## Fusion analysis

The seven strategies are:

1. simple mean;
2. MoCA-anchored weighted mean;
3. Euclidean combination;
4. exponential combination;
5. logarithmic combination;
6. multiple linear regression;
7. penalized spline regression.

Their operational definitions are explicit in `fit_nca_fusion.py`. The simple mean averages BA and CA on their common age scale. The weighted mean applies the non-negative, normalized MoCA-anchored coefficients. The Euclidean model uses the root-mean-square magnitude of BA and CA. The exponential model applies an exponential transformation to the weighted standardized component ages and maps the resulting score back to the chronological-age scale using an outer-training calibration model. The logarithmic model applies the natural logarithm to the summed component ages and uses the same training-only age-scale calibration. Multiple linear regression predicts chronological age from BA and CA, while the penalized-spline model applies cubic spline bases followed by ridge penalization.

Within each outer fold, all standardization, MoCA-anchored weight estimation, architecture comparison, and research-threshold selection are restricted to the outer-training participants. The resulting parameters are applied once to the held-out participants. Held-out MoCA values do not contribute to weight estimation.

For the weighted mean, non-negative coefficients are estimated by regressing negative standardized MoCA on standardized BA and CA in the training data. The coefficients are normalized to sum to one and are then applied to the age-scale BA and CA estimates.

## Running individual stages

```bash
python train_brain_age.py \
  --input /path/to/brain_normative.csv \
  --config config.yaml \
  --output-dir outputs/brain_age
```

When an aligned table of raw TIV-adjusted morphometry is available, add `--raw-tiv-input /path/to/brain_raw_tiv.csv` to estimate the paired normalization ablation with participant-level bootstrap inference.

```bash
python train_cognitive_age.py \
  --input /path/to/cognitive_normative.csv \
  --config config.yaml \
  --output-dir outputs/cognitive_age
```

```bash
python fit_nca_fusion.py \
  --input /path/to/nca_evaluation.csv \
  --config config.yaml \
  --brain-model outputs/brain_age/nca_brain_pipeline.joblib \
  --cognitive-model outputs/cognitive_age/nca_cognitive_pipeline.joblib \
  --output-dir outputs/fusion
```

```bash
python evaluate_oof.py \
  --input outputs/fusion/nca_outer_fold_predictions.csv \
  --config config.yaml \
  --output-dir outputs/evaluation
```

## Running the complete workflow

```bash
python run_workflow.py \
  --brain-data /path/to/brain_normative.csv \
  --cognitive-data /path/to/cognitive_normative.csv \
  --fusion-data /path/to/nca_evaluation.csv \
  --config config.yaml \
  --output-dir outputs
```

Each execution writes the resolved settings, fold assignments, model-selection results, out-of-fold predictions, fitted bias coefficients, and statistical summaries to the requested output directory.

## Statistical outputs

`evaluate_oof.py` reports:

- BA, CA, and NCA AUCs with diagnosis-stratified participant-level bootstrap confidence intervals;
- sensitivity, specificity, and balanced accuracy at fold-specific research thresholds;
- continuous age-calibration intercepts and slopes;
- paired DeLong comparisons between the retained weighted mean and the six alternative fusion strategies;
- Holm-adjusted p values for that six-comparison family;
- descriptive AUCs by sex, age, and education group when those fields are available.

Probability-calibration metrics are not calculated because the workflow evaluates continuous age scores and does not fit a mapping from NCA to diagnostic probability.

## Data governance

Only de-identified, authorized data should be used. Cohort-specific extraction, linkage, and access-control procedures must remain within the approved computing environment for each repository. Output files should be reviewed before external sharing to ensure that local identifiers or protected metadata have not been retained.

## License

MIT License. See `LICENSE`.
