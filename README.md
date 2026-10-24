# Multimodal facial phenotyping for sarcopenia

Public companion code for **“Multimodal facial phenotyping reveals sarcopenia-related features and supports cross-cohort screening.”** The repository implements the manuscript-defined parts of the workflow that can be shared without disclosing identifiable facial images, participant-level data, derived cohort files, trained models, or restricted UK Biobank materials.

## What this repository implements

The study used AWGS 2025 sarcopenia labels and BMI to select a model branch: lower BMI, <24 kg m<sup>−2</sup>; higher BMI, ≥24 kg m<sup>−2</sup>. BMI is used only for routing and is not a model predictor.

The public code implements the following analysis contracts:

1. A single participant-level train/test split stratified by BMI branch, sarcopenia status, and sex.
2. Training-only standardization and recursive feature elimination, including a reusable non-repeated stratified cross-validation summary for ROC AUC and AUPR.
3. BMI-routed application of paired locked 3D and thermal scores.
4. Equal-weight late fusion of the two modality scores.
5. Branch-specific ROC AUC, AUPR, and two-sided 95% DeLong confidence intervals.
6. Sensitivity, specificity, positive predictive value, negative predictive value, and referral proportion across prespecified thresholds from 0.30 to 0.60.

The model-development protocol in the manuscript evaluated support-vector machines, random forests, XGBoost, LightGBM, AdaBoost, and CatBoost with non-repeated fivefold stratified cross-validation. The specific feature matrices, fitted models, selected ensembles, tuning configurations, and model weights are restricted.

## Quick start

Use Python 3.11. Install the public dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run the evaluator on a CSV containing locked paired predictions:

```bash
python -m sarcopenia_face.evaluation --input path/to/locked_scores.csv
```

Run the tests:

```bash
python -m unittest discover -s tests
```

## Input contract for locked-score evaluation

The evaluator accepts one row per participant with four columns:

| Column | Meaning |
| --- | --- |
| `bmi` | Positive BMI value used for branch routing. |
| `label` | Sarcopenia status encoded as `0` or `1`. |
| `score_3d` | Locked 3D-model score from the BMI-selected branch. |
| `score_thermal` | Locked thermal-model score from the BMI-selected branch. |

Scores must be finite values from 0 to 1. Fusion requires both modalities, so incomplete paired records are rejected rather than silently dropped. Each BMI branch requires at least two cases and two controls for DeLong intervals. The evaluator reports aggregate metrics only and does not write participant-level output unless a user explicitly supplies an output path.

## Repository layout

```text
sarcopenia_face/
  development.py    # split strata, training-only scaling/RFE, and CV summaries
  scoring.py        # BMI routing and equal-weight late fusion
  evaluation.py     # AUC, AUPR, and DeLong confidence intervals
  screening.py      # prespecified threshold-dependent screening summaries
tests/
  test_public_workflow.py
```

## Interpretation and scope

The output is a continuous research screening score. It is not a calibrated absolute-risk estimate or a diagnosis. The manuscript's 0.45 value is a representative working point, rather than a clinically optimized cutoff; the code evaluates supplied thresholds and does not choose one.

## License

The public code is distributed under the MIT License. Please cite the associated manuscript when using this repository.
