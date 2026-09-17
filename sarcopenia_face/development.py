"""Public helpers for the manuscript-aligned model-development design.

These utilities implement the parts of the training design that can be shared
without releasing participant data, facial features, selected model weights, or
study-specific hyperparameters. They do not recreate the locked study models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .scoring import bmi_branch


CANDIDATE_CLASSIFIERS = (
    "support-vector machine",
    "random forest",
    "XGBoost",
    "LightGBM",
    "AdaBoost",
    "CatBoost",
)


def branch_sex_label_strata(
    bmi: Sequence[float], labels: Sequence[int], sex: Sequence[object]
) -> list[str]:
    """Build the single split strata used by the manuscript design.

    Each participant is assigned to a stratum defined by BMI branch,
    sarcopenia label, and sex. The function deliberately accepts no
    participant identifier: identifiers must never be used as model inputs.
    """
    if not (len(bmi) == len(labels) == len(sex)):
        raise ValueError("bmi, labels, and sex must have the same length")
    if not bmi:
        raise ValueError("at least one participant is required")

    strata: list[str] = []
    for row_number, (value, label, sex_value) in enumerate(
        zip(bmi, labels, sex, strict=True), start=1
    ):
        if label not in (0, 1):
            raise ValueError(f"row {row_number}: label must be 0 or 1")
        if sex_value is None or not str(sex_value).strip():
            raise ValueError(f"row {row_number}: sex must be present")
        strata.append(f"{bmi_branch(value)}|{sex_value}|{label}")
    return strata


def make_locked_split(
    bmi: Sequence[float],
    labels: Sequence[int],
    sex: Sequence[object],
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one participant-level train/test split stratified by study strata.

    Save the returned indices once and reuse them for both facial modalities.
    This function does not tune a split or inspect model performance.
    """
    strata = branch_sex_label_strata(bmi, labels, sex)
    indices = np.arange(len(strata))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strata,
    )
    return np.sort(train_index), np.sort(test_index)


def make_training_rfe_pipeline(
    estimator: Any,
    *,
    n_features_to_select: int,
    step: int | float = 1,
) -> Pipeline:
    """Create an unfitted training-only scaling and RFE pipeline.

    ``estimator`` must expose ``coef_`` or ``feature_importances_`` after
    fitting, as required by scikit-learn RFE. Fit the pipeline only with the
    training partition or inside cross-validation folds.
    """
    if n_features_to_select < 1:
        raise ValueError("n_features_to_select must be at least 1")
    pipeline = Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "rfe",
                RFE(
                    estimator=clone(estimator),
                    n_features_to_select=n_features_to_select,
                    step=step,
                ),
            ),
        ]
    )
    return pipeline


def fit_training_rfe(
    features: Any,
    labels: Sequence[int],
    estimator: Any,
    *,
    n_features_to_select: int,
    step: int | float = 1,
) -> Pipeline:
    """Fit scaling and RFE on training data only.

    Applying this function to a full cohort would leak test-set information.
    Use ``make_training_rfe_pipeline`` inside cross-validation.
    """
    return make_training_rfe_pipeline(
        estimator,
        n_features_to_select=n_features_to_select,
        step=step,
    ).fit(features, labels)


def cross_validated_candidate_summary(
    features: Any,
    labels: Sequence[int],
    candidates: dict[str, Any],
    *,
    folds: int = 5,
    random_state: int | None = None,
) -> list[dict[str, float | str]]:
    """Summarize non-repeated stratified CV ROC AUC and AUPR for candidates.

    Each candidate should be an unfitted scikit-learn pipeline that contains all
    learned preprocessing and feature selection. The function reports results;
    it deliberately does not select or fit a final model.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not candidates:
        raise ValueError("at least one candidate model is required")
    checked_labels = list(labels)
    if not checked_labels or set(checked_labels) - {0, 1} or len(set(checked_labels)) != 2:
        raise ValueError("labels must contain both 0 and 1")
    if min(checked_labels.count(0), checked_labels.count(1)) < folds:
        raise ValueError("each label class must contain at least one participant per fold")

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    summaries: list[dict[str, float | str]] = []
    for name, estimator in candidates.items():
        scores = cross_validate(
            clone(estimator),
            features,
            checked_labels,
            cv=splitter,
            scoring={"roc_auc": "roc_auc", "aupr": "average_precision"},
            return_train_score=False,
        )
        summaries.append(
            {
                "candidate": str(name),
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "cv_roc_auc_sd": float(np.std(scores["test_roc_auc"], ddof=1)),
                "cv_aupr_mean": float(np.mean(scores["test_aupr"])),
                "cv_aupr_sd": float(np.std(scores["test_aupr"], ddof=1)),
            }
        )
    return summaries
