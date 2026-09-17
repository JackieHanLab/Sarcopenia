"""Public utilities for manuscript-aligned sarcopenia face-score evaluation."""

from .development import (
    CANDIDATE_CLASSIFIERS,
    cross_validated_candidate_summary,
    fit_training_rfe,
    make_locked_split,
    make_training_rfe_pipeline,
)
from .scoring import BMI_CUTOFF, bmi_branch, equal_weight_fusion
from .screening import DEFAULT_THRESHOLDS, threshold_screening_metrics

__all__ = [
    "BMI_CUTOFF",
    "CANDIDATE_CLASSIFIERS",
    "DEFAULT_THRESHOLDS",
    "bmi_branch",
    "cross_validated_candidate_summary",
    "equal_weight_fusion",
    "fit_training_rfe",
    "make_locked_split",
    "make_training_rfe_pipeline",
    "threshold_screening_metrics",
]
