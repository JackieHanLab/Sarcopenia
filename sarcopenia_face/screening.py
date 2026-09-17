"""Threshold-dependent screening summaries for locked continuous scores."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite


DEFAULT_THRESHOLDS = tuple(round(value / 100, 2) for value in range(30, 61, 5))


def _validated_scores(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[list[int], list[float]]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("at least one participant is required")

    checked_labels: list[int] = []
    checked_scores: list[float] = []
    for row_number, (label, score) in enumerate(
        zip(labels, scores, strict=True), start=1
    ):
        if label not in (0, 1):
            raise ValueError(f"row {row_number}: label must be 0 or 1")
        try:
            value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {row_number}: score must be a finite number") from exc
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"row {row_number}: score must be between 0 and 1")
        checked_labels.append(int(label))
        checked_scores.append(value)

    if len(set(checked_labels)) != 2:
        raise ValueError("both label classes are required for screening metrics")
    return checked_labels, checked_scores


def threshold_screening_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float | int]]:
    """Calculate manuscript-defined screening metrics for supplied thresholds.

    A participant is referred when the continuous score is greater than or equal
    to the threshold. Thresholds are evaluated as supplied and are never
    optimized by this function.
    """
    checked_labels, checked_scores = _validated_scores(labels, scores)
    cases = sum(checked_labels)
    controls = len(checked_labels) - cases
    results: list[dict[str, float | int]] = []

    for threshold in thresholds:
        try:
            value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("thresholds must be finite numbers") from exc
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")

        referred = [score >= value for score in checked_scores]
        true_positive = sum(
            flag and label == 1
            for flag, label in zip(referred, checked_labels, strict=True)
        )
        false_positive = sum(
            flag and label == 0
            for flag, label in zip(referred, checked_labels, strict=True)
        )
        false_negative = cases - true_positive
        true_negative = controls - false_positive
        positive_predictive_value = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        negative_predictive_value = (
            true_negative / (true_negative + false_negative)
            if true_negative + false_negative
            else 0.0
        )
        results.append(
            {
                "threshold": value,
                "n": len(checked_labels),
                "cases": cases,
                "sensitivity": true_positive / cases,
                "specificity": true_negative / controls,
                "positive_predictive_value": positive_predictive_value,
                "negative_predictive_value": negative_predictive_value,
                "referral_proportion": sum(referred) / len(referred),
            }
        )
    return results
