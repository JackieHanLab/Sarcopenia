"""Evaluate locked paired model scores by BMI branch without applying a cutoff."""

from __future__ import annotations

import argparse
import csv
import sys
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping, TextIO

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .scoring import Branch, bmi_branch, equal_weight_fusion


REQUIRED_COLUMNS = ("bmi", "label", "score_3d", "score_thermal")
SUMMARY_COLUMNS = (
    "bmi_branch",
    "n",
    "cases",
    "auc_3d",
    "auc_3d_ci95_low",
    "auc_3d_ci95_high",
    "auc_thermal",
    "auc_thermal_ci95_low",
    "auc_thermal_ci95_high",
    "auc_fusion",
    "auc_fusion_ci95_low",
    "auc_fusion_ci95_high",
    "aupr_3d",
    "aupr_thermal",
    "aupr_fusion",
)
Prediction = tuple[int, float, float, float]
Z_975 = 1.959963984540054


def _number(row: Mapping[str, object], column: str, row_number: int) -> float:
    value = row.get(column)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: column {column!r} must contain a finite number"
        ) from exc
    if not isfinite(number):
        raise ValueError(
            f"row {row_number}: column {column!r} must contain a finite number"
        )
    return number


def _midranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, retaining average ranks for ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _auc_with_delong_ci(
    labels: list[int], scores: list[float]
) -> tuple[float, float, float]:
    """Calculate ROC AUC and its two-sided 95% DeLong confidence interval."""
    label_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    positive = score_array[label_array == 1]
    negative = score_array[label_array == 0]
    positive_count = len(positive)
    negative_count = len(negative)
    if positive_count < 2 or negative_count < 2:
        raise ValueError("DeLong intervals require at least two cases and two controls")

    positive_ranks = _midranks(positive)
    negative_ranks = _midranks(negative)
    pooled_ranks = _midranks(np.concatenate((positive, negative)))
    auc_from_ranks = float(
        pooled_ranks[:positive_count].sum() / (positive_count * negative_count)
        - (positive_count + 1) / (2.0 * negative_count)
    )
    auc = float(roc_auc_score(label_array, score_array))
    if not np.isclose(auc, auc_from_ranks, rtol=0.0, atol=1e-12):
        raise ArithmeticError("DeLong rank calculation disagrees with ROC AUC")

    positive_placements = (
        pooled_ranks[:positive_count] - positive_ranks
    ) / negative_count
    negative_placements = 1.0 - (
        pooled_ranks[positive_count:] - negative_ranks
    ) / positive_count
    variance = (
        np.var(positive_placements, ddof=1) / positive_count
        + np.var(negative_placements, ddof=1) / negative_count
    )
    standard_error = float(np.sqrt(max(float(variance), 0.0)))
    lower = max(0.0, auc - Z_975 * standard_error)
    upper = min(1.0, auc + Z_975 * standard_error)
    return auc, lower, upper


def evaluate_paired_predictions(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, int | float | str]]:
    """Summarize ROC AUC and DeLong intervals for each BMI branch.

    Each row must contain BMI, binary label, 3D score, and thermal score.
    All metrics use the same complete paired participants. No cutoff is applied.
    """
    grouped: dict[Branch, list[Prediction]] = {"lower": [], "higher": []}

    for row_number, row in enumerate(rows, start=2):
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"row {row_number}: missing required columns {missing}")

        bmi = _number(row, "bmi", row_number)
        label_value = _number(row, "label", row_number)
        if label_value not in (0.0, 1.0):
            raise ValueError(f"row {row_number}: label must be 0 or 1")
        score_3d = _number(row, "score_3d", row_number)
        score_thermal = _number(row, "score_thermal", row_number)

        branch = bmi_branch(bmi)
        fusion = equal_weight_fusion(score_3d, score_thermal)
        grouped[branch].append((int(label_value), score_3d, score_thermal, fusion))

    if not any(grouped.values()):
        raise ValueError("input contains no participant rows")

    summaries: list[dict[str, int | float | str]] = []
    for branch in ("lower", "higher"):
        predictions = grouped[branch]
        if not predictions:
            continue

        labels = [row[0] for row in predictions]
        cases = sum(labels)
        if cases == 0 or cases == len(labels):
            raise ValueError(
                f"BMI branch {branch!r} must contain both label classes to calculate AUC"
            )

        auc_3d, low_3d, high_3d = _auc_with_delong_ci(
            labels, [row[1] for row in predictions]
        )
        auc_thermal, low_thermal, high_thermal = _auc_with_delong_ci(
            labels, [row[2] for row in predictions]
        )
        auc_fusion, low_fusion, high_fusion = _auc_with_delong_ci(
            labels, [row[3] for row in predictions]
        )
        summaries.append(
            {
                "bmi_branch": branch,
                "n": len(predictions),
                "cases": cases,
                "auc_3d": auc_3d,
                "auc_3d_ci95_low": low_3d,
                "auc_3d_ci95_high": high_3d,
                "auc_thermal": auc_thermal,
                "auc_thermal_ci95_low": low_thermal,
                "auc_thermal_ci95_high": high_thermal,
                "auc_fusion": auc_fusion,
                "auc_fusion_ci95_low": low_fusion,
                "auc_fusion_ci95_high": high_fusion,
                "aupr_3d": float(
                    average_precision_score(labels, [row[1] for row in predictions])
                ),
                "aupr_thermal": float(
                    average_precision_score(labels, [row[2] for row in predictions])
                ),
                "aupr_fusion": float(
                    average_precision_score(labels, [row[3] for row in predictions])
                ),
            }
        )
    return summaries


def _write_summary(
    summaries: list[dict[str, int | float | str]], destination: TextIO
) -> None:
    writer = csv.DictWriter(destination, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    writer.writerows(summaries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate branch-specific ROC AUC and DeLong 95% CIs from locked paired scores."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV with bmi, label, score_3d, and score_thermal columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for aggregate metrics (default: print to standard output)",
    )
    args = parser.parse_args(argv)

    try:
        with args.input.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError("input CSV must include a header row")
            missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"input CSV is missing required columns {missing}")
            summaries = evaluate_paired_predictions(reader)

        if args.output is None:
            _write_summary(summaries, sys.stdout)
        else:
            with args.output.open("w", newline="", encoding="utf-8") as destination:
                _write_summary(summaries, destination)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
