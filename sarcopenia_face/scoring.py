"""BMI routing and late fusion of locked three-dimensional and thermal scores."""

from math import isfinite
from typing import Literal


BMI_CUTOFF = 24.0
Branch = Literal["lower", "higher"]


def _finite_number(name: str, value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def bmi_branch(bmi: float) -> Branch:
    """Select the manuscript branch: BMI <24 or BMI >=24 kg m-2."""
    value = _finite_number("BMI", bmi)
    if value <= 0:
        raise ValueError("BMI must be positive")
    return "lower" if value < BMI_CUTOFF else "higher"


def _model_score(name: str, score: float) -> float:
    value = _finite_number(name, score)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def equal_weight_fusion(score_3d: float, score_thermal: float) -> float:
    """Return the arithmetic mean of paired 3D and thermal model scores."""
    three_d = _model_score("score_3d", score_3d)
    thermal = _model_score("score_thermal", score_thermal)
    return (three_d + thermal) / 2.0
