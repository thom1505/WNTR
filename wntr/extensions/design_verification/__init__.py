"""Scenario-based hydraulic design-verification extension."""

from .constraints import (
    evaluate_maximum_velocity,
    evaluate_minimum_pressure,
)
from .models import ConstraintResult


__all__ = [
    "ConstraintResult",
    "evaluate_maximum_velocity",
    "evaluate_minimum_pressure",
]