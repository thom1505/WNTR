"""Data models for hydraulic design verification."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintResult:
    """Store the result of one hydraulic-constraint assessment.

    Parameters
    ----------
    critical_value
        Minimum pressure or maximum absolute velocity found in the
        hydraulic-results table.
    compliance_pct
        Percentage of assessed values satisfying the constraint.
    feasible
        Whether the required compliance percentage was achieved.
    critical_component
        Name of the junction or pipe containing the critical value.
    critical_time
        Simulation time corresponding to the critical value.
    """

    critical_value: float
    compliance_pct: float
    feasible: bool
    critical_component: str
    critical_time: Hashable