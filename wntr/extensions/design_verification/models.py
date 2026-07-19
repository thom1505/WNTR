"""Data models for hydraulic design verification."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
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


@dataclass(frozen=True)
class PipeDesign:
    """Describe proposed diameter changes for selected pipes.

    Parameters
    ----------
    name
        User-facing name for the design alternative.
    diameters_m
        Mapping of pipe names to proposed diameters in metres.
    """

    name: str
    diameters_m: Mapping[str, float]


@dataclass(frozen=True)
class HydraulicScenario:
    """Describe hydraulic operating conditions for one assessment.

    Parameters
    ----------
    name
        User-facing name for the hydraulic scenario.
    demand_multiplier
        Global multiplier applied to all junction demands.
    demand_model
        Hydraulic demand formulation, normally ``DD`` or ``PDD``.
    duration_s
        Total simulation duration in seconds. A value of zero
        represents a steady-state simulation.
    hydraulic_timestep_s
        Hydraulic calculation timestep in seconds.
    report_timestep_s
        Hydraulic-results reporting timestep in seconds.
    minimum_pressure_m
        Optional global minimum pressure for pressure-dependent
        demand analysis, in metres.
    required_pressure_m
        Optional global required pressure for pressure-dependent
        demand analysis, in metres.
    """

    name: str
    demand_multiplier: float = 1.0
    demand_model: str = "DD"
    duration_s: int = 0
    hydraulic_timestep_s: int = 3600
    report_timestep_s: int = 3600
    minimum_pressure_m: float | None = None
    required_pressure_m: float | None = None