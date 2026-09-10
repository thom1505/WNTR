"""Data models for hydraulic design verification."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wntr.network import Options

if TYPE_CHECKING:
    from .pumps import AllPumpCurveResult


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
    """Describe simulation options for one hydraulic assessment.

    Parameters
    ----------
    name
        User-facing name for the hydraulic scenario.
    options
        WNTR simulation options. The design-verification extension
        applies the time and hydraulic option groups to an independent
        copy of the water network model.
    """

    name: str
    options: Options


@dataclass(frozen=True)
class VerificationResult:
    """Store the combined result of one hydraulic verification run.

    Parameters
    ----------
    design_name
        Name of the applied pipe design, or ``None`` for the
        unchanged baseline network.
    scenario_name
        Name of the hydraulic scenario.
    simulator_name
        Name of the hydraulic simulator used.
    pressure_result
        Minimum-pressure constraint result.
    velocity_result
        Maximum-velocity constraint result.
    feasible
        Whether pressure, velocity and applicable pump operating-curve
        requirements were met.
    simulation_error_code
        Simulator error code, when one is supplied by WNTR.
    pump_result
        Head-pump operating-curve audit result, when available.
    """

    design_name: str | None
    scenario_name: str
    simulator_name: str
    pressure_result: ConstraintResult
    velocity_result: ConstraintResult
    feasible: bool
    simulation_error_code: int | None = None
    pump_result: AllPumpCurveResult | None = None