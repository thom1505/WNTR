"""Scenario-based hydraulic design-verification extension."""

from .constraints import (
    evaluate_maximum_velocity,
    evaluate_minimum_pressure,
)
from .design import apply_pipe_design
from .models import (
    ConstraintResult,
    HydraulicScenario,
    PipeDesign,
)
from .scenarios import apply_hydraulic_scenario


__all__ = [
    "ConstraintResult",
    "HydraulicScenario",
    "PipeDesign",
    "apply_hydraulic_scenario",
    "apply_pipe_design",
    "evaluate_maximum_velocity",
    "evaluate_minimum_pressure",
]