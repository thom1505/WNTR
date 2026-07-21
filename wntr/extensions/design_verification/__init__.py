"""Scenario-based hydraulic design-verification extension."""

from .batch import run_verification_batch
from .constraints import (
    evaluate_maximum_velocity,
    evaluate_minimum_pressure,
)
from .design import apply_pipe_design
from .models import (
    ConstraintResult,
    HydraulicScenario,
    PipeDesign,
    VerificationResult,
)
from .runner import run_design_verification
from .scenarios import apply_hydraulic_scenario


__all__ = [
    "ConstraintResult",
    "HydraulicScenario",
    "PipeDesign",
    "VerificationResult",
    "apply_hydraulic_scenario",
    "apply_pipe_design",
    "evaluate_maximum_velocity",
    "evaluate_minimum_pressure",
    "run_design_verification",
    "run_verification_batch",
]