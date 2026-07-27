"""Hydraulic design-verification extension for WNTR.

The extension applies proposed pipe designs and hydraulic operating
scenarios to independent copies of a water network model, runs hydraulic
simulation, and evaluates engineering constraints.
"""

from .batch import run_verification_batch
from .constraints import (
    evaluate_maximum_velocity,
    evaluate_minimum_pressure,
)
from .exceptions import (
    DesignVerificationError,
    IncompleteHydraulicResultsError,
    InvalidConstraintError,
    InvalidDesignError,
    InvalidScenarioError,
    UnsupportedSimulatorError,
)
from .design import apply_pipe_design
from .models import (
    ConstraintResult,
    HydraulicScenario,
    PipeDesign,
    VerificationResult,
)
from .pumps import (
    AllPumpCurveResult,
    PumpCurveResult,
    audit_head_pump_curves,
)
from .runner import run_design_verification
from .scenarios import apply_hydraulic_scenario

__all__ = [
    "AllPumpCurveResult",
    "ConstraintResult",
    "DesignVerificationError",
    "IncompleteHydraulicResultsError",
    "InvalidConstraintError",
    "InvalidDesignError",
    "InvalidScenarioError",
    "UnsupportedSimulatorError",
    "HydraulicScenario",
    "PipeDesign",
    "PumpCurveResult",
    "VerificationResult",
    "apply_hydraulic_scenario",
    "apply_pipe_design",
    "audit_head_pump_curves",
    "evaluate_maximum_velocity",
    "evaluate_minimum_pressure",
    "run_design_verification",
    "run_verification_batch",
]