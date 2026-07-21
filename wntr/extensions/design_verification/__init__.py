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
from .robustness import (
    run_robust_verification,
    summarize_robustness,
    wilson_score_interval,
)
from .runner import run_design_verification
from .scenarios import apply_hydraulic_scenario
from .uncertainty import (
    UncertaintyVariable,
    sample_uncertainty,
)

__all__ = [
    "ConstraintResult",
    "HydraulicScenario",
    "PipeDesign",
    "UncertaintyVariable",
    "VerificationResult",
    "apply_hydraulic_scenario",
    "apply_pipe_design",
    "evaluate_maximum_velocity",
    "evaluate_minimum_pressure",
    "run_design_verification",
    "run_robust_verification",
    "run_verification_batch",
    "sample_uncertainty",
    "summarize_robustness",
    "wilson_score_interval",
]
