"""Exceptions raised by hydraulic design verification."""


class DesignVerificationError(Exception):
    """Base exception for design-verification failures."""


class InvalidDesignError(
    DesignVerificationError,
    ValueError,
):
    """Raised when a proposed pipe design is invalid."""


class InvalidScenarioError(
    DesignVerificationError,
    ValueError,
):
    """Raised when a hydraulic scenario is invalid."""


class InvalidConstraintError(
    DesignVerificationError,
    ValueError,
):
    """Raised when hydraulic constraint settings are invalid."""


class UnsupportedSimulatorError(
    DesignVerificationError,
    ValueError,
):
    """Raised when the requested simulator is unsupported."""


class IncompleteHydraulicResultsError(
    DesignVerificationError,
    RuntimeError,
):
    """Raised when simulator results cannot support verification."""
