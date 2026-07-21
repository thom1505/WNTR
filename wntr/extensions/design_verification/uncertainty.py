"""Generate reproducible uncertainty realizations for hydraulic studies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UncertaintyVariable:
    """Define a bounded continuous uncertainty variable.

    Parameters
    ----------
    name
        Unique variable name used as a sample-table column.
    lower
        Inclusive lower bound.
    upper
        Inclusive upper bound. A value equal to ``lower`` represents
        a fixed parameter retained in the uncertainty table.
    """

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Uncertainty-variable name cannot be empty.")

        lower = _finite_float(self.lower, "lower")
        upper = _finite_float(self.upper, "upper")

        if upper < lower:
            raise ValueError(
                f"Upper bound for {self.name!r} cannot be below "
                "the lower bound."
            )

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def _finite_float(value: object, parameter_name: str) -> float:
    """Return a finite floating-point value."""
    if isinstance(value, bool):
        raise ValueError(f"{parameter_name} must be numeric, not Boolean.")

    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{parameter_name} must be numeric.") from error

    if not math.isfinite(converted):
        raise ValueError(f"{parameter_name} must be finite.")

    return converted


def _normalise_variables(
    variables: Sequence[UncertaintyVariable],
) -> list[UncertaintyVariable]:
    """Validate uncertainty variables and preserve their order."""
    if isinstance(variables, (str, bytes)):
        raise TypeError(
            "variables must be a sequence of UncertaintyVariable objects."
        )

    values = list(variables)
    if not values:
        raise ValueError("variables must contain at least one item.")

    for variable in values:
        if not isinstance(variable, UncertaintyVariable):
            raise TypeError(
                "Each uncertainty variable must be an "
                "UncertaintyVariable object."
            )

    names = [variable.name for variable in values]
    if len(names) != len(set(names)):
        raise ValueError("Uncertainty-variable names must be unique.")

    reserved = {
        "realization_id",
        "realization_number",
        "realization_hash",
        "sampling_method",
        "random_seed",
        "numpy_version",
    }
    conflicts = sorted(set(names).intersection(reserved))
    if conflicts:
        raise ValueError(
            "Uncertainty-variable names conflict with reserved columns: "
            + ", ".join(conflicts)
        )

    return values


def _normalise_method(method: object) -> str:
    """Return a canonical uncertainty-sampling method."""
    if not isinstance(method, str):
        raise ValueError("method must be a string.")

    normalized = method.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "LHS": "LATIN_HYPERCUBE",
        "LATIN_HYPERCUBE": "LATIN_HYPERCUBE",
        "LATINHYPERCUBE": "LATIN_HYPERCUBE",
        "RANDOM": "RANDOM",
        "MONTE_CARLO": "RANDOM",
        "MONTECARLO": "RANDOM",
    }

    if normalized not in aliases:
        raise ValueError(
            "method must be LATIN_HYPERCUBE/LHS or RANDOM/MONTE_CARLO."
        )

    return aliases[normalized]


def _canonical_json(value: object) -> str:
    """Serialize sample metadata deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def sample_uncertainty(
    variables: Sequence[UncertaintyVariable],
    sample_count: int,
    *,
    seed: int = 42,
    method: str = "LATIN_HYPERCUBE",
) -> pd.DataFrame:
    """Generate a reproducible table of uncertainty realizations.

    The generated values should be exported and treated as the exact
    experiment inputs. Recording them avoids dependence on random-number
    implementation details across future NumPy versions.

    Parameters
    ----------
    variables
        Bounded continuous variables to sample.
    sample_count
        Number of realizations. Must be a positive whole number.
    seed
        Random seed used to generate the sample.
    method
        ``LATIN_HYPERCUBE``/``LHS`` or ``RANDOM``/``MONTE_CARLO``.

    Returns
    -------
    pandas.DataFrame
        One row per realization, including stable identifiers, sampling
        metadata, and sampled variable values.
    """
    variable_values = _normalise_variables(variables)

    if isinstance(sample_count, bool):
        raise ValueError("sample_count must be a positive whole number.")

    try:
        sample_count_int = int(sample_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "sample_count must be a positive whole number."
        ) from error

    if sample_count_int != sample_count or sample_count_int <= 0:
        raise ValueError("sample_count must be a positive whole number.")

    if isinstance(seed, bool):
        raise ValueError("seed must be a whole number, not Boolean.")

    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("seed must be a whole number.") from error

    if seed_int != seed:
        raise ValueError("seed must be a whole number.")

    method_name = _normalise_method(method)
    rng = np.random.default_rng(seed_int)
    sampled_columns: dict[str, np.ndarray] = {}

    for variable in variable_values:
        if variable.lower == variable.upper:
            unit_values = np.zeros(sample_count_int, dtype=float)
        elif method_name == "LATIN_HYPERCUBE":
            unit_values = (
                np.arange(sample_count_int, dtype=float)
                + rng.random(sample_count_int)
            ) / sample_count_int
            rng.shuffle(unit_values)
        else:
            unit_values = rng.random(sample_count_int)

        sampled_columns[variable.name] = (
            variable.lower
            + unit_values * (variable.upper - variable.lower)
        )

    samples = pd.DataFrame(sampled_columns)
    realization_hashes: list[str] = []
    realization_ids: list[str] = []

    for row_number, row in samples.iterrows():
        payload = {
            "realization_number": int(row_number) + 1,
            "sampling_method": method_name,
            "random_seed": seed_int,
            "values": {
                variable.name: float(row[variable.name])
                for variable in variable_values
            },
        }
        digest = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        realization_hashes.append(digest)
        realization_ids.append(
            f"u-{int(row_number) + 1:05d}-{digest[:10]}"
        )

    samples.insert(
        0,
        "realization_number",
        np.arange(1, sample_count_int + 1, dtype=int),
    )
    samples.insert(0, "realization_id", realization_ids)
    samples.insert(2, "realization_hash", realization_hashes)
    samples.insert(3, "sampling_method", method_name)
    samples.insert(4, "random_seed", seed_int)
    samples.insert(5, "numpy_version", np.__version__)

    return samples
