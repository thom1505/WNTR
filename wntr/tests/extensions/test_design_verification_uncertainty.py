"""Tests for uncertainty-aware design verification utilities."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from wntr.extensions.design_verification import (
    UncertaintyVariable,
    sample_uncertainty,
    summarize_robustness,
    wilson_score_interval,
)


def test_latin_hypercube_sampling_is_reproducible_and_bounded():
    variables = [
        UncertaintyVariable("demand_multiplier", 0.8, 1.3),
        UncertaintyVariable("roughness_c", 110.0, 150.0),
    ]

    first = sample_uncertainty(variables, 20, seed=42, method="LHS")
    second = sample_uncertainty(
        variables,
        20,
        seed=42,
        method="LATIN_HYPERCUBE",
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["realization_id"].is_unique
    assert first["realization_hash"].is_unique
    assert first["demand_multiplier"].between(0.8, 1.3).all()
    assert first["roughness_c"].between(110.0, 150.0).all()


def test_sampling_rejects_duplicate_variable_names():
    variables = [
        UncertaintyVariable("demand", 0.8, 1.2),
        UncertaintyVariable("demand", 0.9, 1.1),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        sample_uncertainty(variables, 10)


def test_fixed_uncertainty_variable_is_retained():
    samples = sample_uncertainty(
        [UncertaintyVariable("source_head_m", 35.0, 35.0)],
        5,
        seed=9,
    )

    assert (samples["source_head_m"] == 35.0).all()


def test_wilson_interval_contains_observed_proportion():
    lower, upper = wilson_score_interval(8, 10, confidence_level=0.95)

    assert 0.0 <= lower < 0.8 < upper <= 1.0


def test_robustness_summary_reports_probability_and_severity():
    summary = pd.DataFrame(
        {
            "status": ["completed", "completed", "failed"],
            "overall_feasible": [True, False, None],
            "design_name": ["Design A", "Design A", "Design A"],
            "simulator_name": ["WNTR", "WNTR", "WNTR"],
            "minimum_pressure_m": [16.0, 14.0, None],
            "maximum_velocity_mps": [2.0, 2.8, None],
            "pressure_margin_m": [1.0, -1.0, None],
            "velocity_margin_mps": [0.5, -0.3, None],
            "pressure_compliance_pct": [100.0, 95.0, None],
            "velocity_compliance_pct": [100.0, 98.0, None],
            "critical_pressure_component": ["J1", "J2", None],
            "critical_velocity_component": ["P1", "P1", None],
        }
    )

    result = summarize_robustness(summary)
    row = result.iloc[0]

    assert row["requested_experiments"] == 3
    assert row["completed_experiments"] == 2
    assert row["simulation_failures"] == 1
    assert row["hydraulically_feasible_experiments"] == 1
    assert row["feasibility_probability_pct"] == pytest.approx(50.0)
    assert row["overall_success_probability_pct"] == pytest.approx(
        100.0 / 3.0
    )
    assert row["expected_pressure_deficit_m"] == pytest.approx(0.5)
    assert row["expected_velocity_exceedance_mps"] == pytest.approx(0.15)
    assert row["most_frequent_critical_velocity_component"] == "P1"
    assert not math.isnan(row["feasibility_ci_lower_pct"])
