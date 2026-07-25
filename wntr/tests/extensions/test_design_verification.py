"""Tests for the design-verification extension."""

import pandas as pd
import pytest
import wntr

from wntr.extensions.design_verification import (
    HydraulicScenario,
    PipeDesign,
    VerificationResult,
    apply_hydraulic_scenario,
    apply_pipe_design,
    AllPumpCurveResult,
    evaluate_maximum_velocity,
    evaluate_minimum_pressure,
    run_design_verification,
)
pytestmark = pytest.mark.extensions


def test_minimum_pressure_summary():
    """Identify the minimum pressure and its location."""
    pressure = pd.DataFrame(
        data=[
            [20.0, 14.0],
            [15.0, 16.0],
        ],
        index=[0, 3600],
        columns=["J1", "J2"],
    )

    result = evaluate_minimum_pressure(
        pressure=pressure,
        minimum_pressure_m=15.0,
    )

    assert result.critical_value == pytest.approx(14.0)
    assert result.compliance_pct == pytest.approx(75.0)
    assert result.feasible is False
    assert result.critical_component == "J2"
    assert result.critical_time == 0


def test_maximum_velocity_uses_absolute_value():
    """Use velocity magnitude when identifying violations."""
    velocity = pd.DataFrame(
        data=[
            [1.0, -2.7],
            [2.5, 0.5],
        ],
        index=[0, 3600],
        columns=["P1", "P2"],
    )

    result = evaluate_maximum_velocity(
        velocity=velocity,
        maximum_velocity_mps=2.5,
    )

    assert result.critical_value == pytest.approx(2.7)
    assert result.compliance_pct == pytest.approx(75.0)
    assert result.feasible is False
    assert result.critical_component == "P2"
    assert result.critical_time == 0


def test_partial_pressure_compliance():
    """Accept a result that meets a reduced compliance requirement."""
    pressure = pd.DataFrame(
        data=[
            [20.0, 14.0],
            [15.0, 16.0],
        ],
        columns=["J1", "J2"],
    )

    result = evaluate_minimum_pressure(
        pressure=pressure,
        minimum_pressure_m=15.0,
        required_compliance_pct=75.0,
    )

    assert result.compliance_pct == pytest.approx(75.0)
    assert result.feasible is True


def test_empty_pressure_table_is_rejected():
    """Reject an empty hydraulic-results table."""
    with pytest.raises(
        ValueError,
        match="pressure cannot be empty",
    ):
        evaluate_minimum_pressure(
            pressure=pd.DataFrame(),
            minimum_pressure_m=15.0,
        )


def test_invalid_velocity_limit_is_rejected():
    """Reject a zero maximum-velocity limit."""
    velocity = pd.DataFrame(
        data=[[1.0]],
        columns=["P1"],
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        evaluate_maximum_velocity(
            velocity=velocity,
            maximum_velocity_mps=0.0,
        )


def test_non_dataframe_pressure_is_rejected():
    """Require pressure results to use a pandas DataFrame."""
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        evaluate_minimum_pressure(
            pressure=[[20.0, 15.0]],
            minimum_pressure_m=15.0,
        )


def test_invalid_compliance_percentage_is_rejected():
    """Reject compliance requirements greater than 100 percent."""
    velocity = pd.DataFrame(
        data=[[1.0]],
        columns=["P1"],
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        evaluate_maximum_velocity(
            velocity=velocity,
            maximum_velocity_mps=2.5,
            required_compliance_pct=101.0,
        )

def build_small_network():
    """Create a small network for pipe-design tests."""
    wn = wntr.network.WaterNetworkModel()

    wn.add_reservoir(
        "R1",
        base_head=50.0,
        coordinates=(0.0, 0.0),
    )

    wn.add_junction(
        "J1",
        base_demand=0.01,
        demand_pattern=None,
        elevation=10.0,
        coordinates=(100.0, 0.0),
    )

    wn.add_pipe(
        "P1",
        start_node_name="R1",
        end_node_name="J1",
        length=100.0,
        diameter=0.150,
        roughness=100.0,
        minor_loss=0.0,
    )

    return wn


def test_apply_pipe_design_changes_copied_network():
    """Apply a diameter change to the copied network."""
    wn = build_small_network()

    design = PipeDesign(
        name="Alternative A",
        diameters_m={
            "P1": 0.250,
        },
    )

    trial_wn, audit = apply_pipe_design(
        wn=wn,
        design=design,
    )

    assert trial_wn.get_link("P1").diameter == pytest.approx(
        0.250
    )

    assert len(audit) == 1
    assert audit[0]["pipe_name"] == "P1"
    assert audit[0]["old_diameter_m"] == pytest.approx(
        0.150
    )
    assert audit[0]["new_diameter_m"] == pytest.approx(
        0.250
    )


def test_apply_pipe_design_preserves_original_network():
    """Confirm that the original model is not modified."""
    wn = build_small_network()

    original_diameter = wn.get_link("P1").diameter

    design = PipeDesign(
        name="Alternative A",
        diameters_m={
            "P1": 0.250,
        },
    )

    trial_wn, _ = apply_pipe_design(
        wn=wn,
        design=design,
    )

    assert wn.get_link("P1").diameter == pytest.approx(
        original_diameter
    )

    assert trial_wn.get_link("P1").diameter == pytest.approx(
        0.250
    )

    assert trial_wn is not wn


def test_apply_pipe_design_rejects_missing_pipe():
    """Reject a design containing an unknown pipe name."""
    wn = build_small_network()

    design = PipeDesign(
        name="Invalid alternative",
        diameters_m={
            "MissingPipe": 0.250,
        },
    )

    with pytest.raises(
        KeyError,
        match="does not exist",
    ):
        apply_pipe_design(
            wn=wn,
            design=design,
        )


def test_apply_pipe_design_rejects_negative_diameter():
    """Reject a proposed diameter that is not positive."""
    wn = build_small_network()

    design = PipeDesign(
        name="Invalid alternative",
        diameters_m={
            "P1": -0.250,
        },
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        apply_pipe_design(
            wn=wn,
            design=design,
        )


def test_apply_pipe_design_rejects_empty_design():
    """Reject a design containing no pipe changes."""
    wn = build_small_network()

    design = PipeDesign(
        name="Empty alternative",
        diameters_m={},
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        apply_pipe_design(
            wn=wn,
            design=design,
        )


def test_apply_hydraulic_scenario_changes_copied_network():
    """Apply hydraulic settings to an independent network copy."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Peak demand PDD",
        demand_multiplier=1.40,
        demand_model="PDD",
        duration_s=86400,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
        minimum_pressure_m=0.0,
        required_pressure_m=15.0,
    )

    scenario_wn, audit = apply_hydraulic_scenario(
        wn=wn,
        scenario=scenario,
    )

    assert (
        scenario_wn.options.hydraulic.demand_multiplier
        == pytest.approx(1.40)
    )

    assert str(
        scenario_wn.options.hydraulic.demand_model
    ).upper() in {"PDD", "PDA"}

    assert scenario_wn.options.time.duration == 86400
    assert (
        scenario_wn.options.time.hydraulic_timestep
        == 3600
    )
    assert (
        scenario_wn.options.time.report_timestep
        == 3600
    )

    assert (
        scenario_wn.options.hydraulic.minimum_pressure
        == pytest.approx(0.0)
    )
    assert (
        scenario_wn.options.hydraulic.required_pressure
        == pytest.approx(15.0)
    )

    assert audit["scenario_name"] == "Peak demand PDD"
    assert (
        audit["applied_settings"]["demand_multiplier"]
        == pytest.approx(1.40)
    )


def test_apply_hydraulic_scenario_preserves_original_network():
    """Confirm that scenario application does not change the original."""
    wn = build_small_network()

    original_multiplier = (
        wn.options.hydraulic.demand_multiplier
    )
    original_duration = wn.options.time.duration
    original_demand_model = str(
        wn.options.hydraulic.demand_model
    )

    scenario = HydraulicScenario(
        name="Peak demand",
        demand_multiplier=1.50,
        demand_model="PDD",
        duration_s=86400,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
        minimum_pressure_m=0.0,
        required_pressure_m=15.0,
    )

    scenario_wn, _ = apply_hydraulic_scenario(
        wn=wn,
        scenario=scenario,
    )

    assert scenario_wn is not wn

    assert (
        wn.options.hydraulic.demand_multiplier
        == pytest.approx(original_multiplier)
    )
    assert wn.options.time.duration == original_duration
    assert (
        str(wn.options.hydraulic.demand_model)
        == original_demand_model
    )

    assert (
        scenario_wn.options.hydraulic.demand_multiplier
        == pytest.approx(1.50)
    )


def test_apply_hydraulic_scenario_accepts_demand_model_alias():
    """Accept DDA as an alias for demand-driven analysis."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Demand-driven analysis",
        demand_model="DDA",
    )

    scenario_wn, _ = apply_hydraulic_scenario(
        wn=wn,
        scenario=scenario,
    )

    assert str(
        scenario_wn.options.hydraulic.demand_model
    ).upper() in {"DD", "DDA"}


def test_apply_hydraulic_scenario_rejects_invalid_multiplier():
    """Reject a demand multiplier that is not positive."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Invalid multiplier",
        demand_multiplier=0.0,
    )

    with pytest.raises(
        ValueError,
        match="demand_multiplier must be greater than zero",
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )


def test_apply_hydraulic_scenario_rejects_invalid_demand_model():
    """Reject an unsupported hydraulic demand model."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Invalid model",
        demand_model="UNKNOWN",
    )

    with pytest.raises(
        ValueError,
        match="demand_model must be one of",
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )


def test_apply_hydraulic_scenario_rejects_negative_duration():
    """Reject a negative hydraulic simulation duration."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Invalid duration",
        duration_s=-3600,
    )

    with pytest.raises(
        ValueError,
        match="duration_s must be greater than or equal to zero",
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )


def test_apply_hydraulic_scenario_rejects_zero_timestep():
    """Reject a hydraulic timestep equal to zero."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Invalid timestep",
        hydraulic_timestep_s=0,
    )

    with pytest.raises(
        ValueError,
        match="hydraulic_timestep_s must be greater than zero",
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )


def test_apply_hydraulic_scenario_rejects_pressure_inconsistency():
    """Require required pressure to exceed minimum pressure."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Invalid pressure settings",
        demand_model="PDD",
        minimum_pressure_m=15.0,
        required_pressure_m=10.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "required_pressure_m must be greater than "
            "minimum_pressure_m"
        ),
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )

def test_run_design_verification_returns_feasible_result():
    """Run a complete feasible hydraulic verification."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Baseline DD",
        demand_model="DD",
        duration_s=0,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
    )

    verification, results, audit = run_design_verification(
        wn=wn,
        scenario=scenario,
        simulator="WNTR",
        minimum_pressure_m=15.0,
        maximum_velocity_mps=2.5,
    )

    assert isinstance(
        verification,
        VerificationResult,
    )

    assert verification.design_name is None
    assert verification.scenario_name == "Baseline DD"
    assert verification.simulator_name == "WNTR"
    assert verification.pressure_result.feasible is True
    assert verification.velocity_result.feasible is True
    assert verification.feasible is True

    assert verification.pressure_result.critical_component == "J1"
    assert verification.velocity_result.critical_component == "P1"

    assert audit["junctions_assessed"] == ["J1"]
    assert audit["pipes_assessed"] == ["P1"]

    assert "pressure" in results.node
    assert "velocity" in results.link

    assert verification.pump_result is not None
    assert verification.pump_result.all_pumps_passed
    assert "pump_curve_audit" in audit
    assert audit["pump_curve_audit"]["all_pumps_passed"] is True

def test_run_design_verification_includes_pump_failure(
    monkeypatch,
):
    """Include pump-curve failure in overall feasibility."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Pump failure",
        demand_model="DD",
        duration_s=0,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
    )

    failed_pump_result = AllPumpCurveResult(
        head_pumps_in_network=1,
        head_pumps_evaluable=1,
        all_head_pumps_evaluable=True,
        all_pumps_passed=False,
        number_of_pumps_exceeding_curves=1,
        total_curve_exceedance_observations=1,
        governing_pump_name="PU1",
        maximum_pump_flow_ratio=1.2,
        pump_results=(),
    )

    monkeypatch.setattr(
        "wntr.extensions.design_verification.runner."
        "audit_head_pump_curves",
        lambda **kwargs: failed_pump_result,
    )

    verification, _, audit = run_design_verification(
        wn=wn,
        scenario=scenario,
        simulator="WNTR",
        minimum_pressure_m=15.0,
        maximum_velocity_mps=2.5,
    )

    assert verification.pressure_result.feasible is True
    assert verification.velocity_result.feasible is True
    assert verification.pump_result is failed_pump_result
    assert verification.pump_result.all_pumps_passed is False
    assert verification.feasible is False
    assert audit["pump_curve_audit"]["all_pumps_passed"] is False


def test_run_design_verification_applies_pipe_design():
    """Apply a proposed pipe diameter before simulation."""
    wn = build_small_network()

    design = PipeDesign(
        name="Larger pipe",
        diameters_m={
            "P1": 0.250,
        },
    )

    scenario = HydraulicScenario(
        name="Baseline DD",
        demand_model="DD",
    )

    verification, _, audit = run_design_verification(
        wn=wn,
        scenario=scenario,
        design=design,
        simulator="WNTR",
        minimum_pressure_m=15.0,
        maximum_velocity_mps=2.5,
    )

    assert verification.design_name == "Larger pipe"
    assert audit["design"] is not None
    assert len(audit["design"]) == 1

    assert (
        audit["design"][0]["pipe_name"]
        == "P1"
    )
    assert (
        audit["design"][0]["old_diameter_m"]
        == pytest.approx(0.150)
    )
    assert (
        audit["design"][0]["new_diameter_m"]
        == pytest.approx(0.250)
    )


def test_run_design_verification_preserves_original_network():
    """Protect the original network during a complete verification."""
    wn = build_small_network()

    original_diameter = wn.get_link("P1").diameter
    original_multiplier = (
        wn.options.hydraulic.demand_multiplier
    )

    design = PipeDesign(
        name="Larger pipe",
        diameters_m={
            "P1": 0.250,
        },
    )

    scenario = HydraulicScenario(
        name="Peak demand",
        demand_multiplier=1.40,
        demand_model="PDD",
        minimum_pressure_m=0.0,
        required_pressure_m=15.0,
    )

    run_design_verification(
        wn=wn,
        scenario=scenario,
        design=design,
        simulator="WNTR",
        minimum_pressure_m=15.0,
        maximum_velocity_mps=2.5,
    )

    assert (
        wn.get_link("P1").diameter
        == pytest.approx(original_diameter)
    )

    assert (
        wn.options.hydraulic.demand_multiplier
        == pytest.approx(original_multiplier)
    )


def test_run_design_verification_detects_pressure_failure():
    """Mark the verification infeasible when pressure is too low."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Pressure failure",
        demand_model="DD",
    )

    verification, _, _ = run_design_verification(
        wn=wn,
        scenario=scenario,
        simulator="WNTR",
        minimum_pressure_m=45.0,
        maximum_velocity_mps=2.5,
    )

    assert verification.pressure_result.feasible is False
    assert verification.velocity_result.feasible is True
    assert verification.feasible is False

    assert (
        verification.pressure_result.critical_value
        < 45.0
    )


def test_run_design_verification_detects_velocity_failure():
    """Mark the verification infeasible when velocity is too high."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Velocity failure",
        demand_model="DD",
    )

    verification, _, _ = run_design_verification(
        wn=wn,
        scenario=scenario,
        simulator="WNTR",
        minimum_pressure_m=15.0,
        maximum_velocity_mps=0.10,
    )

    assert verification.pressure_result.feasible is True
    assert verification.velocity_result.feasible is False
    assert verification.feasible is False

    assert (
        verification.velocity_result.critical_value
        > 0.10
    )


def test_run_design_verification_rejects_unknown_simulator():
    """Reject an unsupported hydraulic simulator name."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Invalid simulator",
    )

    with pytest.raises(
        ValueError,
        match="simulator must be WNTR or EPANET",
    ):
        run_design_verification(
            wn=wn,
            scenario=scenario,
            simulator="UNKNOWN",
        )


def test_apply_hydraulic_scenario_applies_pressure_exponent():
    """Apply and audit a custom pressure exponent."""
    wn = build_small_network()
    original_pressure_exponent = (
        wn.options.hydraulic.pressure_exponent
    )

    scenario = HydraulicScenario(
        name="Custom PDD exponent",
        demand_model="PDD",
        minimum_pressure_m=0.0,
        required_pressure_m=15.0,
        pressure_exponent=0.65,
    )

    scenario_wn, audit = apply_hydraulic_scenario(
        wn=wn,
        scenario=scenario,
    )

    assert (
        scenario_wn.options.hydraulic.pressure_exponent
        == pytest.approx(0.65)
    )
    assert (
        wn.options.hydraulic.pressure_exponent
        == pytest.approx(original_pressure_exponent)
    )
    assert (
        audit["original_settings"]["pressure_exponent"]
        == pytest.approx(original_pressure_exponent)
    )
    assert (
        audit["applied_settings"]["pressure_exponent"]
        == pytest.approx(0.65)
    )


def test_apply_hydraulic_scenario_preserves_pressure_exponent_when_omitted():
    """Preserve the existing exponent when none is requested."""
    wn = build_small_network()
    wn.options.hydraulic.pressure_exponent = 0.70

    scenario = HydraulicScenario(
        name="Existing PDD exponent",
        demand_model="PDD",
        minimum_pressure_m=0.0,
        required_pressure_m=15.0,
    )

    scenario_wn, audit = apply_hydraulic_scenario(
        wn=wn,
        scenario=scenario,
    )

    assert (
        scenario_wn.options.hydraulic.pressure_exponent
        == pytest.approx(0.70)
    )
    assert (
        audit["original_settings"]["pressure_exponent"]
        == pytest.approx(0.70)
    )
    assert (
        audit["applied_settings"]["pressure_exponent"]
        == pytest.approx(0.70)
    )


@pytest.mark.parametrize(
    "pressure_exponent",
    [0.0, -0.5],
)
def test_apply_hydraulic_scenario_rejects_nonpositive_pressure_exponent(
    pressure_exponent,
):
    """Reject zero and negative pressure exponents."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="Invalid pressure exponent",
        demand_model="PDD",
        minimum_pressure_m=0.0,
        required_pressure_m=15.0,
        pressure_exponent=pressure_exponent,
    )

    with pytest.raises(
        ValueError,
        match="pressure_exponent must be greater than zero",
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )


def test_dd_pressure_validation_ignores_inherited_unused_pdd_settings():
    """Do not reject DD because of unused inherited PDD settings."""
    wn = build_small_network()
    wn.options.hydraulic.minimum_pressure = 15.0
    wn.options.hydraulic.required_pressure = 10.0
    wn.options.hydraulic.pressure_exponent = 0.0

    scenario = HydraulicScenario(
        name="DD with unused PDD settings",
        demand_model="DD",
    )

    scenario_wn, audit = apply_hydraulic_scenario(
        wn=wn,
        scenario=scenario,
    )

    assert str(
        scenario_wn.options.hydraulic.demand_model
    ).upper() in {"DD", "DDA"}
    assert (
        scenario_wn.options.hydraulic.minimum_pressure
        == pytest.approx(15.0)
    )
    assert (
        scenario_wn.options.hydraulic.required_pressure
        == pytest.approx(10.0)
    )
    assert (
        scenario_wn.options.hydraulic.pressure_exponent
        == pytest.approx(0.0)
    )
    assert (
        audit["applied_settings"]["minimum_pressure_m"]
        == pytest.approx(15.0)
    )


def test_dd_pressure_validation_rejects_explicit_invalid_pdd_settings():
    """Validate PDD settings explicitly supplied to a DD scenario."""
    wn = build_small_network()

    scenario = HydraulicScenario(
        name="DD with invalid requested PDD settings",
        demand_model="DD",
        minimum_pressure_m=15.0,
        required_pressure_m=10.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "required_pressure_m must be greater than "
            "minimum_pressure_m"
        ),
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )


def test_pdd_pressure_validation_rejects_inherited_invalid_settings():
    """Validate inherited pressure settings for PDD simulation."""
    wn = build_small_network()
    wn.options.hydraulic.minimum_pressure = 15.0
    wn.options.hydraulic.required_pressure = 10.0

    scenario = HydraulicScenario(
        name="PDD with invalid inherited settings",
        demand_model="PDD",
    )

    with pytest.raises(
        ValueError,
        match=(
            "required_pressure_m must be greater than "
            "minimum_pressure_m"
        ),
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )


def test_run_design_verification_with_epanet():
    """Run a complete hydraulic verification using EPANET 2.2."""
    wn = build_small_network()

    original_diameter = wn.get_link("P1").diameter
    original_multiplier = (
        wn.options.hydraulic.demand_multiplier
    )

    scenario = HydraulicScenario(
        name="EPANET baseline DD",
        demand_model="DD",
        duration_s=0,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
    )

    verification, results, audit = run_design_verification(
        wn=wn,
        scenario=scenario,
        simulator="EPANET",
        minimum_pressure_m=15.0,
        maximum_velocity_mps=2.5,
    )

    assert isinstance(
        verification,
        VerificationResult,
    )
    assert verification.simulator_name == "EPANET"
    assert verification.scenario_name == "EPANET baseline DD"

    assert verification.pressure_result.feasible is True
    assert verification.velocity_result.feasible is True
    assert verification.feasible is True

    assert verification.pressure_result.critical_component == "J1"
    assert verification.velocity_result.critical_component == "P1"

    assert audit["simulator_name"] == "EPANET"
    assert audit["junctions_assessed"] == ["J1"]
    assert audit["pipes_assessed"] == ["P1"]

    assert "pressure" in results.node
    assert "velocity" in results.link
    assert "flowrate" in results.link

    assert verification.pump_result is not None
    assert (
        verification.pump_result.head_pumps_in_network
        == 0
    )

    assert (
        wn.get_link("P1").diameter
        == pytest.approx(original_diameter)
    )
    assert (
        wn.options.hydraulic.demand_multiplier
        == pytest.approx(original_multiplier)
    )


def test_run_design_verification_with_epanet_pump_audit():
    """Pass EPANET pump results through the curve-domain audit."""
    wn = wntr.network.WaterNetworkModel()

    wn.add_reservoir(
        "R1",
        base_head=20.0,
    )
    wn.add_junction(
        "J1",
        base_demand=0.0,
        elevation=0.0,
    )
    wn.add_junction(
        "J2",
        base_demand=0.01,
        elevation=0.0,
    )

    wn.add_curve(
        "curve1",
        "HEAD",
        [
            (0.005, 30.0),
            (0.015, 20.0),
            (0.030, 0.0),
        ],
    )
    wn.add_pump(
        "PU1",
        "R1",
        "J1",
        pump_type="HEAD",
        pump_parameter="curve1",
    )
    wn.add_pipe(
        "L1",
        start_node_name="J1",
        end_node_name="J2",
        length=100.0,
        diameter=0.150,
        roughness=100.0,
        minor_loss=0.0,
    )

    scenario = HydraulicScenario(
        name="EPANET pump audit",
        demand_model="DD",
        duration_s=0,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
    )

    verification, results, audit = run_design_verification(
        wn=wn,
        scenario=scenario,
        simulator="EPANET",
        minimum_pressure_m=0.0,
        maximum_velocity_mps=5.0,
    )

    pump_result = verification.pump_result

    assert pump_result is not None
    assert pump_result.head_pumps_in_network == 1
    assert pump_result.head_pumps_evaluable == 1
    assert pump_result.all_head_pumps_evaluable
    assert pump_result.all_pumps_passed
    assert (
        pump_result.number_of_pumps_exceeding_curves
        == 0
    )

    assert len(pump_result.pump_results) == 1

    pump_detail = pump_result.pump_results[0]

    assert pump_detail.pump_name == "PU1"
    assert pump_detail.evaluable
    assert pump_detail.passed
    assert pump_detail.active_observations >= 1
    assert (
        pump_detail.curve_maximum_flow_m3s
        == pytest.approx(0.030)
    )
    assert pump_detail.maximum_flow_ratio < 1.0

    assert verification.pressure_result.feasible
    assert verification.velocity_result.feasible
    assert verification.feasible

    assert "flowrate" in results.link
    assert "status" in results.link
    assert "setting" in results.link
    assert results.link["flowrate"]["PU1"].iloc[0] > 0.0

    assert (
        audit["pump_curve_audit"][
            "head_pumps_in_network"
        ]
        == 1
    )
    assert (
        audit["pump_curve_audit"][
            "all_pumps_passed"
        ]
        is True
    )
