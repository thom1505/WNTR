.. _hydraulic_design_verification:

Hydraulic Design Verification
=============================

**Summary:** The ``design_verification`` extension applies proposed pipe
designs and hydraulic operating scenarios to independent copies of a
WNTR water-distribution network model. It runs a hydraulic simulation
and checks whether pressure, velocity and pump operating requirements
are satisfied.

**Point of contact:** Rheal Thomas,
https://github.com/thom1505

Overview
--------

The extension provides a repeatable hydraulic-verification layer for
design, rehabilitation, optimisation and scenario-analysis workflows.

It can:

* apply proposed pipe diameters without modifying the original network;
* apply hydraulic operating scenarios;
* run simulations using ``WNTRSimulator`` or ``EpanetSimulator``;
* assess minimum junction pressure;
* assess maximum absolute pipe velocity;
* audit head-pump flows against the maximum flow represented by their
  pump curves;
* identify critical components and simulation times;
* calculate compliance percentages;
* return an overall hydraulic-feasibility result; and
* produce an audit record describing the verification run.

Overall feasibility
-------------------

A verification is feasible only when:

* the required pressure compliance is achieved;
* the required velocity compliance is achieved; and
* all head pumps pass the pump-curve audit.

A network containing no head pumps does not fail solely because the
pump audit is not applicable.

Basic use
---------

A hydraulic scenario can be created as follows:

.. doctest::

   >>> from wntr.extensions.design_verification import HydraulicScenario
   >>> scenario = HydraulicScenario(
   ...     name="Baseline DD",
   ...     demand_model="DD",
   ...     duration_s=0,
   ...     hydraulic_timestep_s=3600,
   ...     report_timestep_s=3600,
   ... )
   >>> scenario.name
   'Baseline DD'
   >>> scenario.demand_model
   'DD'

The following example assumes that ``wn`` is an existing
``WaterNetworkModel``:

.. code-block:: python

   from wntr.extensions.design_verification import (
       HydraulicScenario,
       PipeDesign,
       run_design_verification,
   )

   scenario = HydraulicScenario(
       name="Peak-demand scenario",
       demand_model="PDD",
       demand_multiplier=1.25,
       duration_s=24 * 3600,
       hydraulic_timestep_s=3600,
       report_timestep_s=3600,
       minimum_pressure_m=0.0,
       required_pressure_m=15.0,
       pressure_exponent=0.5,
   )

   design = PipeDesign(
       name="Candidate pipe design",
       diameters_m={
           "P1": 0.300,
           "P2": 0.250,
       },
   )

   verification, hydraulic_results, audit = run_design_verification(
       wn=wn,
       scenario=scenario,
       design=design,
       simulator="WNTR",
       minimum_pressure_m=15.0,
       maximum_velocity_mps=2.5,
       required_compliance_pct=100.0,
   )

   print(verification.feasible)
   print(verification.pressure_result)
   print(verification.velocity_result)
   print(verification.pump_result)
   print(audit)

Returned information
--------------------

``run_design_verification`` returns three objects.

``verification``
   A structured result containing the pressure assessment, velocity
   assessment, pump-curve audit, simulator error code and overall
   feasibility result.

``hydraulic_results``
   The raw hydraulic simulation results returned by WNTR.

``audit``
   A dictionary recording the applied design, hydraulic scenario,
   simulator, engineering limits, assessed components and pump-curve
   audit.

Pressure assessment
-------------------

Pressure requirements are applied to demand junctions. Reservoirs and
tanks are excluded from the minimum-pressure assessment.

The pressure result reports:

* the minimum simulated pressure;
* the pressure-compliance percentage;
* whether the required compliance was achieved;
* the critical junction; and
* the critical simulation time.

Velocity assessment
-------------------

Velocity requirements are applied to pipes. Pumps and valves are
excluded from the pipe-velocity assessment.

Absolute velocity is used so that reverse flow does not conceal a
velocity-limit exceedance.

The velocity result reports:

* the maximum absolute pipe velocity;
* the velocity-compliance percentage;
* whether the required compliance was achieved;
* the critical pipe; and
* the critical simulation time.

Pump-curve audit
----------------

The pump audit evaluates active head-pump flow observations against the
maximum flow represented by each pump curve.

When pump-speed information is available, the represented curve limit
is adjusted using the simulated or configured operating speed.

A head pump fails the audit when an active flow exceeds its adjusted
curve-domain limit. A head pump that cannot be evaluated does not
silently pass the audit.

The pump audit checks whether simulated flow remains within the
represented curve domain. It does not assess pump efficiency, energy
consumption, cavitation, net positive suction head or manufacturer
selection requirements.

Protection of the original network
----------------------------------

Pipe-design and hydraulic-scenario changes are applied to independent
network copies. The original ``WaterNetworkModel`` supplied to
``run_design_verification`` is therefore preserved.

Limitations
-----------

Verification results depend on the quality and completeness of the
network model, hydraulic options, demand assumptions, pump curves,
controls and simulation convergence.

Pressure and velocity limits are supplied by the user. Default values
are not intended to replace applicable utility standards, engineering
codes or regulatory requirements.

The extension verifies simulated hydraulic behaviour. It does not
certify a design for construction or operation.