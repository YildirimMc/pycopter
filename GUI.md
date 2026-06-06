# PyCopter GUI Contract

This file is the handoff contract for wiring the GUI to the rotor calculation
layer. Any future GUI-facing input or output must be added here with endpoint,
units, range/default, and a short description.

## Calculation Endpoints

Use these Python endpoints from `interface.py` or any future GUI layer.

### Single Rotor Hover

```python
from pycopter import HoverSolver, HoverSolverSettings, OperatingPoint, RotorSpec

rotor = RotorSpec.from_uniform_blade(...)
solver = HoverSolver(settings=HoverSolverSettings(...))
result = solver.solve(rotor, OperatingPoint(...))
loads = result.load_table()
```

For high-fidelity blade geometry, construct `RotorSpec(...)` with a
`list[BladeStation]` instead of `from_uniform_blade(...)`.

### Coaxial Hover

```python
from pycopter import CoaxialSpec, OperatingPoint, solve_coaxial_hover

result = solve_coaxial_hover(
    CoaxialSpec(upper_rotor=upper, lower_rotor=lower, ...),
    OperatingPoint(...),
)
```

### Legacy Compatibility

```python
from pycopter import Rotor

rotor = Rotor(...)
rotor.hover(weight=mass_kg, density=density_kg_m3)
result = rotor.hover_result
```

The legacy endpoint keeps old attributes such as `hover_thrust`,
`hover_power_total`, `ct`, `cp`, and `theta`, but new GUI work should prefer the
typed endpoints above.

## Existing Inputs To Keep

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `airfoil` | `RotorSpec.airfoil`, `BladeStation.airfoil` | text | default `naca0012` | NACA or UIUC airfoil name. Station airfoil overrides global airfoil. |
| `num_blades` | `RotorSpec.num_blades` | count | `1-12`, default `2` for drones | Number of blades on one rotor, not total aircraft blades. |
| `chord` | `RotorSpec.from_uniform_blade(chord_m=...)` | m | `0.001-2.5`, default project/preset value | Uniform chord fallback when no station table is used. |
| `rotor_diam` | `RotorSpec.rotor_diameter_m` | m | `0.05-30`, default project/preset value | Rotor disk diameter. |
| `tip_speed_mach` | `RotorSpec.tip_speed_mach` | Mach | `0.02-0.9`, default preset value | Legacy speed input; prefer `headspeed_rpm` for drones. |
| `washout` | `from_uniform_blade(washout_deg=...)` | deg | `-30 to 30`, default preset value | Linear tip twist fallback when no station table is used. |
| `root_cutout` | `RotorSpec.root_cutout_ratio` | r/R | `0-0.95`, default `0.02` | Inner radius excluded from blade/disk loading. |
| `gross` | `OperatingPoint.gross_mass_kg` | kg | `0.01-100000`, default preset value | Aircraft mass used to derive target hover thrust. |
| `density` | `OperatingPoint.density_kg_m3` | kg/m3 | `0.01-2.0`, default `1.225` | Air density. |
| `transmission_loss` | GUI post-process only | fraction | `0-0.5`, default preset value | Apply outside solver when converting shaft power to motor/engine input power. |
| `velocity` | legacy `Rotor.forward_flight(...)` | km/hr in GUI | `0-9999`, default preset value | Forward flight remains low-confidence and secondary. |
| `fpa` | legacy `Rotor.forward_flight(..., flat_plate_area=...)` | m2 | `0-9999`, default preset value | Equivalent flat plate area for legacy forward-flight parasite power. |

## New Inputs

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `rotor_system_type` | GUI branch to `HoverSolver.solve` or `solve_coaxial_hover` | enum | `single`, `coaxial`; default `single` | Selects single rotor or stacked coaxial calculation. |
| `headspeed_rpm` | `RotorSpec.headspeed_rpm` | rpm | `100-100000`; default derived from tip Mach | Direct rotor speed input. Preferred for sub-10 kg electric drones. |
| `headspeed_input_mode` | GUI branch | enum | `rpm`, `tip_mach`; default `rpm` | Choose whether `headspeed_rpm` or `tip_speed_mach` controls omega. |
| `kinematic_viscosity_m2_s` | `OperatingPoint.kinematic_viscosity_m2_s` | m2/s | `1.0e-5-2.5e-5`; default `1.5e-5` | Used for local Reynolds number at each blade station. |
| `blade_element_count` | `HoverSolverSettings.blade_element_count` | count | `20-200`; default `60` | Radial resolution of the BEMT integration. |
| `trim_mode` | `OperatingPoint.trim_mode` | enum | `target_thrust`, `fixed_collective`; default `target_thrust` | Solve collective for target thrust or evaluate a fixed pitch. |
| `collective_pitch_deg` | `OperatingPoint.collective_pitch_deg` | deg | `-5 to 25`; default `8` | Used in fixed-collective mode and equal-collective coaxial studies. |
| `max_collective_deg` | `HoverSolverSettings.max_collective_deg` | deg | `0-35`; default `20` | Upper collective search bound for target-thrust trim. |
| `min_collective_deg` | `HoverSolverSettings.min_collective_deg` | deg | `-10 to 10`; default `-5` | Lower collective search bound for target-thrust trim. |
| `polar_alpha_min_deg` | `XfoilPolarProvider.alpha_min_deg` | deg | `-20 to 5`; default `-10` | Lower AoA bound for generated XFOIL polar tables. |
| `polar_alpha_max_deg` | `XfoilPolarProvider.alpha_max_deg` | deg | `10-15`; default `15` | Upper AoA bound for generated XFOIL polar tables. Requests above 15 deg are capped because XFOIL often fails there and this is an estimator. |
| `tip_loss_model` | `HoverSolverSettings.tip_loss_model` | enum | `prandtl`, `none`; default `prandtl` | Enables Prandtl finite-blade tip loss. |
| `root_loss_model` | `HoverSolverSettings.root_loss_model` | enum | `prandtl`, `none`; default `prandtl` | Enables Prandtl-style root loss near blade cutout. |
| `induced_power_factor` | `HoverSolverSettings.induced_power_factor` | factor | `1.0-1.3`; default `1.05` | Nonideal induced-power correction; set `1.0` for pure BEMT. |

## Blade Station Table Inputs

These fields define `BladeStation(...)` rows. The table must be sorted by
`station_r_over_R` and include at least two rows.

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `station_r_over_R` | `BladeStation.r_over_R` | r/R | `0.02-1.0`; monotonic | Radial station location. |
| `station_chord_m` | `BladeStation.chord_m` | m | `0.001-2.5`; default current chord | Local blade chord. |
| `station_twist_deg` | `BladeStation.twist_deg` | deg | `-30 to 30`; default linear washout | Local built-in twist relative to collective. |
| `station_airfoil` | `BladeStation.airfoil` | text | default global airfoil | Optional local airfoil for blended blades. |
| `station_pitch_axis_frac` | `BladeStation.pitch_axis_frac` | chord fraction | `0-1`; default `0.25` | Pitch axis for aerodynamic pitch moment about control linkage. |

## Coaxial Inputs

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `coaxial_spacing_ratio` | `CoaxialSpec.spacing_ratio` | z/R | `0.05-1.5`; default `0.25` | Vertical spacing between rotor disks divided by upper rotor radius. |
| `coaxial_trim_mode` | `CoaxialSpec.trim_mode` | enum | `equal_thrust`, `equal_collective`; default `equal_thrust` | Equal thrust trims both rotors to half target thrust; equal collective evaluates fixed pitch. |
| `lower_collective_offset_deg` | `CoaxialSpec.lower_collective_offset_deg` | deg | `-10 to 10`; default `0` | Lower rotor collective offset for equal-collective studies. |
| `lower_rotor_scale` | GUI convenience before `RotorSpec` creation | factor | `0.5-1.5`; default `1.0` | Optional lower rotor diameter/chord scale if not entering a separate lower station table. |

## Hover Outputs

All single-rotor hover outputs are on `HoverResult`.

| Output key | Endpoint | Units | Description |
|---|---|---:|---|
| `collective_pitch_deg` | `HoverResult.collective_pitch_deg` | deg | Collective used or solved for the hover result. |
| `total_thrust_N` | `HoverResult.total_thrust_N` | N | Integrated rotor thrust. |
| `per_blade_thrust_N` | `HoverResult.per_blade_thrust_N` | N/blade | Total thrust carried by one blade. |
| `total_torque_Nm` | `HoverResult.total_torque_Nm` | N*m | Shaft torque consistent with corrected shaft power. |
| `per_blade_torque_Nm` | `HoverResult.per_blade_torque_Nm` | N*m/blade | Shaft torque contribution per blade. |
| `power_W` | `HoverResult.power_W` | W | Shaft power from induced plus profile components. |
| `induced_power_W` | `HoverResult.induced_power_W` | W | Induced component after `induced_power_factor`. |
| `profile_power_W` | `HoverResult.profile_power_W` | W | Profile drag power from section Cd integration. |
| `ideal_power_W` | `HoverResult.ideal_power_W` | W | Ideal actuator-disk hover power. |
| `figure_of_merit` | `HoverResult.figure_of_merit` | ratio | `ideal_power_W / power_W`. |
| `mean_induced_velocity_m_s` | `HoverResult.mean_induced_velocity_m_s` | m/s | Thrust-weighted induced velocity. |
| `ct` | `HoverResult.ct` | ratio | Rotor thrust coefficient. |
| `cp` | `HoverResult.cp` | ratio | Rotor power coefficient. |
| `solidity` | `HoverResult.solidity` | ratio | Blade solidity from radial chord distribution. |
| `mean_loss_factor` | `HoverResult.mean_loss_factor` | ratio | Thrust-weighted Prandtl loss factor. |
| `root_flap_bending_moment_Nm_per_blade` | `HoverResult.root_flap_bending_moment_Nm_per_blade` | N*m/blade | Root bending estimate from radial thrust loads. |
| `root_lag_moment_Nm_per_blade` | `HoverResult.root_lag_moment_Nm_per_blade` | N*m/blade | In-plane/lag moment estimate from tangential loads. |
| `aerodynamic_pitching_moment_Nm_per_blade` | `HoverResult.aerodynamic_pitching_moment_Nm_per_blade` | N*m/blade | Section Cm plus pitch-axis offset integrated along one blade. |

## Blade Load Table Outputs

Use `HoverResult.load_table()` or `HoverResult.element_loads`. Each row is one
radial element and is per blade.

| Output key | Units | Description |
|---|---:|---|
| `r_m`, `r_over_R`, `dr_m` | m, ratio, m | Radial element location and width. |
| `chord_m`, `twist_deg`, `collective_deg` | m, deg, deg | Local geometry and collective. |
| `phi_deg`, `alpha_deg` | deg | Inflow angle and local effective angle of attack. |
| `reynolds`, `mach` | ratio | Local section Reynolds and Mach values. |
| `cl`, `cd`, `cm` | ratio | Local section coefficients from polar provider. |
| `alpha_clamped` | bool | True when requested AoA was outside the generated polar table and coefficients were clamped. |
| `loss_factor` | ratio | Combined root/tip loss factor. |
| `induced_velocity_m_s` | m/s | Self-induced element velocity. |
| `external_axial_velocity_m_s` | m/s | Axial inflow from upper rotor for coaxial lower rotor. |
| `dL_N`, `dD_N`, `dT_N`, `dQ_Nm`, `dP_W` | N, N, N, N*m, W | Integrated element forces, torque, and power per blade. |
| `normal_force_N_per_m`, `tangential_force_N_per_m` | N/m | Distributed normal and in-plane loads. |
| `pitch_moment_Nm` | N*m | Element pitch moment about `station_pitch_axis_frac`. |

## Coaxial Outputs

All coaxial outputs are on `CoaxialHoverResult`.

| Output key | Endpoint | Units | Description |
|---|---|---:|---|
| `upper` | `CoaxialHoverResult.upper` | `HoverResult` | Upper rotor hover result and blade loads. |
| `lower` | `CoaxialHoverResult.lower` | `HoverResult` | Lower rotor hover result and blade loads with upper-wake inflow. |
| `isolated_upper` | `CoaxialHoverResult.isolated_upper` | `HoverResult` | Upper rotor reference without coaxial interference. |
| `isolated_lower` | `CoaxialHoverResult.isolated_lower` | `HoverResult` | Lower rotor reference without upper wake. |
| `total_thrust_N` | `CoaxialHoverResult.total_thrust_N` | N | Upper plus lower thrust. |
| `total_power_W` | `CoaxialHoverResult.total_power_W` | W | Upper plus lower shaft power. |
| `interference_power_delta_W` | `CoaxialHoverResult.interference_power_delta_W` | W | Coaxial power minus isolated pair power. |
| `interference_loss_ratio` | `CoaxialHoverResult.interference_loss_ratio` | ratio | Interference delta divided by isolated pair power. |
| `lower_external_velocity_mean_m_s` | `CoaxialHoverResult.lower_external_velocity_mean_m_s` | m/s | Mean upper-wake velocity applied to lower elements. |
| `wake_radius_m` | `CoaxialHoverResult.wake_radius_m` | m | Contracted upper wake radius at the lower rotor plane. |
| `wake_velocity_m_s` | `CoaxialHoverResult.wake_velocity_m_s` | m/s | Upper wake axial velocity at the lower rotor plane. |

## GUI Change Requests

- Stop using total blade count as a fake multi-rotor/coaxial model. `num_blades`
  is now blades per rotor; use `rotor_system_type='coaxial'` for coaxial cases.
- Add an explicit radial blade station table. The old chord/washout fields can
  remain as a "uniform blade" quick-entry mode.
- Prefer `headspeed_rpm` for drone workflows. Keep `tip_speed_mach` only as a
  legacy/large-helicopter input mode.
- Separate fuel/electric range UI from rotor hover physics. The new solver
  reports shaft power only; motor, ESC, battery, and fuel models should consume
  `power_W` downstream.
- Display blade load tables and root moments as first-class outputs. These are
  the new product focus and should not be hidden in debug text.

## Theory Notes

- The hover solver uses blade-element/momentum theory with full inflow-angle
  trigonometry, local `alpha = theta - phi`, and local `Cl/Cd(M, Re, alpha)`.
  This follows the analysis structure described by NASA/Army hover work on
  laminar rotor performance: https://ntrs.nasa.gov/api/citations/20170005472/downloads/20170005472.pdf
- Coaxial v1 applies a momentum-theory upper wake to the lower rotor. NASA
  NDARC theory notes that in hover the lower rotor acts in the contracted wake
  of the upper rotor, and uses separate upper/lower induced-power treatment:
  https://rotorcraft.arc.nasa.gov/Publications/files/NDARCTheory_v1_6_938.pdf
- The coaxial model is still an estimator, not a free-wake or CFD model. It is
  scientifically stronger than the old total-blade-count shortcut, but should be
  calibrated against measured rotor data before design-critical use.
