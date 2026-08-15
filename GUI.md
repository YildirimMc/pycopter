# PyCopter GUI Contract

This file is the handoff contract for wiring the GUI to the rotor calculation
layer. Any future GUI-facing input or output must be added here with endpoint,
units, range/default, and a short description.

## Calculation Endpoints

Use these Python endpoints from `interface.py` or any future GUI layer.

### Single Rotor Hover

```python
from pycopter import HoverSolver, HoverSolverSettings, OperatingPoint, RotorSpec
from pycopter.polars import XfoilPolarProvider

rotor = RotorSpec.from_uniform_blade(...)
polar_provider = XfoilPolarProvider(...)
solver = HoverSolver(
    polar_provider=polar_provider,
    settings=HoverSolverSettings(...),
)
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
| `airfoil` | `RotorSpec.airfoil`, `BladeStation.airfoil` | text | default `naca0012` | NACA or UIUC airfoil name. Blank station airfoil cells inherit this global airfoil; nonblank station cells intentionally override it. |
| `num_blades` | `RotorSpec.num_blades` | count | `1-12`, default `2` for drones | Number of blades on one rotor, not total aircraft blades. |
| `chord` | `RotorSpec.from_uniform_blade(chord_m=...)` | m | `0.001-2.5`, default project/preset value | Uniform chord fallback when no station table is used. |
| `rotor_diam` | `RotorSpec.rotor_diameter_m` | m | `0.05-30`, default project/preset value | Rotor disk diameter. |
| `tip_speed_mach` | `RotorSpec.tip_speed_mach` | Mach | calculated from `headspeed_rpm` | Report-only value in the Web UI. Legacy configs may load it to derive RPM, but drone workflows enter headspeed directly. |
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
| `headspeed_rpm` | upper/reference `RotorSpec.headspeed_rpm` | rpm | `100-100000`; default derived from tip Mach | Direct rotor speed input. In coaxial mode this is the upper/reference rotor RPM; the lower rotor can follow a separate gear ratio. |
| `root_twist_deg` | `BladeStation.twist_deg` via uniform quick-entry | deg | `-30 to 30`; default `10` | Built-in pitch/twist at the first blade station for uniform quick-entry. |
| `tip_twist_deg` | `BladeStation.twist_deg` via uniform quick-entry | deg | `-30 to 30`; default `1` | Built-in pitch/twist at the blade tip for uniform quick-entry. |
| `headspeed_input_mode` | GUI branch | enum | `rpm`; default `rpm` | Compatibility key retained for saved configs; the Web UI uses RPM input only. |
| `kinematic_viscosity_m2_s` | `OperatingPoint.kinematic_viscosity_m2_s` | m2/s | `1.0e-5-2.5e-5`; default `1.5e-5` | Used for local Reynolds number at each blade station. |
| `blade_element_count` | `HoverSolverSettings.blade_element_count` | count | `20-200`; default `60` | Radial resolution of the BEMT integration. |
| `trim_mode` | `OperatingPoint.trim_mode` | enum | `target_thrust`, `fixed_collective`; default `target_thrust` | Solve collective for target thrust or evaluate a fixed pitch. |
| `collective_pitch_deg` | `OperatingPoint.collective_pitch_deg` | deg | `-5 to 25`; default `8` | Used in fixed-collective mode and equal-collective coaxial studies. |
| `max_collective_deg` | `HoverSolverSettings.max_collective_deg` | deg | `0-35`; default `15` | Upper collective search bound for target-thrust trim. |
| `min_collective_deg` | `HoverSolverSettings.min_collective_deg` | deg | `-10 to 10`; default `0` | Lower collective search bound for target-thrust trim. |
| `new_polars` | `XfoilPolarProvider.new_polar` | bool | default `true` | Run XFOIL for missing polar bins. When false, cached polars are used only and XFOIL is never launched, even if other settings request unavailable alpha/Re/Mach data. |
| `polar_alpha_min_deg` | `XfoilPolarProvider.alpha_min_deg` | deg | `-20 to 5`; default `-3` | Lower AoA bound for generated XFOIL polar tables. |
| `polar_alpha_max_deg` | `XfoilPolarProvider.alpha_max_deg` | deg | `10-18`; default `18` | Upper AoA bound for generated XFOIL polar tables. Requests above 18 deg are capped because XFOIL often fails there and this is an estimator. |
| `xfoil_parallel_workers` | `XfoilPolarProvider.parallel_workers` | count | `2-16`; default `8` | Maximum MPI workers used to generate independent missing XFOIL polar bins. Use `8` for the current hover workflow unless explicitly debugging. |
| `xfoil_parallel_backend` | `XfoilPolarProvider.parallel_backend` | enum | `mpi`, `serial`; default `mpi` | `mpi` requires a working MPI runtime and runs missing XFOIL polar bins through `mpi4py.futures`. `serial` is only for explicit non-parallel debugging. |
| `xfoil_cache_directory` | `XfoilPolarProvider.cache_directory` | path | optional; default `data/XFOIL6.99/tmp/gui-cache` | Advanced override for generated polar files. The GUI default is a stable ignored cache so non-XFOIL setting changes can reuse existing polar bins. |
| `tip_loss_model` | `HoverSolverSettings.tip_loss_model` | enum | `prandtl`, `none`; default `prandtl` | Enables Prandtl finite-blade tip loss. |
| `root_loss_model` | `HoverSolverSettings.root_loss_model` | enum | `prandtl`, `none`; default `prandtl` | Enables Prandtl-style root loss near blade cutout. |
| `induced_power_factor` | `HoverSolverSettings.induced_power_factor` | factor | `1.0-1.3`; default `1.05` | Nonideal induced-power correction; set `1.0` for pure BEMT. |
| `geometry_mode` | GUI branch inside `build_rotor_spec` | enum | `uniform`, `station_table`; default `uniform` | Whether blade geometry comes from the root/tip quick-entry fields or from the station table. |
| `station_count` | `station_rows_from_uniform(station_count=...)` | count | `2-40`; default `4` | Number of control points generated by the uniform quick-entry mode. |
| `station_pitch_axis_frac` | `BladeStation.pitch_axis_frac` default | chord fraction | `0-1`; default `0.25` | Moment reference axis used by uniform quick-entry and by new station rows. Per-row values still override it. |
| `speed_of_sound_m_s` | `OperatingPoint.speed_of_sound_m_s`, `RotorSpec.speed_of_sound_m_s` | m/s | `250-400`; default `343` | Sets both the local Mach used in polar lookup and the reported tip Mach. |
| `altitude_m` | GUI display convenience | m | `-500 to 11000`; default `0` | Never reaches the solver on its own. `Apply ISA` derives `density`, `kinematic_viscosity_m2_s` and `speed_of_sound_m_s` from it and writes them into those fields. |
| `temperature_C` | GUI display convenience | degC | `-60 to 60`; default `15` | Outside air temperature at `altitude_m`, used by the same `Apply ISA` action. Standard-day pressure with a non-standard temperature is the usual ISA + delta T construction. |
| `flight_mode` | GUI branch | enum | `hover`, `forward_flight`; default `hover` | Selects whether the legacy single-rotor forward-flight estimate is offered alongside hover. |
| `element_spacing` | `HoverSolverSettings.element_spacing` | enum | `uniform`, `cosine`; default `uniform` | Radial distribution of blade elements. `cosine` clusters elements at the root cutout and the tip without changing the element count or the span. |
| `thrust_tolerance` | `HoverSolverSettings.thrust_tolerance` | fraction | `1e-6 to 1e-1`; default `1e-3` | Relative thrust error accepted by the collective trim loop. |
| `max_trim_iterations` | `HoverSolverSettings.max_trim_iterations` | count | `5-400`; default `80` | Iteration limit of the collective trim loop. |
| `polar_alpha_step_deg` | `XfoilPolarProvider.alpha_step_deg` | deg | `0.05-5.0`; default `1.0` | Alpha resolution of a generated polar. Finer steps cost XFOIL time and produce a different cached table, so they use a separate cache entry. |
| `polar_reynolds_bin` | `XfoilPolarProvider.reynolds_bin` | Re | `1000-1000000`; default `100000` | Width of a Reynolds bin. A request is rounded into the nearest bin instead of generating a polar per element. |
| `polar_mach_bin` | `XfoilPolarProvider.mach_bin` | Mach | `0.01-0.5`; default `0.1` | Width of a Mach bin, with the same purpose as `polar_reynolds_bin`. |
| `polar_n_crit` | `XfoilPolarProvider.n_crit` | ratio | `1-14`; default `9` | XFOIL e^n transition amplification factor. It changes the generated table, so it is part of the cache key. |
| `xfoil_max_iterations` | `XfoilPolarProvider.max_iterations` | count | `20-2000`; default `400` | XFOIL viscous iteration limit per alpha point. |
| `xfoil_timeout_s` | `XfoilPolarProvider.timeout` | s | `5-600`; default `60` | Per-polar XFOIL process timeout. |

## Propulsion Estimation Inputs

These fields are post-processing inputs. They must not change rotor hover
physics, and the UI must display only the selected propulsion model's derived
outputs.

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `propulsion_model` | GUI branch | enum | `electric`, `fossil`; default `electric` | Selects electric battery/motor estimates or fossil-fuel engine estimates. |
| `battery_capacity_Wh` | GUI post-process only | Wh | `>0`, default `100` | Total battery energy for small electric aircraft estimates. |
| `battery_usable_fraction` | GUI post-process only | fraction | `0.05-1.0`, default `0.8` | Fraction of nominal battery energy assumed usable. |
| `motor_efficiency` | GUI post-process only | fraction | `0.01-1.0`, default `0.85` | Motor efficiency used to convert shaft power to electrical input power. |
| `esc_efficiency` | GUI post-process only | fraction | `0.01-1.0`, default `0.95` | ESC efficiency used with motor efficiency for electric input power. |
| `fuel_capacity_kg` | GUI post-process only | kg | `>=0`, default `1.5` | Fuel mass for fossil-fuel endurance estimates. |
| `specific_fuel_consumption_kg_per_kWh` | GUI post-process only | kg/kWh | `>0`, default `0.3` | Fuel consumption per engine input energy. |

## Session And UI Preferences

These live behind `Preferences...` in the app menu. They change how much work
the UI asks for and how much of it it keeps, and never reach the solver
equations. They round-trip through save/load like any other input.

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `sweep_points` | `sweep_interference_loss(z_over_R_values=...)` length | count | `3-200`; default `25` | Number of points in the coaxial spacing sweep. |
| `plot_render_dpi` | `pn.pane.Matplotlib.dpi` | dpi | `72-300`; default `144` | Rasterization density of the figure, at 1.5x the CSS pixel size by default. |
| `log_scrollback_lines` | GUI log buffer and xterm scrollback | count | `100-20000`; default `1000` | How many log lines the session keeps. |
| `max_session_runs` | `RunStore.max_runs` | count | `10-2000`; default `200` | How many run records a session keeps before the oldest is dropped. |

## Blade Station Table Inputs

These fields define `BladeStation(...)` rows. The table must be sorted by
`station_r_over_R` and include at least two rows.

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `station_r_over_R` | `BladeStation.r_over_R` | r/R | `0.02-1.0`; monotonic | Radial station location. |
| `station_chord_m` | `BladeStation.chord_m` | m | `0.001-2.5`; default current chord | Local blade chord. |
| `station_twist_deg` | `BladeStation.twist_deg` | deg | `-30 to 30`; default linear washout | Local built-in twist relative to collective. |
| `station_airfoil` | `BladeStation.airfoil` | text | blank means global airfoil | Optional local airfoil for blended blades. |
| `station_pitch_axis_frac` | `BladeStation.pitch_axis_frac` | chord fraction | `0-1`; default `0.25` | Moment reference axis as a chord fraction; this is not blade pitch or twist. |

## Coaxial Inputs

| GUI key | Endpoint | Units | Range / default | Description |
|---|---|---:|---|---|
| `coaxial_spacing_ratio` | `CoaxialSpec.spacing_ratio` | z/R | `0.05-1.5`; default `0.25` | Vertical spacing between rotor disks divided by upper rotor radius. |
| `coaxial_trim_mode` | `CoaxialSpec.trim_mode` | enum | `torque_balance`, `equal_thrust`, `equal_collective`; default `torque_balance` | `torque_balance` solves total aircraft thrust and zero net aircraft yaw torque. Equal thrust and equal collective are retained as comparison/debug modes. |
| `lower_collective_offset_deg` | `CoaxialSpec.lower_collective_offset_deg` | deg | `-10 to 10`; default `0` | Lower rotor collective offset for equal-collective studies. |
| `lower_rotor_speed_ratio` | lower `RotorSpec.headspeed_rpm / upper RotorSpec.headspeed_rpm` | ratio | `0.25-3.0`; default `1.0` | Lower rotor gear ratio relative to the upper/reference RPM. A value of `1.05` means the lower rotor runs 5% faster; all coaxial trim objectives are solved with this lower rotor RPM. |
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
| `aircraft_yaw_torque_Nm` | `HoverResult.aircraft_yaw_torque_Nm` | N*m | Signed aircraft reaction yaw torque. Positive is counterclockwise/left yaw viewed from above; negative is clockwise/right yaw. |
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
| `clamped_element_count` | `HoverResult.clamped_element_count` | count | Elements whose requested alpha fell outside the loaded polar table. Drives the amber warning bar; the GUI must not parse log text for it. |
| `clamped_alpha_range_deg` | `HoverResult.clamped_alpha_range_deg` | deg | Requested alpha range of those clamped elements, or `None` when there are none. |
| `substituted_polar_bins` | `HoverResult.substituted_polar_bins` | text | Polar bins that answered a request outside the provider's supported Reynolds/Mach range, so the returned polar describes a different flow condition. Ordinary Re/Mach binning is not listed here. |
| `warnings` | `HoverResult.warnings` | text | Structured warning lines built from the two fields above. `CoaxialHoverResult.warnings` labels each one with the rotor it came from. |
| `aerodynamic_pitching_moment_Nm_per_blade` | `HoverResult.aerodynamic_pitching_moment_Nm_per_blade` | N*m/blade | Section Cm plus pitch-axis offset integrated along one blade. Use this as aerodynamic hinge-moment input for pitch-link/servo sizing, then add linkage ratio, friction, inertia, transients, and safety factor. |

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
| `net_aircraft_yaw_torque_Nm` | `CoaxialHoverResult.net_aircraft_yaw_torque_Nm` | N*m | Sum of upper and lower signed aircraft yaw reaction torques. Positive is counterclockwise/left yaw viewed from above; negative is clockwise/right yaw. |
| `interference_power_delta_W` | `CoaxialHoverResult.interference_power_delta_W` | W | Coaxial power minus isolated pair power. |
| `interference_loss_ratio` | `CoaxialHoverResult.interference_loss_ratio` | ratio | Interference delta divided by isolated pair power. |
| `lower_external_velocity_mean_m_s` | `CoaxialHoverResult.lower_external_velocity_mean_m_s` | m/s | Mean upper-wake velocity applied to lower elements. |
| `wake_radius_m` | `CoaxialHoverResult.wake_radius_m` | m | Contracted upper wake radius at the lower rotor plane. |
| `wake_velocity_m_s` | `CoaxialHoverResult.wake_velocity_m_s` | m/s | Upper wake axial velocity at the lower rotor plane. |

## Run Records And Sweeps

A calculation is a persisted record, not a mutable last result.

```python
from gui.runs import RunRecord, RunStore

store = RunStore(max_runs=200)
record = store.create("coax z/R 0.25", {**config, "station_rows": rows}, status="running")
store.update(record.run_id, status="done", result=result, warnings=result.warnings)
store.set_baseline(record.run_id)
```

- Records are append-only within a browser session and are never written to
  disk on their own. `Export session runs` serializes `RunStore.to_payload()`.
- `inputs` is the full GUI key snapshot plus `station_rows`, copied on create so
  later inspector edits cannot rewrite history. It is sufficient to re-solve a
  run and, through `gui.runs.diff_inputs`, to diff two runs field by field.
- Deltas are computed in the GUI from `result`, never stored. One record may be
  flagged baseline; deleting it moves the flag.
- A sweep is one parent record whose points are children linked by `parent_id`.
  Deleting the parent deletes its points.

```python
from pycopter import sweep_interference_loss

sweep = sweep_interference_loss(
    coaxial_spec,
    operating_point,
    z_over_R_values,
    polar_provider=provider,
    settings=settings,
    progress=lambda index, total, z_over_R, point: ...,
    cancel=lambda: cancel_event.is_set(),
)
```

- `progress` is called after each point with the 1-based index, the total, the
  spacing, and the `InterferenceSweepPoint` just completed, so a caller can draw
  a partial curve while the sweep runs.
- `cancel` is polled before each point. A cancelled sweep still returns every
  point solved so far, with `InterferenceSweep.cancelled` set.
- A point that fails to converge is recorded with its error rather than
  aborting the sweep, so one unreachable spacing does not discard the curve.
- The Web UI runs sweeps on a worker thread and drains progress through a queue
  on a Panel periodic callback, so the UI stays responsive and CANCEL works.
  Without a Panel event loop it falls back to solving on the calling thread.

## Polar Bin Provenance

`XfoilPolarProvider.polar_bin_report(conditions=None)` returns one dict per
airfoil/Re/Mach bin with `airfoil`, `reynolds`, `mach`, `alpha_min_deg`,
`alpha_max_deg`, `source`, `substituted`, `status`, `lookups`, `cache_file` and
`label`. Passing `conditions` narrows the report to the bins those flow
conditions resolve to, which is how one calculation reports only its own bins
from a session-wide provider.

- `source` is `cache` when the bin was read from an existing cache file and
  `generated` when XFOIL produced it in this session.
- `substituted` is true only when the requested Reynolds or Mach fell outside
  the provider's supported range and was clamped, so the returned polar
  describes a different flow condition than the blade element asked for.
  Rounding into a Re/Mach bin is the documented binning scheme, not a
  substitution.
- `status` collapses the two into the single value the POLARS tab shows.
- `cache_bin_count()` and `cache_file_count()` back the XFOIL cache readouts in
  the command bar and on the XFOIL tab.

## GUI Change Requests

- Stop using total blade count as a fake multi-rotor/coaxial model. `num_blades`
  is now blades per rotor; use `rotor_system_type='coaxial'` for coaxial cases.
- Add an explicit radial blade station table. The old chord/washout fields can
  remain as a "uniform blade" quick-entry mode.
- Prefer `headspeed_rpm` for drone workflows. Keep `tip_speed_mach` only as a
  calculated/report-only value.
- Replace the old `washout` quick-entry field with explicit root and tip twist
  fields. Legacy saved configs may map `washout` to a tip twist for loading.
- Separate fuel/electric range UI from rotor hover physics. The new solver
  reports shaft power only; motor, ESC, battery, and fuel models should consume
  `power_W` downstream.
- Keep fossil-fuel and electric estimates visually separated in the GUI. Output
  summaries and propulsion-specific plots must show only the active propulsion
  model.
- Display blade load tables and root moments as first-class outputs. These are
  the new product focus and should not be hidden in debug text.
- Wire XFOIL settings through `XfoilPolarProvider(...)`. Generated polars are
  calculation scratch data: keep the default temporary cache or an ignored
  custom path, and never track them in Git.
- The root-level `webui.py` launcher starts the local Panel Web UI. Closing the
  owning terminal window or pressing Ctrl+C stops the server.
- Summary and Blade Element Loads tables are display outputs. They should allow
  row/text selection and clipboard copying, but not cell editing.
- Blade Element Loads can exceed the visible width because it exposes every
  per-element flow and load term. The table container must horizontally scroll
  rather than clipping columns, with `r_over_R` frozen as the row key.
- Export controls stay reachable at all times, now in the app menu behind the
  PYCOPTER brand: plot PNG, summary CSV, blade loads CSV, and session runs. If
  the requested artifact has not been generated yet, the GUI logs a warning in
  the output field instead of failing silently.
- `Interference Loss vs Spacing` is a coaxial-only plot. It reuses the current
  rotor, airfoil, XFOIL/cache, solver, propulsion-independent hover, and coaxial
  trim settings while sweeping `coaxial_spacing_ratio` linearly from `z/R=0.05`
  to `z/R=1.50`.

## Layout And Sizing Contract

The dashboard is a fixed-height app shell that fills the browser viewport. Top
to bottom: a pinned command bar, an optional warning bar, the middle band, a
drag rule, and a pinned log.

| Region | Size | Behaviour |
|---|---|---|
| Command bar | 34 px | pinned |
| Warning bar | 28 px | pinned, only while the active run has warnings |
| Icon rail | 60 px wide | fixed; collapses the inspector |
| Inspector | 320 px wide | fixed, scrolls internally |
| Result canvas | flex, min 656 px | takes all remaining width |
| Run rail | 384 px wide | fixed, scrolls internally |
| Drag rule | 7 px | `ns-resize` |
| Log | 30 px min, resizable | pinned |

- Switching inspector tabs, expanding the log, and collapsing the inspector must
  not change the page height or any other region's size. Each region absorbs its
  own overflow.
- The inspector is tab-gated: all six panels stay built and only the active one
  is visible, so every field keeps its value and round-trips through save/load
  regardless of which tab is showing.
- Collapsing the inspector gives its 320 px to the canvas and nothing else
  moves.
- The output log sits under a drag grip and can be pulled upwards, taking its
  extra height from the result area. Its start height is also its minimum, and
  it cannot grow past the point where the result area would fall below
  `RESULT_MIN_HEIGHT`. Double-clicking the grip restores the minimum, and the
  `EXPAND`/`COLLAPSE` button toggles between the minimum and 196 px.
- Dragging updates inline heights so the layout reflows every frame, and
  commits the final height to Python on release. The terminal is refitted
  explicitly through the Panel terminal view, because Bokeh does not run a
  layout pass for a plain height change and xterm would otherwise keep its old
  row count inside a taller box.
- The 60 + 320 + 384 fixed regions plus the 656 px minimum canvas are the
  ~1420 px below which the body scrolls horizontally instead of clipping. The
  canvas takes all remaining width, so a larger monitor yields a larger plot
  rather than empty margin.
- The plot picker, `GENERATE`, and the overlay-compare checkbox sit directly
  above the result tabs, with the plot they drive, not in the inspector.
- Panel renders every component into its own shadow root. A CSS rule written as
  `.container .bk-input` never reaches the widget, because the container is
  outside the boundary. Rules that touch a widget's internals are passed to that
  widget as its own `stylesheets` entry; rules that style a host element, or
  that stay inside one component, live in `RAW_CSS`. The same boundary is why
  browser-side helpers walk shadow roots instead of calling
  `document.querySelector`.
- Matplotlib figures are rendered at the measured on-screen size of the plot
  frame, not at a fixed `figsize`. `ResultAreaProbe` reports that size from the
  browser because Panel cannot; figures rasterize at 1.5x the CSS pixel size so
  they stay sharp on high-density displays while axis text keeps its true size.
- Window resizing does not regenerate sweep plots, because each one re-runs the
  hover solver many times. The existing image is kept inside the frame by CSS
  until the next `Generate Plot` or hover calculation redraws it.

## Plot Axis Rules

`PycopterWebApp._axis_layout` decides the arrangement from the measured data,
so the same plot adapts to a different rotor design. Three rules, in order:

1. **Same scale, same axis.** Two quantities whose magnitudes are within
   `SHARED_AXIS_MAX_RATIO` (5) share one y-axis, so the smaller still spans at
   least a fifth of it. Induced velocity and loss factor differ by 3.6x and
   share an axis.
2. **Different scale, second axis.** Beyond that ratio a quantity takes the
   right-hand axis of the same panel. Element torque is 3% of element thrust,
   and Cd is 3% of Cl, so both are twinned; on a shared axis they would read as
   a flat line at zero.
3. **A third scale gets its own panel.** A quantity that fits neither axis
   moves to a stacked panel sharing the x-axis, up to `MAX_PLOT_PANELS`. Do not
   reintroduce a third y-axis on an offset spine: it is hard to read and was
   removed for that reason.

Two further constraints:

- **Never put identically shaped curves on opposing autoscaled axes.** Both
  axes stretch their series to fill the panel, so the curves land exactly on
  top of each other and one silently hides the other. `_shapes_coincide`
  normalises both series to 0..1 and treats a maximum gap below
  `COINCIDENT_SHAPE_GAP` as coincident; such a pair is split across panels.
  Reynolds and Mach are the live example, both linear in radius with a measured
  gap of 0.0000.
- **Series that share a unit and scale stay together.** `Forward Flight Powers
  vs Velocity` deliberately keeps induced, profile, parasite, and total power
  on one axis because their relative size is the point of the plot.

Presentation follows from the layout: colour identifies the quantity, line
style identifies the rotor, and opacity plus a dash pattern identify an overlaid
run, so a coaxial case keeps the same coding as a single rotor. An axis carrying
exactly one quantity is tinted to match its curve; a shared axis stays neutral
and lists both labels. Twin axes are created through `_twin_axis` so
`_finish_plot` can colour the matching spine and skip the duplicate grid, and
each panel gets one combined legend.

`overlay selected runs` draws every checked run in the rail on the current
per-element plot. Overlays are drawn first, wider and translucent, under the
active run, so an overlay whose shape coincides with the primary series reads as
a halo around it instead of silently hiding it. Overlay data joins the same
`_axis_layout` measurement, so a compared run cannot fall off a scale that was
chosen without it.

The coaxial `SHOWING` toggle chooses which `HoverResult` the canvas, summary and
loads read from: `UPPER`, `LOWER`, `TOTAL` (both rotors), or `Δ ISOLATED`
(each rotor minus its isolated reference, element by element). All three views
follow the same choice so they never disagree.

## XFOIL Cache And Parallelism

- `HoverSolver` now asks compatible polar providers to prefetch radial
  airfoil/Re/Mach bins before element iterations. For `XfoilPolarProvider`,
  these missing bins are independent XFOIL runs.
- With `xfoil_parallel_backend='mpi'` and `xfoil_parallel_workers=8`, the
  provider uses `mpi4py.futures.MPIPoolExecutor(max_workers=8)`. A working MPI
  runtime such as MS-MPI must be installed; missing runtime support is an error,
  not an implicit serial fallback.
- The existing `alpha_clamped`, `reynolds`, `mach`, `cl`, `cd`, and `cm` outputs
  continue to expose the section data produced from generated polars.
  `HoverResult` additionally reports `clamped_element_count` and
  `substituted_polar_bins` so the GUI can drive its warning state from data.
- `new_polars=false` is a strict cache-only mode. A missing cache bin raises an
  error instead of launching XFOIL. If a solved blade element asks for alpha
  outside the loaded polar range, the solver clamps coefficients to the table
  edge and the Web UI logs a warning.
- `polar_alpha_step_deg` and `polar_n_crit` change the generated table itself,
  so they are part of the cache file key. Changing either one regenerates the
  affected bins rather than reusing a coarser or differently-tripped polar.

## Theory Notes

- The hover solver uses blade-element/momentum theory with full inflow-angle
  trigonometry, local `alpha = theta - phi`, and local `Cl/Cd(M, Re, alpha)`.
  This follows the analysis structure described by NASA/Army hover work on
  laminar rotor performance: https://ntrs.nasa.gov/api/citations/20170005472/downloads/20170005472.pdf
- Coaxial v1 applies a momentum-theory upper wake to the lower rotor. NASA
  NDARC theory notes that in hover the lower rotor acts in the contracted wake
  of the upper rotor, and uses separate upper/lower induced-power treatment:
  https://rotorcraft.arc.nasa.gov/Publications/files/NDARCTheory_v1_6_938.pdf
- The current coaxial implementation is axial-wake only. Counter-rotation,
  torque cancellation, and swirl recovery are not modeled as a lower-rotor
  incidence benefit. NASA TP-3675 notes that upper-wake contraction lets some
  outboard lower-rotor area see cleaner air, but also that lower-rotor axial
  convection differs from an isolated rotor and that swirl recovery is secondary
  for most operational coaxial helicopters:
  https://ntrs.nasa.gov/api/citations/19970015550/downloads/19970015550.pdf
- Equal-thrust coaxial trim can therefore show a noticeably higher lower
  collective when the lower disk is inside the upper axial wake. Treat the split
  as a conservative first-order estimate until calibrated against a measured
  coaxial rotor or upgraded to a free-wake/swirl model.
- The production coaxial hover trim mode is `torque_balance`, which solves
  `T_upper + T_lower = W` and `net_aircraft_yaw_torque_Nm = 0`. This permits
  unequal thrust sharing when the lower rotor is torque-heavy in the upper wake.
  Equal-thrust trim is intentionally kept as a comparison mode because it can
  produce yaw imbalance.
- `lower_rotor_speed_ratio` changes the lower rotor RPM before trim is solved.
  Torque-balanced hover still solves the same lift and yaw equations; changing
  the gear ratio changes the collectives and thrust split required to satisfy
  those equations.
- Rotor `rotation_direction=+1` means counterclockwise rotor rotation viewed
  from above; `-1` means clockwise. Shaft torque is reported as a positive
  magnitude, while aircraft yaw reaction torque is
  `-rotation_direction * shaft_torque`. Therefore positive yaw torque means the
  aircraft tends counterclockwise/left, and negative yaw torque means clockwise/right.
- The coaxial model is still an estimator, not a free-wake or CFD model. It is
  scientifically stronger than the old total-blade-count shortcut, but should be
  calibrated against measured rotor data before design-critical use.
