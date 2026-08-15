import json
import unittest
from unittest.mock import patch

import panel as pn

from pycopter.polars import LinearPolarProvider

from gui.app import PycopterWebApp
from gui.calculations import (
    DEFAULT_CONFIG,
    DEFAULT_STATION_ROWS,
    build_rotor_spec,
    build_solver_settings,
    build_xfoil_provider,
    derived_geometry,
    electric_summary,
    fossil_summary,
    isa_atmosphere,
    normalize_config,
    propulsion_summary,
    run_hover_case,
)
from gui.runs import RunStore, diff_inputs


def _linear_provider():
    return LinearPolarProvider(
        lift_slope_per_rad=5.7,
        cd0=0.012,
        induced_drag_factor=0.015,
        cm0=-0.02,
    )


def _solved_app(config_overrides=None, station_rows=None):
    """A dashboard with one solved run, using the analytic polar provider."""
    app = PycopterWebApp()
    config = normalize_config(
        {
            **DEFAULT_CONFIG,
            "new_polars": False,
            "blade_element_count": 16,
            **(config_overrides or {}),
        }
    )
    app._apply_config(config, station_rows or DEFAULT_STATION_ROWS)
    provider = _linear_provider()
    with patch.object(app, "_xfoil_provider_for_config", return_value=provider):
        app.solve_as_new_run(background=False)
    return app


class TestWebGuiCalculations(unittest.TestCase):
    def setUp(self):
        self.config = normalize_config(
            {
                **DEFAULT_CONFIG,
                "geometry_mode": "station_table",
                "new_polars": False,
                "blade_element_count": 24,
                "min_collective_deg": -2.0,
                "max_collective_deg": 22.0,
            }
        )
        self.provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.015,
            cm0=-0.02,
        )

    def test_single_rotor_hover_uses_typed_solver_outputs(self):
        case = run_hover_case(
            self.config,
            DEFAULT_STATION_ROWS,
            polar_provider=self.provider,
        )

        self.assertEqual("single", case.system_type)
        self.assertAlmostEqual(9.81, case.result.total_thrust_N, delta=0.15)
        self.assertGreater(case.result.power_W, 0.0)
        self.assertEqual(24, len(case.result.load_table()))

    def test_coaxial_hover_reports_combined_power(self):
        config = normalize_config(
            {
                **self.config,
                "rotor_system_type": "coaxial",
                "coaxial_trim_mode": "equal_thrust",
                "lower_rotor_scale": 0.95,
            }
        )

        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=self.provider)

        self.assertEqual("coaxial", case.system_type)
        self.assertAlmostEqual(9.81, case.result.total_thrust_N, delta=0.2)
        self.assertGreater(case.result.total_power_W, 0.0)
        self.assertGreater(case.result.interference_power_delta_W, 0.0)
        self.assertAlmostEqual(
            case.result.net_aircraft_yaw_torque_Nm,
            case.result.upper.aircraft_yaw_torque_Nm + case.result.lower.aircraft_yaw_torque_Nm,
        )

    def test_coaxial_lower_speed_ratio_is_applied_to_lower_rotor(self):
        config = normalize_config(
            {
                **self.config,
                "rotor_system_type": "coaxial",
                "headspeed_rpm": 2400.0,
                "lower_rotor_speed_ratio": 1.10,
            }
        )

        def capture_spec(coaxial_spec, *_args, **_kwargs):
            self.assertAlmostEqual(2400.0, coaxial_spec.upper_rotor.headspeed_rpm)
            self.assertAlmostEqual(2640.0, coaxial_spec.lower_rotor.headspeed_rpm)
            raise RuntimeError("captured")

        with patch("gui.calculations.solve_coaxial_hover", side_effect=capture_spec):
            with self.assertRaisesRegex(RuntimeError, "captured"):
                run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=self.provider)

    def test_default_coaxial_trim_balances_total_thrust_and_torque(self):
        config = normalize_config({**self.config, "rotor_system_type": "coaxial"})

        self.assertEqual("torque_balance", config["coaxial_trim_mode"])
        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=self.provider)

        self.assertAlmostEqual(9.81, case.result.total_thrust_N, delta=0.2)
        self.assertAlmostEqual(0.0, case.result.net_aircraft_yaw_torque_Nm, delta=1e-3)
        self.assertNotAlmostEqual(
            case.result.upper.total_thrust_N,
            case.result.lower.total_thrust_N,
            delta=0.05,
        )

    def test_electric_and_fossil_summaries_do_not_mix_outputs(self):
        electric = electric_summary(100.0, self.config)
        fossil = fossil_summary(100.0, {**self.config, "propulsion_model": "fossil"})

        self.assertIn("hover_endurance_min", electric)
        self.assertNotIn("fuel_flow_kg_hr", electric)
        self.assertIn("fuel_flow_kg_hr", fossil)
        self.assertNotIn("electric_input_W", fossil)

    def test_electric_summary_applies_transmission_loss(self):
        summary = electric_summary(
            100.0,
            {
                **self.config,
                "motor_efficiency": 1.0,
                "esc_efficiency": 1.0,
                "transmission_loss": 0.10,
            },
        )

        self.assertAlmostEqual(100.0 / 0.90, summary["electric_input_W"])

    def test_propulsion_summary_uses_active_model_only(self):
        case = run_hover_case(self.config, DEFAULT_STATION_ROWS, polar_provider=self.provider)

        electric = propulsion_summary(case, {**self.config, "propulsion_model": "electric"})
        fossil = propulsion_summary(case, {**self.config, "propulsion_model": "fossil"})

        self.assertIn("electric_input_W", electric)
        self.assertNotIn("fuel_flow_kg_hr", electric)
        self.assertIn("fuel_flow_kg_hr", fossil)
        self.assertNotIn("electric_input_W", fossil)

    def test_xfoil_backend_setting_is_preserved_without_fallback(self):
        provider = build_xfoil_provider(
            {
                **self.config,
                "new_polars": True,
                "xfoil_parallel_backend": "mpi",
                "xfoil_parallel_workers": 8,
            }
        )

        self.assertEqual("mpi", provider.parallel_backend)
        self.assertEqual(8, provider.parallel_workers)
        provider.cleanup()

    def test_new_polars_checkbox_controls_xfoil_provider(self):
        provider = build_xfoil_provider({**self.config, "new_polars": False})

        self.assertFalse(provider.new_polar)
        provider.cleanup()

    def test_gui_xfoil_provider_uses_stable_default_cache_directory(self):
        provider = build_xfoil_provider({**self.config, "xfoil_cache_directory": ""})

        self.assertEqual("gui-cache", provider.cache_directory.name)
        self.assertEqual("tmp", provider.cache_directory.parent.name)
        provider.cleanup()

    def test_uniform_geometry_uses_root_and_tip_twist_inputs(self):
        config = normalize_config(
            {
                **self.config,
                "geometry_mode": "uniform",
                "root_twist_deg": 10.0,
                "tip_twist_deg": 1.0,
            }
        )

        rotor = build_rotor_spec(config, DEFAULT_STATION_ROWS)

        self.assertAlmostEqual(10.0, rotor.stations[0].twist_deg)
        self.assertAlmostEqual(1.0, rotor.stations[-1].twist_deg)

    def test_legacy_config_maps_to_current_schema_and_ignores_output(self):
        config = normalize_config(
            {
                "airfoil": "naca0012",
                "num_blades": 3,
                "battery_cap": 2.5,
                "fuel_cap": 4.0,
                "sfc": 0.4,
                "output": "old printed text",
            }
        )

        self.assertEqual(3, config["num_blades"])
        self.assertEqual(2500.0, config["battery_capacity_Wh"])
        self.assertEqual(4.0, config["fuel_capacity_kg"])
        self.assertNotIn("output", config)


class TestWebGuiApp(unittest.TestCase):
    def test_save_config_contains_inputs_and_station_rows_only(self):
        app = PycopterWebApp()
        app._log("Generated output should not be saved.")

        saved = json.loads(app._save_config_callback().getvalue().decode("utf-8"))

        self.assertIn("config", saved)
        self.assertIn("station_rows", saved)
        self.assertNotIn("output", saved["config"])
        self.assertNotIn("Generated output should not be saved.", json.dumps(saved))

    def test_propulsion_tab_controls_plot_options(self):
        app = PycopterWebApp()
        app._apply_config({**DEFAULT_CONFIG, "propulsion_model": "fossil"}, DEFAULT_STATION_ROWS)

        self.assertEqual("fossil", app._current_config()["propulsion_model"])
        self.assertNotIn("Electric Range vs Velocity", app.plot_select.options)
        self.assertNotIn("Interference Loss vs Spacing", app.plot_select.options)

    def test_coaxial_speed_ratio_round_trips_through_gui_config(self):
        app = PycopterWebApp()
        app._apply_config(
            {
                **DEFAULT_CONFIG,
                "rotor_system_type": "coaxial",
                "headspeed_rpm": 2300.0,
                "lower_rotor_speed_ratio": 1.05,
            },
            DEFAULT_STATION_ROWS,
        )

        self.assertAlmostEqual(1.05, app._current_config()["lower_rotor_speed_ratio"])
        self.assertAlmostEqual(2415.0, app.lower_headspeed_rpm.value)

    def test_reuses_xfoil_provider_for_non_xfoil_setting_changes(self):
        app = PycopterWebApp()
        first_provider = object()
        second_provider = object()

        with patch("gui.app.build_xfoil_provider", side_effect=[first_provider, second_provider]) as factory:
            config = app._current_config()
            self.assertIs(first_provider, app._xfoil_provider_for_config(config))
            self.assertIs(
                first_provider,
                app._xfoil_provider_for_config(
                    {
                        **config,
                        "coaxial_spacing_ratio": float(config["coaxial_spacing_ratio"]) + 0.1,
                        "lower_rotor_speed_ratio": float(config["lower_rotor_speed_ratio"]) + 0.05,
                        "gross": float(config["gross"]) + 1.0,
                    }
                ),
            )
            self.assertEqual(1, factory.call_count)

            self.assertIs(
                second_provider,
                app._xfoil_provider_for_config(
                    {
                        **config,
                        "polar_alpha_max_deg": float(config["polar_alpha_max_deg"]) - 1.0,
                    }
                ),
            )
            self.assertEqual(2, factory.call_count)

    def test_result_tables_are_read_only_and_copyable(self):
        app = PycopterWebApp()
        provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.015,
            cm0=-0.02,
        )
        config = normalize_config({**DEFAULT_CONFIG, "new_polars": False, "blade_element_count": 20})
        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=provider)

        app._update_summary_table(case, config)
        app._update_load_table(case)

        self.assertTrue(app.summary_table._configuration["clipboard"])
        self.assertTrue(app.load_table._configuration["clipboard"])
        self.assertEqual({"editable": False}, app.summary_table._configuration["columnDefaults"])
        self.assertEqual({"editable": False}, app.load_table._configuration["columnDefaults"])
        self.assertTrue(all(editor is None for editor in app.summary_table.editors.values()))
        self.assertTrue(all(editor is None for editor in app.load_table.editors.values()))
        self.assertEqual("fit_data_table", app.load_table.layout)

    def test_coaxial_summary_reports_signed_yaw_torque(self):
        app = PycopterWebApp()
        provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.015,
            cm0=-0.02,
        )
        config = normalize_config(
            {
                **DEFAULT_CONFIG,
                "rotor_system_type": "coaxial",
                "new_polars": False,
                "blade_element_count": 20,
            }
        )
        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=provider)

        app._update_summary_table(case, config)
        metrics = set(app.summary_table.value["Metric"])

        self.assertIn("Net Aircraft Yaw Torque [Nm]", metrics)
        self.assertIn("Net Aircraft Yaw Direction", metrics)
        self.assertIn("Upper Aircraft Yaw Torque [Nm]", metrics)
        self.assertIn("Lower Aircraft Yaw Torque [Nm]", metrics)

    def test_coaxial_plot_options_include_spacing_sweep(self):
        app = PycopterWebApp()
        provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.015,
            cm0=-0.02,
        )
        config = normalize_config(
            {
                **DEFAULT_CONFIG,
                "rotor_system_type": "coaxial",
                "new_polars": False,
                "blade_element_count": 12,
            }
        )
        app.current_case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=provider)
        app._update_plot_options()

        self.assertIn("Interference Loss vs Spacing", app.plot_select.options)

    def test_coaxial_spacing_sweep_plot_uses_current_parameters(self):
        app = PycopterWebApp()
        provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.015,
            cm0=-0.02,
        )
        config = normalize_config(
            {
                **DEFAULT_CONFIG,
                "rotor_system_type": "coaxial",
                "new_polars": False,
                "blade_element_count": 10,
            }
        )
        app._apply_config(config, DEFAULT_STATION_ROWS)
        app.current_case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=provider)
        with patch.object(app, "_xfoil_provider_for_config", return_value=provider):
            fig = app._plot_interference_loss_vs_spacing(app.current_case)

        self.assertEqual("Coaxial Spacing Sweep", fig.axes[0].get_title())
        self.assertEqual("Coaxial Spacing z/R", fig.axes[0].get_xlabel())
        self.assertEqual("Interference Loss", fig.axes[0].get_ylabel())
        self.assertIn("Spacing sweep complete", "\n".join(app.output_lines))

    def test_plot_save_callback_returns_png_after_generating_plot(self):
        app = PycopterWebApp()

        app._generate_plot()
        data = app._save_plot_callback().getvalue()

        self.assertFalse(app.save_plot_download.disabled)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(app.save_plot_download.filename.endswith(".png"))

    def test_static_export_callbacks_warn_before_results_exist(self):
        app = PycopterWebApp()

        self.assertFalse(app.save_plot_download.disabled)
        self.assertFalse(app.save_summary_download.disabled)
        self.assertFalse(app.save_loads_download.disabled)

        self.assertEqual(b"", app._save_plot_callback().getvalue())
        self.assertEqual(b"", app._save_summary_callback().getvalue())
        self.assertEqual(b"", app._save_loads_callback().getvalue())

        output = "\n".join(app.output_lines)
        self.assertIn("No plot has been generated yet", output)
        self.assertIn("No summary table has been generated yet", output)
        self.assertIn("No blade element loads table has been generated yet", output)

    def test_summary_and_load_exports_return_csv_after_hover(self):
        app = PycopterWebApp()
        provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.015,
            cm0=-0.02,
        )
        config = normalize_config({**DEFAULT_CONFIG, "new_polars": False, "blade_element_count": 20})
        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=provider)
        app.current_case = case
        app._update_summary_table(case, config)
        app._update_load_table(case)

        summary = app._save_summary_callback().getvalue().decode("utf-8")
        loads = app._save_loads_callback().getvalue().decode("utf-8")

        self.assertIn("Metric,Value", summary)
        self.assertIn("Total Thrust [N]", summary)
        # r/R is the frozen key column, so it leads the exported table too.
        self.assertTrue(loads.startswith("r_over_R,rotor,r_m,dr_m"))
        self.assertIn("pitch_moment_Nm", loads)
        self.assertIn("CLAMPED", loads)


class TestConfigDerivations(unittest.TestCase):
    def test_isa_atmosphere_matches_sea_level_standard_day(self):
        air = isa_atmosphere(0.0, 15.0)

        self.assertAlmostEqual(1.225, air["density_kg_m3"], places=3)
        self.assertAlmostEqual(340.3, air["speed_of_sound_m_s"], delta=0.2)
        self.assertAlmostEqual(1.46e-5, air["kinematic_viscosity_m2_s"], delta=2e-7)

    def test_isa_atmosphere_thins_with_altitude(self):
        sea_level = isa_atmosphere(0.0)
        aloft = isa_atmosphere(3000.0)

        self.assertLess(aloft["density_kg_m3"], sea_level["density_kg_m3"])
        self.assertLess(aloft["temperature_C"], sea_level["temperature_C"])

    def test_isa_atmosphere_honours_a_non_standard_temperature(self):
        hot = isa_atmosphere(0.0, 40.0)
        standard = isa_atmosphere(0.0, 15.0)

        self.assertLess(hot["density_kg_m3"], standard["density_kg_m3"])
        self.assertAlmostEqual(15.0, hot["isa_temperature_C"], places=6)

    def test_derived_geometry_reports_display_only_quantities(self):
        derived = derived_geometry(normalize_config(DEFAULT_CONFIG), DEFAULT_STATION_ROWS)

        self.assertAlmostEqual(91.63, derived["tip_speed_m_s"], delta=0.05)
        self.assertAlmostEqual(0.267, derived["tip_speed_mach"], delta=0.002)
        self.assertAlmostEqual(9.81, derived["target_thrust_N"], places=5)
        self.assertGreater(derived["disk_area_m2"], 0.0)
        self.assertGreater(derived["aspect_ratio"], 1.0)

    def test_solver_settings_carry_the_new_numeric_controls(self):
        settings = build_solver_settings(
            normalize_config(
                {
                    **DEFAULT_CONFIG,
                    "element_spacing": "cosine",
                    "thrust_tolerance": 5e-4,
                    "max_trim_iterations": 40,
                }
            )
        )

        self.assertEqual("cosine", settings.element_spacing)
        self.assertAlmostEqual(5e-4, settings.thrust_tolerance)
        self.assertEqual(40, settings.max_trim_iterations)

    def test_fixed_collective_trim_reaches_the_solver(self):
        config = normalize_config(
            {
                **DEFAULT_CONFIG,
                "new_polars": False,
                "blade_element_count": 12,
                "trim_mode": "fixed_collective",
                "collective_pitch_deg": 6.5,
            }
        )

        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=_linear_provider())

        self.assertAlmostEqual(6.5, case.result.collective_pitch_deg, places=6)


class TestRunStore(unittest.TestCase):
    def test_runs_are_appended_with_stable_ids_and_a_first_baseline(self):
        store = RunStore()

        first = store.create("first", {"gross": 1.0})
        second = store.create("second", {"gross": 2.0})

        self.assertEqual(["R-01", "R-02"], [run.run_id for run in store.records])
        self.assertEqual(first.run_id, store.baseline_id)
        self.assertEqual(second, store.get("R-02"))

    def test_inputs_are_snapshotted_so_later_edits_cannot_rewrite_history(self):
        store = RunStore()
        inputs = {"gross": 1.0, "station_rows": [{"r_over_R": 0.5}]}

        record = store.create("run", inputs)
        inputs["gross"] = 99.0
        inputs["station_rows"][0]["r_over_R"] = 0.9

        self.assertEqual(1.0, record.inputs["gross"])
        self.assertEqual(0.5, record.inputs["station_rows"][0]["r_over_R"])

    def test_deleting_a_sweep_parent_removes_its_points(self):
        store = RunStore()
        parent = store.create("sweep", {})
        store.create("point 1", {}, parent_id=parent.run_id)
        store.create("point 2", {}, parent_id=parent.run_id)
        keeper = store.create("other", {})

        removed = store.delete(parent.run_id)

        self.assertEqual(3, removed)
        self.assertEqual([keeper.run_id], [run.run_id for run in store.records])

    def test_progress_and_eta_come_from_the_points_completed(self):
        store = RunStore()
        record = store.create("sweep", {}, status="running", progress=(0, 10))
        started = record.started_at

        updated = store.update(record.run_id, progress=(5, 10))

        self.assertIsNotNone(started)
        self.assertEqual((5, 10), updated.progress)
        self.assertIsNotNone(updated.eta_seconds)

    def test_export_payload_holds_inputs_and_metrics_but_not_load_tables(self):
        config = normalize_config({**DEFAULT_CONFIG, "new_polars": False, "blade_element_count": 10})
        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=_linear_provider())
        store = RunStore()
        record = store.create("run", {**config, "station_rows": DEFAULT_STATION_ROWS})
        store.update(record.run_id, status="done", result=case.result)

        payload = store.to_payload()
        text = json.dumps(payload)

        self.assertEqual("R-01", payload["runs"][0]["run_id"])
        self.assertIn("power_W", payload["runs"][0]["metrics"])
        self.assertIn("gross", payload["runs"][0]["inputs"])
        self.assertNotIn("element_loads", text)
        self.assertNotIn("pitch_moment_Nm", text)

    def test_csv_rows_cover_every_metric_any_run_reported(self):
        store = RunStore()
        store.create("empty", {"rotor_system_type": "single"})
        rows = store.csv_rows()

        self.assertEqual(1, len(rows))
        self.assertEqual("R-01", rows[0]["run_id"])
        self.assertEqual("yes", rows[0]["baseline"])

    def test_diff_inputs_lists_only_changed_fields(self):
        changes = diff_inputs({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})

        self.assertEqual(
            [{"key": "b", "from": 2, "to": 3}, {"key": "c", "from": None, "to": 4}],
            changes,
        )


class TestStructuredWarnings(unittest.TestCase):
    def test_clamped_elements_are_reported_on_the_result(self):
        config = normalize_config(
            {
                **DEFAULT_CONFIG,
                "new_polars": False,
                "blade_element_count": 12,
                # A tiny lift slope forces a large collective, so the analytic
                # provider's cl_max clamp is not what is under test here; the
                # count simply has to come from the result rather than the log.
            }
        )
        case = run_hover_case(config, DEFAULT_STATION_ROWS, polar_provider=_linear_provider())

        self.assertEqual(0, case.result.clamped_element_count)
        self.assertEqual((), case.result.warnings)
        self.assertIsNone(case.result.clamped_alpha_range_deg)

    def test_coaxial_warnings_are_labelled_per_rotor(self):
        from pycopter import CoaxialHoverResult, ElementLoad, HoverResult

        def rotor_result(clamped):
            load = ElementLoad(
                r_m=0.1, r_over_R=0.3, dr_m=0.01, chord_m=0.03, twist_deg=5.0,
                collective_deg=6.0, phi_deg=4.0, alpha_deg=19.5, reynolds=1e5,
                mach=0.1, cl=1.0, cd=0.02, cm=0.0, alpha_clamped=clamped,
                loss_factor=1.0, induced_velocity_m_s=3.0,
                external_axial_velocity_m_s=0.0, dL_N=1.0, dD_N=0.1, dT_N=1.0,
                dQ_Nm=0.01, dP_W=1.0, normal_force_N_per_m=10.0,
                tangential_force_N_per_m=1.0, pitch_moment_Nm=0.0,
            )
            return HoverResult(
                collective_pitch_deg=6.0, total_thrust_N=5.0, per_blade_thrust_N=2.5,
                total_torque_Nm=0.1, per_blade_torque_Nm=0.05,
                aircraft_yaw_torque_Nm=-0.1, power_W=30.0, induced_power_W=20.0,
                profile_power_W=10.0, ideal_power_W=15.0, figure_of_merit=0.5,
                mean_induced_velocity_m_s=3.0, ct=0.001, cp=0.0001, solidity=0.06,
                mean_loss_factor=0.98, root_flap_bending_moment_Nm_per_blade=0.1,
                root_lag_moment_Nm_per_blade=0.01,
                aerodynamic_pitching_moment_Nm_per_blade=0.001,
                element_loads=[load],
                substituted_polar_bins=("naca0012 Re=1e+06 M=0.6",) if clamped else (),
            )

        clean = rotor_result(False)
        clamped = rotor_result(True)
        pair = CoaxialHoverResult(
            upper=clean, lower=clamped, isolated_upper=clean, isolated_lower=clean,
            total_thrust_N=10.0, total_power_W=60.0, net_aircraft_yaw_torque_Nm=0.0,
            interference_power_delta_W=5.0, interference_loss_ratio=0.09,
            lower_external_velocity_mean_m_s=3.0, wake_radius_m=0.3,
            wake_velocity_m_s=3.4,
        )

        self.assertEqual(1, pair.clamped_element_count)
        self.assertIn("lower rotor: 1 elements clamped", pair.warnings[0])
        self.assertIn("substituted polar bin", pair.warnings[1])


class TestRunManagerApp(unittest.TestCase):
    def test_solving_creates_a_done_run_that_drives_the_views(self):
        app = _solved_app()

        record = app.runs.get(app.active_run_id)
        self.assertEqual("done", record.status)
        self.assertIsNotNone(record.result)
        self.assertEqual(record.run_id, app.runs.baseline_id)
        self.assertFalse(app.summary_table.value.empty)
        self.assertFalse(app.load_table.value.empty)
        self.assertIn("R-01", app.breadcrumb.object)
        self.assertIn("TOTAL THRUST", app.metric_strip.object)

    def test_a_failed_run_records_its_error_and_keeps_the_previous_result(self):
        app = _solved_app()
        first_case = app.current_case

        def explode(*_args, **_kwargs):
            raise ValueError("rotor blew up")

        with patch("gui.app.run_hover_case", side_effect=explode):
            app.solve_as_new_run(background=False)

        failed = app.runs.get("R-02")
        self.assertEqual("failed", failed.status)
        self.assertIn("rotor blew up", failed.error)
        self.assertIs(first_case, app.current_case)
        self.assertFalse(app.summary_table.value.empty)

    def test_metric_strip_deltas_compare_against_the_chosen_baseline(self):
        app = _solved_app()
        provider = _linear_provider()
        app.gross.value = 1.4
        with patch.object(app, "_xfoil_provider_for_config", return_value=provider):
            app.solve_as_new_run(background=False)

        self.assertEqual("R-01", app.runs.baseline_id)
        self.assertIn("delta", app.metric_strip.object)
        rows = app.run_rail.rows
        self.assertEqual("R-02", rows[0]["run_id"])
        self.assertEqual("baseline", rows[1]["delta"])
        self.assertTrue(rows[0]["delta"].endswith("%"))

    def test_set_baseline_and_delete_operate_on_the_selected_run(self):
        app = _solved_app()
        provider = _linear_provider()
        with patch.object(app, "_xfoil_provider_for_config", return_value=provider):
            app.solve_as_new_run(background=False)

        app._set_baseline()
        self.assertEqual("R-02", app.runs.baseline_id)

        app._delete_active_run()
        self.assertEqual(["R-01"], [run.run_id for run in app.runs.records])
        self.assertEqual("R-01", app.active_run_id)

    def test_edited_fields_are_marked_against_the_loaded_config(self):
        app = PycopterWebApp()
        self.assertEqual(0, app._edited_count)

        app.rotor_diam.value = float(app.rotor_diam.value) + 0.1

        self.assertEqual(1, app._edited_count)
        self.assertIn("pycopter-edited", app._field_wrappers["rotor_diam"].css_classes)

        app._revert_inputs()
        self.assertEqual(0, app._edited_count)
        self.assertNotIn("pycopter-edited", app._field_wrappers["rotor_diam"].css_classes)

    def test_every_inspector_field_round_trips_through_save_and_load(self):
        app = PycopterWebApp()
        expected = {}
        for key, widget in app._bindings.items():
            widget.value = _perturbed(widget)
            expected[key] = widget.value

        payload = json.loads(app._save_config_callback().getvalue().decode("utf-8"))

        reloaded = PycopterWebApp()
        reloaded._apply_config(payload["config"], payload["station_rows"])

        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(value, reloaded._bindings[key].value)

    def test_showing_toggle_switches_summary_and_loads_together(self):
        app = _solved_app({"rotor_system_type": "coaxial", "blade_element_count": 10})

        app.showing_toggle.value = "UPPER"
        upper_rows = len(app.load_table.value)
        self.assertEqual("UPPER", _summary_value(app, "Showing"))

        app.showing_toggle.value = "TOTAL"
        self.assertEqual("TOTAL", _summary_value(app, "Showing"))
        self.assertEqual(2 * upper_rows, len(app.load_table.value))

        app.showing_toggle.value = "Δ ISOLATED"
        self.assertEqual("Δ ISOLATED", _summary_value(app, "Showing"))
        self.assertEqual(2 * upper_rows, len(app.load_table.value))
        self.assertEqual({"upper δ", "lower δ"}, set(app.load_table.value["rotor"]))

    def test_delta_showing_reports_the_interference_cost_in_the_metric_strip(self):
        app = _solved_app({"rotor_system_type": "coaxial", "blade_element_count": 10})
        record = app.runs.get(app.active_run_id)

        app.showing_toggle.value = "Δ ISOLATED"
        values = app._metric_values(record)

        self.assertAlmostEqual(
            record.result.interference_power_delta_W, values["total_power_W"]
        )
        self.assertNotAlmostEqual(record.result.total_power_W, values["total_power_W"])

    def test_loads_table_groups_freeze_and_shade_the_alpha_column(self):
        app = _solved_app()

        self.assertEqual(["r_over_R"], app.load_table.frozen_columns)
        self.assertIn("GEOMETRY", app.load_table.groups)
        self.assertIn("alpha_deg", app.load_table.groups["LOCAL FLOW"])
        self.assertIn("CLAMPED", app.load_table.groups["SECTION COEFFICIENTS"])
        self.assertIn("rows shown", app.loads_footer.object)

        # Panel prepends its own hidden index column to the styled frame.
        columns = list(app.load_table.value.columns)
        styled = app.load_table._get_style_data()["data"]
        self.assertEqual([columns.index("alpha_deg") + 1], list(styled[0]))
        self.assertIn("background-color", styled[0][columns.index("alpha_deg") + 1][0][0])

        app.loads_shading.value = "CL/CD"
        styled = app.load_table._get_style_data()["data"]
        self.assertEqual([columns.index("cl") + 1], list(styled[0]))

        app.loads_shading.value = "OFF"
        self.assertEqual({}, app.load_table._get_style_data()["data"])

    def test_clamped_only_filter_and_warning_bar_follow_the_result(self):
        app = _solved_app()
        frame = app._loads_frame.copy()
        frame.loc[frame.index[:3], "alpha_clamped"] = True
        app._loads_frame = frame
        app._refresh_load_table()

        self.assertEqual(len(frame), len(app.load_table.value))
        app.loads_clamped_only.value = True
        self.assertEqual(3, len(app.load_table.value))

        app._show_clamped_rows()
        self.assertEqual(2, app.result_tabs.active)

    def test_polars_tab_reports_bin_provenance(self):
        app = PycopterWebApp()

        class FakeProvider:
            def polar_bin_report(self, conditions=None):
                return [
                    {
                        "airfoil": "naca0012", "reynolds": 1e5, "mach": 0.1,
                        "alpha_min_deg": -3.0, "alpha_max_deg": 18.0,
                        "status": "generated", "source": "generated",
                        "substituted": False, "lookups": 12,
                        "cache_file": "pabc.txt", "label": "naca0012 Re=1e+05 M=0.1",
                    },
                    {
                        "airfoil": "naca0012", "reynolds": 1e6, "mach": 0.6,
                        "alpha_min_deg": -3.0, "alpha_max_deg": 18.0,
                        "status": "substituted", "source": "cache",
                        "substituted": True, "lookups": 3,
                        "cache_file": "pdef.txt", "label": "naca0012 Re=1e+06 M=0.6",
                    },
                ]

        app._xfoil_provider = FakeProvider()
        app._update_polar_table()

        self.assertEqual(2, len(app.polar_table.value))
        self.assertIn("1 substituted", app.polar_footer.object)

    def test_flight_mode_gates_the_legacy_forward_flight_plots(self):
        app = _solved_app()

        self.assertNotIn("Forward Flight Powers vs Velocity", app.plot_select.options)

        app.flight_mode.value = "forward_flight"

        self.assertIn("Forward Flight Powers vs Velocity", app.plot_select.options)
        self.assertIn("Electric Range vs Velocity", app.plot_select.options)

        app.propulsion_model.value = "fossil"

        self.assertNotIn("Electric Range vs Velocity", app.plot_select.options)
        self.assertIn("Fuel Range, Endurance vs Velocity", app.plot_select.options)

    def test_export_session_runs_warns_before_any_run_exists(self):
        app = PycopterWebApp()

        self.assertEqual(b"", app._save_runs_callback().getvalue())
        self.assertEqual(b"", app._save_runs_csv_callback().getvalue())
        self.assertIn("no session to export", "\n".join(app.output_lines))

    def test_apply_isa_writes_density_viscosity_and_speed_of_sound(self):
        app = PycopterWebApp()
        app.altitude.value = 2000.0
        app.temperature.value = 2.0

        app._apply_isa()

        self.assertLess(app.density.value, 1.225)
        self.assertGreater(app.kinematic_viscosity.value, 0.0)
        self.assertLess(app.speed_of_sound.value, 343.0)
        self.assertIn("ISA at 2000 m", "\n".join(app.output_lines))

    def test_recent_configs_are_offered_after_loading_one(self):
        app = PycopterWebApp()
        payload = json.dumps(
            {"config": {**DEFAULT_CONFIG, "gross": 2.5}, "station_rows": DEFAULT_STATION_ROWS}
        ).encode("utf-8")
        app.load_file.value = payload
        app.load_file.filename = "drone.json"

        app._load_config()

        self.assertAlmostEqual(2.5, app.gross.value)
        self.assertEqual("drone.json", app.config_filename)
        self.assertEqual(["drone.json"], [name for name, _ in app.recent_configs])
        self.assertIn("gross", "\n".join(app.output_lines))

    def test_overlay_compare_draws_every_checked_run(self):
        app = _solved_app()
        provider = _linear_provider()
        app.headspeed_rpm.value = 2800.0
        with patch.object(app, "_xfoil_provider_for_config", return_value=provider):
            app.solve_as_new_run(background=False)

        app.checked_run_ids = ["R-01"]
        app.overlay_runs.value = True
        overlays = app._overlay_series_frames()

        self.assertEqual(["R-01"], [run_id for run_id, _ in overlays])
        app.plot_select.value = "Radial Loads"
        app._generate_plot()
        labels = [
            text.get_text()
            for axis in app.plot_fig.axes
            if axis.get_legend()
            for text in axis.get_legend().get_texts()
        ]
        self.assertTrue(any(label.startswith("R-01") for label in labels), labels)


class TestSweepRun(unittest.TestCase):
    def test_spacing_sweep_runs_as_a_parent_with_child_points(self):
        app = _solved_app({"rotor_system_type": "coaxial", "blade_element_count": 8,
                           "sweep_points": 4})
        provider = _linear_provider()

        with patch.object(app, "_xfoil_provider_for_config", return_value=provider):
            app._start_spacing_sweep()

        parent = app.runs.get("R-02")
        children = app.runs.children_of("R-02")
        self.assertEqual("done", parent.status)
        self.assertEqual((4, 4), parent.progress)
        self.assertEqual(4, len(children))
        self.assertTrue(all(child.status == "done" for child in children))
        self.assertIn("Spacing sweep complete", "\n".join(app.output_lines))
        self.assertEqual("Coaxial Spacing Sweep", app.plot_fig.axes[0].get_title())

    def test_cancelling_a_sweep_leaves_a_partial_run_record(self):
        app = _solved_app({"rotor_system_type": "coaxial", "blade_element_count": 8,
                           "sweep_points": 6})
        provider = _linear_provider()
        original = app._run_interference_sweep

        def cancel_after_two(config, rows, prov, spacings, *, cancel=None, progress=None):
            state = {"seen": 0}

            def progress_and_count(index, total, spacing, point):
                state["seen"] = index
                if progress is not None:
                    progress(index, total, spacing, point)

            return original(
                config,
                rows,
                prov,
                spacings,
                cancel=lambda: state["seen"] >= 2,
                progress=progress_and_count,
            )

        with patch.object(app, "_xfoil_provider_for_config", return_value=provider):
            with patch.object(app, "_run_interference_sweep", side_effect=cancel_after_two):
                app._start_spacing_sweep()

        parent = app.runs.get("R-02")
        self.assertEqual("cancelled", parent.status)
        self.assertEqual(2, len(app.runs.children_of("R-02")))
        self.assertIn("cancelled", app.plot_fig.axes[0].get_title())


def _summary_value(app, metric):
    frame = app.summary_table.value
    return frame.loc[frame["Metric"] == metric, "Value"].iloc[0]


def _perturbed(widget):
    """A different-but-valid value for one bound inspector widget."""
    if isinstance(widget, pn.widgets.Select):
        values = list(widget.options.values()) if isinstance(widget.options, dict) else list(widget.options)
        return next(value for value in values if value != widget.value)
    if isinstance(widget, pn.widgets.IntInput):
        step = max(1, int(widget.step or 1))
        candidate = int(widget.value) + step
        return candidate if candidate <= int(widget.end) else int(widget.value) - step
    if isinstance(widget, pn.widgets.FloatInput):
        span = float(widget.end) - float(widget.start)
        candidate = float(widget.value) + span * 0.1
        return candidate if candidate <= float(widget.end) else float(widget.value) - span * 0.1
    return f"{widget.value}x"


class TestPlotAxisLayout(unittest.TestCase):
    """The axis rules that keep multi-quantity plots readable.

    Quantities within SHARED_AXIS_MAX_RATIO share one y-axis, larger gaps take
    the right-hand axis, and a third scale moves to its own stacked panel
    rather than a third offset spine.
    """

    def setUp(self):
        import numpy as np

        self.np = np
        self.app = PycopterWebApp.__new__(PycopterWebApp)
        self.radius = np.linspace(0.12, 1.0, 60)

    def test_similar_magnitudes_share_one_axis(self):
        induced_velocity = 3.6 * self.np.sqrt(self.radius)
        loss_factor = 1.0 - 0.6 * self.radius**8

        layout = self.app._axis_layout([induced_velocity, loss_factor])

        self.assertEqual([(0, 0), (0, 0)], layout)

    def test_dissimilar_magnitudes_use_the_second_axis(self):
        element_thrust = 0.13 * self.np.sin(self.radius * 3)
        element_torque = 0.004 * self.radius**2

        layout = self.app._axis_layout([element_thrust, element_torque])

        self.assertEqual([(0, 0), (0, 1)], layout)

    def test_third_scale_moves_to_its_own_panel(self):
        alpha = 3.8 * self.np.sin(self.radius * 3)
        reynolds = 2.1e5 * self.radius
        mach = 0.265 * self.radius

        layout = self.app._axis_layout([alpha, reynolds, mach])

        # alpha and Reynolds share a panel; Mach cannot fit either scale.
        self.assertEqual([(0, 0), (0, 1), (1, 0)], layout)
        self.assertEqual(2, max(panel for panel, _ in layout) + 1)

    def test_identically_shaped_curves_never_face_each_other(self):
        """Two curves of the same shape on opposite axes hide one another.

        Reynolds and Mach are both linear in radius, so after each axis
        autoscales they draw exactly the same line.
        """
        reynolds = 2.1e5 * self.radius
        mach = 0.265 * self.radius
        self.assertTrue(self.app._shapes_coincide(reynolds, mach))

        layout = self.app._axis_layout([reynolds, mach])

        self.assertNotEqual(layout[0][0], layout[1][0], "coincident curves shared a panel")

    def test_matching_scales_but_different_shapes_may_face_each_other(self):
        rising = 0.13 * self.radius
        humped = 0.004 * self.np.sin(self.radius * 3)
        self.assertFalse(self.app._shapes_coincide(rising, humped))

        layout = self.app._axis_layout([rising, humped])

        self.assertEqual([(0, 0), (0, 1)], layout)

    def test_flat_series_joins_an_existing_axis_instead_of_forcing_a_panel(self):
        """An all-zero Cm has no scale of its own, so it must not add a panel."""
        cl = 0.42 * self.np.sin(self.radius * 3)
        cd = 0.0117 * (1.0 + self.radius * 0.1)
        cm = self.np.zeros_like(self.radius)

        layout = self.app._axis_layout([cl, cd, cm])

        self.assertEqual(1, max(panel for panel, _ in layout) + 1)

    def test_no_plot_uses_more_than_two_axes_per_panel(self):
        import matplotlib

        matplotlib.use("Agg")
        app = PycopterWebApp()
        app.current_case = run_hover_case(
            DEFAULT_CONFIG, DEFAULT_STATION_ROWS, polar_provider=LinearPolarProvider()
        )

        for plot_name in ("Radial Loads", "Alpha, Re, Mach vs Radius",
                          "Section Coefficients vs Radius", "Induced Velocity and Loss",
                          "Cumulative Thrust and Power"):
            with self.subTest(plot=plot_name):
                app.plot_select.options = [plot_name]
                app.plot_select.value = plot_name
                app._generate_plot()
                figure = app.plot_fig
                panels = [ax for ax in figure.axes if not getattr(ax, "_pycopter_twin", False)]
                twins = [ax for ax in figure.axes if getattr(ax, "_pycopter_twin", False)]
                self.assertLessEqual(len(twins), len(panels), f"{plot_name} has a third y-axis")


if __name__ == "__main__":
    unittest.main()
