import json
import unittest
from unittest.mock import patch

from pycopter.polars import LinearPolarProvider

from gui.app import PycopterWebApp
from gui.calculations import (
    DEFAULT_CONFIG,
    DEFAULT_STATION_ROWS,
    build_rotor_spec,
    build_xfoil_provider,
    electric_summary,
    fossil_summary,
    normalize_config,
    propulsion_summary,
    run_hover_case,
)


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
        self.assertIn("r_m,r_over_R", loads)
        self.assertIn("pitch_moment_Nm", loads)


if __name__ == "__main__":
    unittest.main()
