import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
