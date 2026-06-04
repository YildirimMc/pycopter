import unittest

from gui.state import (
    DashboardSettings,
    FlightSettings,
    RotorSettings,
    format_metric_rows,
    validate_airfoil_name,
)


class DashboardStateTests(unittest.TestCase):
    def test_loads_legacy_preset_keys(self):
        settings = DashboardSettings.from_mapping(
            {
                "airfoil": "NACA0015",
                "num_blades": 5,
                "chord": 0.17,
                "rotor_diam": 8.05,
                "tip_speed_mach": 0.604,
                "washout": -9.0,
                "root_cutout": 0.032,
                "gross": 1361,
                "density": 1.225,
                "transmission_loss": 0.11,
                "fuel_cap": 175,
                "sfc": 0.424,
                "battery_cap": 1500,
                "velocity": 100.0,
                "fpa": 0.557,
                "output": "legacy text output is ignored",
            }
        )

        self.assertEqual(settings.rotor.airfoil, "naca0015")
        self.assertEqual(settings.rotor.rotor_diameter, 8.05)
        self.assertEqual(settings.rotor.rotor_root_cutout, 0.032)
        self.assertEqual(settings.flight.gross_weight, 1361)
        self.assertEqual(settings.flight.flat_plate_area, 0.557)

    def test_velocity_conversion_uses_meters_per_second(self):
        flight = FlightSettings(velocity=90.0)

        self.assertAlmostEqual(flight.velocity_mps, 25.0)

    def test_rotor_kwargs_match_current_rotor_constructor(self):
        rotor = RotorSettings(
            airfoil="NACA23012",
            num_blades=4,
            chord=0.4,
            rotor_diameter=12.0,
            tip_speed_mach=0.62,
            washout=-6.5,
            rotor_root_cutout=0.03,
            new_polar=False,
        )

        self.assertEqual(
            rotor.to_rotor_kwargs(),
            {
                "airfoil": "naca23012",
                "num_blades": 4,
                "chord": 0.4,
                "rotor_diameter": 12.0,
                "tip_speed_mach": 0.62,
                "washout": -6.5,
                "rotor_root_cutout": 0.03,
                "new_polar": False,
            },
        )

    def test_airfoil_validator_accepts_naca_profiles(self):
        self.assertEqual(validate_airfoil_name("NACA23012"), "naca23012")

    def test_airfoil_validator_rejects_non_naca_profiles(self):
        with self.assertRaises(ValueError):
            validate_airfoil_name("sc1095")

    def test_metric_rows_are_grouped_and_formatted(self):
        rows = format_metric_rows(
            [
                ("Geometry", "Tip speed", 214.0322, "m/s"),
                ("Hover", "Figure of merit", 0.75291, ""),
            ]
        )

        self.assertIn("Geometry", rows)
        self.assertIn("Tip speed", rows)
        self.assertIn("214.032", rows)
        self.assertIn("Figure of merit", rows)
        self.assertIn("0.753", rows)


if __name__ == "__main__":
    unittest.main()
