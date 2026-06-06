import unittest
from math import pi, sqrt

from pycopter.bemt import HoverSolver
from pycopter.models import BladeStation, HoverSolverSettings, OperatingPoint, RotorSpec
from pycopter.polars import LinearPolarProvider


RHO_SEA_LEVEL_KG_M3 = 1.225
GRAVITY_M_S2 = 9.81


def ideal_hover_power_W(thrust_N: float, rotor: RotorSpec) -> float:
    disk_area_m2 = pi * (rotor.radius_m**2 - rotor.root_radius_m**2)
    return thrust_N**1.5 / sqrt(2.0 * RHO_SEA_LEVEL_KG_M3 * disk_area_m2)


def nasa_uh60_hover_cp(thrust_coefficient: float) -> float:
    # NASA UH-60A OGE production-validation hover fit:
    # Cp = 7.846e-5 + 1.0443 * Ct^1.5
    return 7.846e-5 + 1.0443 * thrust_coefficient**1.5


class TestRealWorldHoverCases(unittest.TestCase):
    def setUp(self):
        self.settings = HoverSolverSettings(
            blade_element_count=72,
            min_collective_deg=-10.0,
            max_collective_deg=30.0,
            thrust_tolerance=0.005,
            induced_power_factor=1.08,
        )
        self.provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.010,
            cl_max=1.3,
        )

    def solve(self, rotor: RotorSpec, mass_kg: float):
        return HoverSolver(self.provider, self.settings).solve(
            rotor,
            OperatingPoint(
                target_thrust_N=mass_kg * GRAVITY_M_S2,
                trim_mode="target_thrust",
            ),
        )

    def test_robinson_r22_hover_power_is_plausible_at_max_gross(self):
        # Public R22 POH values: 2 blades, 25 ft 2 in main rotor diameter,
        # 7.2-7.7 in chord, -8 deg blade twist, 622 kg Beta/Beta II gross
        # weight. Use a single representative chord for this low-order model.
        rotor = RotorSpec.from_uniform_blade(
            airfoil="naca0012",
            num_blades=2,
            chord_m=0.190,
            rotor_diameter_m=7.67,
            headspeed_rpm=510.0,
            washout_deg=-8.0,
            root_cutout_ratio=0.08,
        )

        result = self.solve(rotor, 622.0)
        ideal_power = ideal_hover_power_W(622.0 * GRAVITY_M_S2, rotor)

        self.assertGreater(result.power_W, ideal_power)
        self.assertLess(result.power_W, 95_000.0)
        self.assertGreater(result.figure_of_merit, 0.45)
        self.assertLess(result.figure_of_merit, 0.75)

    def test_uh60_hover_power_tracks_published_nasa_coefficient_fit(self):
        # Public UH-60 data: 4 blades, 16.36 m diameter, 20.76 in chord,
        # -18 deg equivalent twist, 258 rpm. Sikorsky lists the UH-60L mission
        # gross weight as 16,976 lb; the UH-60A NASA fit is used as a main
        # rotor hover-power coefficient sanity check.
        rotor = RotorSpec.from_uniform_blade(
            airfoil="naca0012",
            num_blades=4,
            chord_m=0.527,
            rotor_diameter_m=16.36,
            headspeed_rpm=258.0,
            washout_deg=-18.0,
            root_cutout_ratio=0.09,
        )
        result = self.solve(rotor, 7700.0)

        expected_cp = nasa_uh60_hover_cp(result.ct)
        self.assertGreater(result.cp, 0.80 * expected_cp)
        self.assertLess(result.cp, 1.05 * expected_cp)
        self.assertGreater(result.figure_of_merit, 0.55)
        self.assertLess(result.figure_of_merit, 0.75)

    def test_oversized_mi6_rotor_can_trim_tiny_target_below_ui_minimum(self):
        # Mi-6-sized rotor: 5 blades, 35 m diameter. This intentionally uses
        # the Web UI root/tip twist convention and a 1 kg target mass.
        rotor = RotorSpec(
            airfoil="naca0012",
            num_blades=5,
            rotor_diameter_m=35.0,
            headspeed_rpm=120.0,
            root_cutout_ratio=0.08,
            stations=[
                BladeStation(0.08, 1.0, 10.0, "naca0012"),
                BladeStation(0.40, 1.0, 7.0, "naca0012"),
                BladeStation(0.75, 1.0, 3.5, "naca0012"),
                BladeStation(1.00, 1.0, 1.0, "naca0012"),
            ],
        )
        settings = HoverSolverSettings(
            blade_element_count=48,
            min_collective_deg=0.0,
            max_collective_deg=25.0,
            thrust_tolerance=0.005,
            induced_power_factor=1.08,
        )

        result = HoverSolver(self.provider, settings).solve(
            rotor,
            OperatingPoint(target_thrust_N=GRAVITY_M_S2, trim_mode="target_thrust"),
        )

        self.assertLess(result.collective_pitch_deg, 0.0)
        self.assertAlmostEqual(GRAVITY_M_S2, result.total_thrust_N, delta=0.10)
        self.assertGreater(result.power_W, 1_000_000.0)


if __name__ == "__main__":
    unittest.main()
