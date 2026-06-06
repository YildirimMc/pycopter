import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pycopter import Rotor
from pycopter.bemt import HoverSolver, solve_coaxial_hover
from pycopter.models import (
    BladeStation,
    CoaxialSpec,
    HoverSolverSettings,
    OperatingPoint,
    RotorSpec,
)
from pycopter.polars import LinearPolarProvider, XfoilPolarProvider


class TestBemtHover(unittest.TestCase):
    def setUp(self):
        self.polar_provider = LinearPolarProvider(
            lift_slope_per_rad=5.7,
            cd0=0.012,
            induced_drag_factor=0.015,
            cm0=-0.02,
        )
        self.rotor = RotorSpec.from_uniform_blade(
            airfoil="naca0012",
            num_blades=2,
            chord_m=0.035,
            rotor_diameter_m=0.7,
            headspeed_rpm=2500.0,
            washout_deg=-6.0,
            root_cutout_ratio=0.12,
            station_count=4,
        )
        self.settings = HoverSolverSettings(
            blade_element_count=36,
            min_collective_deg=-2.0,
            max_collective_deg=22.0,
            thrust_tolerance=1e-3,
            collective_tolerance_deg=1e-4,
            induced_power_factor=1.0,
        )

    def test_uniform_geometry_creates_radial_station_table(self):
        self.assertEqual(4, len(self.rotor.stations))
        self.assertAlmostEqual(0.12, self.rotor.stations[0].r_over_R)
        self.assertAlmostEqual(1.0, self.rotor.stations[-1].r_over_R)
        self.assertAlmostEqual(0.035, self.rotor.chord_at(0.5))
        self.assertLess(self.rotor.twist_at(1.0), self.rotor.twist_at(0.12))
        self.assertGreater(self.rotor.solidity, 0.0)

    def test_hover_trim_matches_target_and_sums_per_blade_loads(self):
        solver = HoverSolver(self.polar_provider, self.settings)
        result = solver.solve(
            self.rotor,
            OperatingPoint(target_thrust_N=10.0, trim_mode="target_thrust"),
        )

        self.assertAlmostEqual(10.0, result.total_thrust_N, delta=0.1)
        self.assertGreater(result.power_W, 0.0)
        self.assertGreater(result.mean_induced_velocity_m_s, 0.0)
        self.assertGreater(result.figure_of_merit, 0.0)
        self.assertLess(result.figure_of_merit, 1.0)

        load_sum = sum(load.dT_N for load in result.element_loads)
        torque_sum = sum(load.dQ_Nm for load in result.element_loads)
        self.assertAlmostEqual(result.per_blade_thrust_N, load_sum, delta=1e-6)
        self.assertAlmostEqual(result.per_blade_torque_Nm, torque_sum, delta=1e-6)
        self.assertAlmostEqual(
            result.total_thrust_N,
            result.per_blade_thrust_N * self.rotor.num_blades,
            delta=1e-6,
        )

    def test_fixed_collective_reports_local_low_reynolds_values(self):
        solver = HoverSolver(self.polar_provider, self.settings)
        result = solver.solve(
            self.rotor,
            OperatingPoint(
                collective_pitch_deg=10.0,
                trim_mode="fixed_collective",
            ),
        )

        self.assertAlmostEqual(10.0, result.collective_pitch_deg)
        self.assertTrue(all(load.reynolds > 1000.0 for load in result.element_loads))
        self.assertTrue(all(load.mach < 0.5 for load in result.element_loads))
        self.assertTrue(any(load.cm != 0.0 for load in result.element_loads))
        self.assertNotEqual(0.0, result.aerodynamic_pitching_moment_Nm_per_blade)

    def test_hover_solver_prefetches_radial_polar_conditions(self):
        class RecordingPolarProvider(LinearPolarProvider):
            def __init__(self):
                super().__init__()
                self.prepared_conditions = []

            def prepare_conditions(self, conditions):
                self.prepared_conditions.extend(list(conditions))

        provider = RecordingPolarProvider()
        solver = HoverSolver(provider, self.settings)
        solver.solve(
            self.rotor,
            OperatingPoint(
                collective_pitch_deg=10.0,
                trim_mode="fixed_collective",
            ),
        )

        self.assertGreaterEqual(
            len(provider.prepared_conditions),
            self.settings.blade_element_count,
        )
        airfoil, reynolds, mach = provider.prepared_conditions[0]
        self.assertEqual("naca0012", airfoil)
        self.assertGreater(reynolds, 0.0)
        self.assertGreaterEqual(mach, 0.0)

    def test_coaxial_hover_reports_lower_rotor_interference(self):
        result = solve_coaxial_hover(
            CoaxialSpec(
                upper_rotor=self.rotor,
                lower_rotor=self.rotor,
                spacing_ratio=0.25,
                trim_mode="equal_thrust",
            ),
            OperatingPoint(target_thrust_N=20.0, trim_mode="target_thrust"),
            polar_provider=self.polar_provider,
            settings=self.settings,
        )

        self.assertAlmostEqual(20.0, result.total_thrust_N, delta=0.2)
        self.assertGreater(result.lower_external_velocity_mean_m_s, 0.0)
        self.assertGreater(result.interference_power_delta_W, 0.0)
        self.assertGreater(result.interference_loss_ratio, 0.0)
        self.assertAlmostEqual(
            result.upper.total_thrust_N,
            result.lower.total_thrust_N,
            delta=0.2,
        )

    def test_legacy_rotor_hover_uses_new_solver_outputs(self):
        rotor = Rotor(
            airfoil="naca0012",
            num_blades=2,
            chord=0.035,
            rotor_diameter=0.7,
            tip_speed_mach=0.265,
            washout=-6.0,
            rotor_root_cutout=0.12,
            new_polar=False,
            polar_provider=self.polar_provider,
            solver_settings=self.settings,
        )

        rotor.hover(weight=1.0, density=1.225)

        self.assertAlmostEqual(9.81, rotor.hover_thrust, delta=0.2)
        self.assertGreater(rotor.hover_power_total, 0.0)
        self.assertTrue(hasattr(rotor, "hover_result"))
        self.assertEqual(len(rotor.hover_result.element_loads), 36)


class TestInputValidation(unittest.TestCase):
    def test_station_table_must_be_monotonic(self):
        with self.assertRaises(ValueError):
            RotorSpec(
                airfoil="naca0012",
                num_blades=2,
                rotor_diameter_m=0.7,
                headspeed_rpm=2500.0,
                stations=[
                    BladeStation(0.8, 0.03, 0.0),
                    BladeStation(0.5, 0.03, 0.0),
                ],
            )


class TestXfoilProviderBounds(unittest.TestCase):
    def test_xfoil_provider_default_cache_directory_is_temporary(self):
        provider = XfoilPolarProvider()
        cache_path = provider._cache_file_path("naca0012", 100000.0, 0.1)
        repo_root = Path(__file__).resolve().parents[1]
        temp_root = repo_root / "data" / "XFOIL6.99" / "tmp"

        self.assertTrue(cache_path.resolve().is_relative_to(temp_root.resolve()))
        self.assertTrue(provider.cache_directory.name.startswith("px"))
        self.assertLessEqual(len(provider.cache_directory.name), 8)
        provider.cleanup()
        self.assertFalse(provider.cache_directory.exists())

    def test_xfoil_provider_cache_filename_stays_within_xfoil_limit(self):
        provider = XfoilPolarProvider(alpha_min_deg=-10.0, alpha_max_deg=15.0)
        cache_path = provider._cache_file_path("naca0012", 225000.0, 0.02)

        self.assertEqual(".txt", cache_path.suffix)
        self.assertLessEqual(len(cache_path.name), 32)
        provider.cleanup()

    def test_xfoil_provider_caps_requested_alpha_to_15_degrees(self):
        calls = {}

        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60):
                self.new_polar = new_polar
                self.timeout = timeout

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                calls["alpha_max_deg"] = alpha_max_deg
                return True

            def read_polar(self):
                return [
                    [-10.0, -0.8, 0.04, 0.0, 0.0],
                    [0.0, 0.0, 0.01, 0.0, 0.0],
                    [15.0, 1.0, 0.05, 0.0, 0.0],
                ]

        with patch("pycopter.polars.Xfoil", FakeXfoil):
            provider = XfoilPolarProvider(alpha_max_deg=25.0)
            coeffs = provider.get_coefficients("naca0012", 18.0, 100000.0, 0.1)

        self.assertLessEqual(calls["alpha_max_deg"], 15.0)
        self.assertEqual(15.0, coeffs.alpha_deg)
        self.assertTrue(coeffs.alpha_clamped)

    def test_xfoil_provider_never_rounds_reynolds_to_zero(self):
        calls = {}

        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60):
                pass

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                calls["reynolds"] = reynolds
                return True

            def read_polar(self):
                return [
                    [-8.0, -0.8, 0.04, 0.0, 0.0],
                    [0.0, 0.0, 0.01, 0.0, 0.0],
                    [15.0, 1.0, 0.05, 0.0, 0.0],
                ]

        with patch("pycopter.polars.Xfoil", FakeXfoil):
            provider = XfoilPolarProvider(reynolds_bin=200000.0)
            provider.get_coefficients("naca0012", 5.0, 1000.0, 0.0)

        self.assertGreater(calls["reynolds"], 0.0)

    def test_xfoil_provider_uses_per_condition_cache_files(self):
        output_paths = []

        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60):
                self.new_polar = new_polar
                self.timeout = timeout
                self.repo_root = Path.cwd()
                self.output_path = self.repo_root / "data" / "XFOIL6.99" / "polar.txt"
                self.output_path_for_xfoil = "data/XFOIL6.99/polar.txt"

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                output_paths.append(self.output_path)
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(
                    "\n" * 12
                    + "-8 0.0 0.01 0 0\n"
                    + "0 0.0 0.01 0 0\n"
                    + "15 1.0 0.05 0 0\n",
                    encoding="utf-8",
                )
                return True

            def read_polar(self):
                import numpy as np

                return np.genfromtxt(self.output_path, skip_header=12)

        with TemporaryDirectory() as tempdir:
            with patch("pycopter.polars.Xfoil", FakeXfoil):
                provider = XfoilPolarProvider(cache_directory=tempdir)
                provider.get_coefficients("naca0012", 5.0, 100000.0, 0.1)
                provider.get_coefficients("naca0012", 5.0, 150000.0, 0.1)

        self.assertEqual(2, len(output_paths))
        self.assertNotEqual(output_paths[0], output_paths[1])
        self.assertTrue(all(path.name != "polar.txt" for path in output_paths))

    def test_xfoil_provider_reuses_disk_cache_when_new_polar_is_false(self):
        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60):
                self.new_polar = new_polar
                self.timeout = timeout
                self.repo_root = Path.cwd()
                self.output_path = self.repo_root / "data" / "XFOIL6.99" / "polar.txt"
                self.output_path_for_xfoil = "data/XFOIL6.99/polar.txt"

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                raise AssertionError("new_polar=False should read the cache file")

            def read_polar(self):
                import numpy as np

                return np.genfromtxt(self.output_path, skip_header=12)

        with TemporaryDirectory() as tempdir:
            provider = XfoilPolarProvider(new_polar=False, cache_directory=tempdir)
            cache_path = provider._cache_file_path("naca0012", 100000.0, 0.1)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                "\n" * 12
                + "-8 0.0 0.01 0 0\n"
                + "0 0.0 0.01 0 0\n"
                + "15 1.0 0.05 0 0\n",
                encoding="utf-8",
            )

            with patch("pycopter.polars.Xfoil", FakeXfoil):
                coeffs = provider.get_coefficients("naca0012", 5.0, 100000.0, 0.1)

        self.assertGreater(coeffs.cl, 0.0)

    def test_xfoil_provider_prefetch_uses_mpi_executor_with_eight_workers(self):
        calls = {}

        class FakeMpiExecutor:
            def __init__(self, max_workers):
                calls["max_workers"] = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, jobs):
                calls["job_count"] = len(jobs)
                return [func(job) for job in jobs]

        def fake_xfoil_job(job):
            return job.result_from_table(
                [
                    [-8.0, -0.8, 0.04, 0.0, 0.0],
                    [0.0, 0.0, 0.01, 0.0, 0.0],
                    [15.0, 1.0, 0.05, 0.0, 0.0],
                ]
            )

        with TemporaryDirectory() as tempdir:
            with patch(
                "pycopter.polars._get_mpi_pool_executor",
                return_value=(FakeMpiExecutor, None),
            ):
                with patch("pycopter.polars._run_xfoil_polar_job", fake_xfoil_job):
                    provider = XfoilPolarProvider(cache_directory=tempdir)
                    provider.prepare_conditions(
                        [
                            ("naca0012", 100000.0, 0.1),
                            ("naca0012", 150000.0, 0.1),
                        ]
                    )

                    coeffs = provider.get_coefficients(
                        "naca0012",
                        5.0,
                        100000.0,
                        0.1,
                    )

        self.assertEqual(8, calls["max_workers"])
        self.assertEqual(2, calls["job_count"])
        self.assertGreater(coeffs.cl, 0.0)

    def test_xfoil_provider_can_require_mpi_backend(self):
        with patch(
            "pycopter.polars._get_mpi_pool_executor",
            return_value=(None, RuntimeError("no mpi runtime")),
        ):
            provider = XfoilPolarProvider(parallel_backend="mpi")
            with self.assertRaises(RuntimeError):
                provider.prepare_conditions(
                    [
                        ("naca0012", 100000.0, 0.1),
                        ("naca0012", 150000.0, 0.1),
                    ]
                )
            provider.cleanup()


class TestRealXfoilHover(unittest.TestCase):
    def test_reported_naca0012_low_mach_polar_filename_case_completes(self):
        repo_root = Path(__file__).resolve().parents[1]
        xfoil_exe = repo_root / "data" / "XFOIL6.99" / "xfoil.exe"
        if not xfoil_exe.exists():
            self.skipTest("XFOIL executable is not available.")

        provider = XfoilPolarProvider(
            new_polar=True,
            alpha_min_deg=-10.0,
            alpha_max_deg=15.0,
            reynolds_bin=25000.0,
            mach_bin=0.02,
            timeout=20,
            parallel_backend="serial",
        )

        coeffs = provider.get_coefficients("naca0012", 5.0, 225000.0, 0.02)

        self.assertGreater(coeffs.cl, 0.0)
        provider.cleanup()

    def test_naca0012_hover_completes_with_xfoil_polars(self):
        repo_root = Path(__file__).resolve().parents[1]
        xfoil_exe = repo_root / "data" / "XFOIL6.99" / "xfoil.exe"
        if not xfoil_exe.exists():
            self.skipTest("XFOIL executable is not available.")

        rotor = Rotor(
            airfoil="naca0012",
            num_blades=2,
            chord=0.035,
            rotor_diameter=0.7,
            tip_speed_mach=0.265,
            washout=-6.0,
            rotor_root_cutout=0.12,
            polar_provider=XfoilPolarProvider(
                new_polar=True,
                alpha_min_deg=-8.0,
                alpha_max_deg=15.0,
                reynolds_bin=200000.0,
                mach_bin=0.1,
                timeout=20,
            ),
            solver_settings=HoverSolverSettings(
                blade_element_count=8,
                min_collective_deg=-2.0,
                max_collective_deg=15.0,
                collective_tolerance_deg=0.05,
                thrust_tolerance=0.03,
            ),
        )

        rotor.hover(weight=1.0, density=1.225, n=8)

        self.assertAlmostEqual(9.81, rotor.hover_thrust, delta=0.35)
        self.assertGreater(rotor.hover_power_total, 0.0)
        self.assertEqual(8, len(rotor.hover_result.element_loads))
        self.assertFalse(any(load.alpha_clamped for load in rotor.hover_result.element_loads))


if __name__ == "__main__":
    unittest.main()
