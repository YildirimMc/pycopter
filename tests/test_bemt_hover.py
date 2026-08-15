import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pycopter import Rotor
from pycopter.bemt import HoverSolver, solve_coaxial_hover, sweep_interference_loss
from pycopter.models import (
    BladeStation,
    CoaxialSpec,
    HoverSolverSettings,
    OperatingPoint,
    RotorSpec,
)
from pycopter.polars import LinearPolarProvider, XfoilPolarProvider
from pycopter.xfoil import Xfoil


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
            result.aircraft_yaw_torque_Nm,
            -result.total_torque_Nm,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            result.total_thrust_N,
            result.per_blade_thrust_N * self.rotor.num_blades,
            delta=1e-6,
        )

    def test_yaw_reaction_torque_uses_rotation_direction_sign(self):
        clockwise_rotor = RotorSpec.from_uniform_blade(
            airfoil="naca0012",
            num_blades=2,
            chord_m=0.035,
            rotor_diameter_m=0.7,
            headspeed_rpm=2500.0,
            washout_deg=-6.0,
            root_cutout_ratio=0.12,
            station_count=4,
            rotation_direction=-1,
        )
        solver = HoverSolver(self.polar_provider, self.settings)

        result = solver.solve(
            clockwise_rotor,
            OperatingPoint(target_thrust_N=10.0, trim_mode="target_thrust"),
        )

        self.assertGreater(result.total_torque_Nm, 0.0)
        self.assertAlmostEqual(
            result.aircraft_yaw_torque_Nm,
            result.total_torque_Nm,
            delta=1e-6,
        )

    def test_target_trim_expands_below_configured_min_collective(self):
        oversized_rotor = RotorSpec(
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
            blade_element_count=32,
            min_collective_deg=0.0,
            max_collective_deg=25.0,
            thrust_tolerance=0.02,
            collective_tolerance_deg=0.02,
        )
        solver = HoverSolver(self.polar_provider, settings)

        result = solver.solve(
            oversized_rotor,
            OperatingPoint(target_thrust_N=9.81, trim_mode="target_thrust"),
        )

        self.assertLess(result.collective_pitch_deg, 0.0)
        self.assertAlmostEqual(9.81, result.total_thrust_N, delta=0.25)

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
        self.assertLess(result.upper.aircraft_yaw_torque_Nm, 0.0)
        self.assertGreater(result.lower.aircraft_yaw_torque_Nm, 0.0)
        self.assertAlmostEqual(
            result.net_aircraft_yaw_torque_Nm,
            result.upper.aircraft_yaw_torque_Nm + result.lower.aircraft_yaw_torque_Nm,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            result.upper.total_thrust_N,
            result.lower.total_thrust_N,
            delta=0.2,
        )

    def test_coaxial_torque_balance_satisfies_weight_and_zero_yaw(self):
        result = solve_coaxial_hover(
            CoaxialSpec(
                upper_rotor=self.rotor,
                spacing_ratio=0.25,
                trim_mode="torque_balance",
            ),
            OperatingPoint(target_thrust_N=20.0, trim_mode="target_thrust"),
            polar_provider=self.polar_provider,
            settings=self.settings,
        )

        self.assertAlmostEqual(20.0, result.total_thrust_N, delta=0.2)
        self.assertAlmostEqual(0.0, result.net_aircraft_yaw_torque_Nm, delta=1e-3)
        self.assertAlmostEqual(
            result.upper.total_torque_Nm,
            result.lower.total_torque_Nm,
            delta=1e-3,
        )
        self.assertGreater(result.upper.total_thrust_N, result.lower.total_thrust_N)

    def test_coaxial_torque_balance_honors_lower_rotor_speed_ratio(self):
        operating_point = OperatingPoint(target_thrust_N=20.0, trim_mode="target_thrust")
        baseline = solve_coaxial_hover(
            CoaxialSpec(
                upper_rotor=self.rotor,
                spacing_ratio=0.25,
                trim_mode="torque_balance",
            ),
            operating_point,
            polar_provider=self.polar_provider,
            settings=self.settings,
        )
        faster_lower = solve_coaxial_hover(
            CoaxialSpec(
                upper_rotor=self.rotor,
                spacing_ratio=0.25,
                trim_mode="torque_balance",
                lower_rotor_speed_ratio=1.10,
            ),
            operating_point,
            polar_provider=self.polar_provider,
            settings=self.settings,
        )

        self.assertAlmostEqual(20.0, faster_lower.total_thrust_N, delta=0.2)
        self.assertAlmostEqual(0.0, faster_lower.net_aircraft_yaw_torque_Nm, delta=1e-3)
        self.assertAlmostEqual(
            self.rotor.headspeed_rpm * 1.10,
            CoaxialSpec(upper_rotor=self.rotor, lower_rotor_speed_ratio=1.10).resolved_lower_rotor.headspeed_rpm,
        )
        self.assertLess(
            faster_lower.lower.collective_pitch_deg,
            baseline.lower.collective_pitch_deg,
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

    def test_xfoil_provider_caps_requested_alpha_to_18_degrees(self):
        calls = {}

        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60, **kwargs):
                self.new_polar = new_polar
                self.timeout = timeout

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                calls["alpha_max_deg"] = alpha_max_deg
                return True

            def read_polar(self):
                return [
                    [-10.0, -0.8, 0.04, 0.0, 0.0],
                    [0.0, 0.0, 0.01, 0.0, 0.0],
                    [18.0, 1.0, 0.05, 0.0, 0.0],
                ]

        with patch("pycopter.polars.Xfoil", FakeXfoil):
            provider = XfoilPolarProvider(alpha_max_deg=25.0)
            coeffs = provider.get_coefficients("naca0012", 20.0, 100000.0, 0.1)

        self.assertLessEqual(calls["alpha_max_deg"], 18.0)
        self.assertEqual(18.0, coeffs.alpha_deg)
        self.assertTrue(coeffs.alpha_clamped)

    def test_xfoil_provider_never_rounds_reynolds_to_zero(self):
        calls = {}

        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60, **kwargs):
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

            def __init__(self, new_polar=True, timeout=60, **kwargs):
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

            def __init__(self, new_polar=True, timeout=60, **kwargs):
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

    def test_xfoil_provider_reuses_disk_cache_when_new_polar_is_true(self):
        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60, **kwargs):
                self.new_polar = new_polar
                self.timeout = timeout
                self.repo_root = Path.cwd()
                self.output_path = self.repo_root / "data" / "XFOIL6.99" / "polar.txt"
                self.output_path_for_xfoil = "data/XFOIL6.99/polar.txt"

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                raise AssertionError("Existing cache files should be reused before XFOIL runs")

            def read_polar(self):
                import numpy as np

                return np.genfromtxt(self.output_path, skip_header=12)

        with TemporaryDirectory() as tempdir:
            provider = XfoilPolarProvider(new_polar=True, cache_directory=tempdir)
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

    def test_xfoil_provider_does_not_launch_when_cache_missing_and_new_polar_is_false(self):
        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60, **kwargs):
                self.new_polar = new_polar
                self.timeout = timeout
                self.repo_root = Path.cwd()
                self.output_path = self.repo_root / "data" / "XFOIL6.99" / "polar.txt"
                self.output_path_for_xfoil = "data/XFOIL6.99/polar.txt"

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                raise AssertionError("XFOIL must not launch when new_polar is false")

            def read_polar(self):
                raise AssertionError("No cache file exists to read")

        with TemporaryDirectory() as tempdir:
            provider = XfoilPolarProvider(new_polar=False, cache_directory=tempdir)
            with patch("pycopter.polars.Xfoil", FakeXfoil):
                with self.assertRaisesRegex(RuntimeError, "XFOIL launch is disabled"):
                    provider.get_coefficients("naca0012", 5.0, 100000.0, 0.1)

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


class TestElementSpacing(unittest.TestCase):
    def setUp(self):
        self.rotor = RotorSpec.from_uniform_blade(
            airfoil="naca0012",
            num_blades=2,
            chord_m=0.035,
            rotor_diameter_m=0.7,
            headspeed_rpm=2500.0,
            washout_deg=-6.0,
            root_cutout_ratio=0.12,
        )
        self.provider = LinearPolarProvider(lift_slope_per_rad=5.7, cd0=0.012)
        self.operating_point = OperatingPoint(target_thrust_N=9.81, trim_mode="target_thrust")

    def test_uniform_spacing_is_still_the_default(self):
        self.assertEqual("uniform", HoverSolverSettings().element_spacing)

    def test_cosine_spacing_clusters_elements_at_the_root_and_the_tip(self):
        settings = HoverSolverSettings(blade_element_count=20, element_spacing="cosine")
        solver = HoverSolver(self.provider, settings)

        result = solver.solve(self.rotor, self.operating_point)
        widths = [load.dr_m for load in result.element_loads]

        self.assertEqual(20, len(widths))
        self.assertLess(widths[0], widths[len(widths) // 2])
        self.assertLess(widths[-1], widths[len(widths) // 2])
        self.assertAlmostEqual(
            sum(widths),
            self.rotor.radius_m - self.rotor.root_cutout_ratio * self.rotor.radius_m,
            places=6,
        )

    def test_cosine_spacing_reaches_the_same_trim_as_uniform(self):
        results = {}
        for spacing in ("uniform", "cosine"):
            solver = HoverSolver(
                self.provider,
                HoverSolverSettings(blade_element_count=60, element_spacing=spacing),
            )
            results[spacing] = solver.solve(self.rotor, self.operating_point)

        self.assertAlmostEqual(
            results["uniform"].total_thrust_N, results["cosine"].total_thrust_N, delta=0.05
        )
        self.assertAlmostEqual(
            results["uniform"].power_W, results["cosine"].power_W, delta=1.0
        )

    def test_invalid_spacing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "element_spacing"):
            HoverSolverSettings(element_spacing="linear")


class TestInterferenceSweep(unittest.TestCase):
    def setUp(self):
        upper = RotorSpec.from_uniform_blade(
            airfoil="naca0012",
            num_blades=2,
            chord_m=0.035,
            rotor_diameter_m=0.7,
            headspeed_rpm=2500.0,
            washout_deg=-6.0,
            root_cutout_ratio=0.12,
            name="upper",
        )
        lower = RotorSpec.from_uniform_blade(
            airfoil="naca0012",
            num_blades=2,
            chord_m=0.035,
            rotor_diameter_m=0.7,
            headspeed_rpm=2500.0,
            washout_deg=-6.0,
            root_cutout_ratio=0.12,
            name="lower",
            rotation_direction=-1,
        )
        self.spec = CoaxialSpec(
            upper_rotor=upper, lower_rotor=lower, trim_mode="equal_thrust"
        )
        self.operating_point = OperatingPoint(gross_mass_kg=1.0, trim_mode="target_thrust")
        self.settings = HoverSolverSettings(blade_element_count=8)
        self.provider = LinearPolarProvider(lift_slope_per_rad=5.7, cd0=0.012)

    def test_sweep_reports_progress_for_every_point(self):
        seen = []

        sweep = sweep_interference_loss(
            self.spec,
            self.operating_point,
            [0.1, 0.3, 0.5],
            polar_provider=self.provider,
            settings=self.settings,
            progress=lambda index, total, spacing, point: seen.append((index, total, spacing)),
        )

        self.assertEqual(3, len(sweep.points))
        self.assertEqual(3, len(sweep.solved_points))
        self.assertFalse(sweep.cancelled)
        self.assertEqual([(1, 3, 0.1), (2, 3, 0.3), (3, 3, 0.5)], seen)
        self.assertEqual([0.1, 0.3, 0.5], [point.z_over_R for point in sweep.points])

    def test_sweep_uses_the_swept_spacing_for_each_point(self):
        sweep = sweep_interference_loss(
            self.spec,
            self.operating_point,
            [0.1, 1.2],
            polar_provider=self.provider,
            settings=self.settings,
        )

        losses = [point.result.interference_loss_ratio for point in sweep.solved_points]
        self.assertNotAlmostEqual(losses[0], losses[1], places=4)

    def test_cancelling_returns_the_points_solved_so_far(self):
        solved = []

        sweep = sweep_interference_loss(
            self.spec,
            self.operating_point,
            [0.1, 0.3, 0.5, 0.7],
            polar_provider=self.provider,
            settings=self.settings,
            progress=lambda index, total, spacing, point: solved.append(index),
            cancel=lambda: len(solved) >= 2,
        )

        self.assertTrue(sweep.cancelled)
        self.assertEqual(2, len(sweep.points))
        self.assertEqual(4, sweep.requested_points)
        self.assertEqual(2, len(sweep.solved_points))

    def test_a_failed_point_is_recorded_without_aborting_the_sweep(self):
        real_solve = sweep_interference_loss.__globals__["solve_coaxial_hover"]
        calls = {"count": 0}

        def flaky(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("did not converge")
            return real_solve(*args, **kwargs)

        with patch("pycopter.bemt.solve_coaxial_hover", side_effect=flaky):
            sweep = sweep_interference_loss(
                self.spec,
                self.operating_point,
                [0.1, 0.3, 0.5],
                polar_provider=self.provider,
                settings=self.settings,
            )

        self.assertEqual(3, len(sweep.points))
        self.assertEqual(2, len(sweep.solved_points))
        self.assertEqual(1, len(sweep.failed_points))
        self.assertIn("did not converge", sweep.failed_points[0].error)


class TestPolarBinProvenance(unittest.TestCase):
    def _fake_xfoil(self, generated):
        class FakeXfoil:
            error_message = ""

            def __init__(self, new_polar=True, timeout=60, **kwargs):
                self.new_polar = new_polar
                self.repo_root = Path.cwd()
                self.output_path = self.repo_root / "polar.txt"
                self.output_path_for_xfoil = "polar.txt"

            def simulate(self, airfoil, mach, reynolds, alpha_min_deg, alpha_max_deg):
                generated.append((airfoil, reynolds, mach))
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(
                    "\n" * 12 + "-8 -0.8 0.04 0 0\n0 0.0 0.01 0 0\n15 1.0 0.05 0 0\n",
                    encoding="utf-8",
                )
                return True

            def read_polar(self):
                import numpy as np

                return np.genfromtxt(self.output_path, skip_header=12)

        return FakeXfoil

    def test_generated_and_cached_bins_are_reported_separately(self):
        generated = []
        with TemporaryDirectory() as tempdir:
            with patch("pycopter.polars.Xfoil", self._fake_xfoil(generated)):
                provider = XfoilPolarProvider(cache_directory=tempdir)
                provider.get_coefficients("naca0012", 5.0, 100000.0, 0.1)
                report = provider.polar_bin_report()

                self.assertEqual(1, len(report))
                self.assertEqual("generated", report[0]["status"])
                self.assertEqual(1, report[0]["lookups"])
                self.assertEqual(1, provider.cache_bin_count())
                self.assertEqual(1, provider.cache_file_count())

                reader = XfoilPolarProvider(cache_directory=tempdir)
                reader.get_coefficients("naca0012", 5.0, 100000.0, 0.1)

                self.assertEqual("cache", reader.polar_bin_report()[0]["status"])

    def test_out_of_range_requests_are_flagged_as_substituted(self):
        generated = []
        with TemporaryDirectory() as tempdir:
            with patch("pycopter.polars.Xfoil", self._fake_xfoil(generated)):
                provider = XfoilPolarProvider(
                    cache_directory=tempdir, max_reynolds=500000.0, max_mach=0.4
                )
                provider.get_coefficients("naca0012", 5.0, 2_000_000.0, 0.9)

        report = provider.polar_bin_report()
        self.assertEqual(1, len(report))
        self.assertTrue(report[0]["substituted"])
        self.assertEqual("substituted", report[0]["status"])
        self.assertEqual("generated", report[0]["source"])

    def test_in_range_binning_is_not_a_substitution(self):
        generated = []
        with TemporaryDirectory() as tempdir:
            with patch("pycopter.polars.Xfoil", self._fake_xfoil(generated)):
                provider = XfoilPolarProvider(cache_directory=tempdir)
                # 137,000 rounds into the 100,000-wide Re bin; that is the
                # documented binning scheme, not a substituted condition.
                provider.get_coefficients("naca0012", 5.0, 137000.0, 0.12)

        self.assertFalse(provider.polar_bin_report()[0]["substituted"])

    def test_substituted_bins_reach_the_hover_result(self):
        generated = []
        with TemporaryDirectory() as tempdir:
            with patch("pycopter.polars.Xfoil", self._fake_xfoil(generated)):
                provider = XfoilPolarProvider(
                    cache_directory=tempdir, max_reynolds=1000.0, max_mach=0.01
                )
                rotor = RotorSpec.from_uniform_blade(
                    airfoil="naca0012",
                    num_blades=2,
                    chord_m=0.035,
                    rotor_diameter_m=0.7,
                    headspeed_rpm=2500.0,
                    root_cutout_ratio=0.12,
                )
                solver = HoverSolver(provider, HoverSolverSettings(blade_element_count=6))
                result = solver.solve_fixed_collective(
                    rotor, OperatingPoint(trim_mode="fixed_collective"), 6.0
                )

        self.assertTrue(result.substituted_polar_bins)
        self.assertTrue(
            any("substituted polar bin" in warning for warning in result.warnings)
        )


class TestXfoilSweepSettings(unittest.TestCase):
    def test_alpha_step_controls_the_generated_sweep(self):
        xfoil = Xfoil(alpha_step_deg=0.5)

        points = xfoil.alpha_points(-3.0, -1.0)

        self.assertEqual([-3.0, -2.5, -2.0, -1.5, -1.0], points)

    def test_alpha_sweep_never_exceeds_the_xfoil_ceiling(self):
        xfoil = Xfoil(alpha_step_deg=4.0)

        points = xfoil.alpha_points(0.0, 25.0)

        self.assertLessEqual(max(points), 18.0)
        self.assertEqual(0.0, points[0])

    def test_alpha_step_and_n_crit_change_the_cache_key(self):
        with TemporaryDirectory() as tempdir:
            coarse = XfoilPolarProvider(cache_directory=tempdir, alpha_step_deg=1.0)
            fine = XfoilPolarProvider(cache_directory=tempdir, alpha_step_deg=0.25)
            turbulent = XfoilPolarProvider(cache_directory=tempdir, n_crit=4.0)

            paths = {
                provider._cache_file_path("naca0012", 100000.0, 0.1)
                for provider in (coarse, fine, turbulent)
            }

        self.assertEqual(3, len(paths))


if __name__ == "__main__":
    unittest.main()
