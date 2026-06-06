"""Legacy-compatible entry points for PyCopter rotor calculations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bemt import HoverSolver
from .models import HoverSolverSettings, OperatingPoint, RotorSpec
from .polars import PolarProvider, XfoilPolarProvider
from .utils import walds_solver


class _LegacyPolarView:
    """Compatibility adapter for old GUI code that expects rotor.polar."""

    def __init__(self, rotor: "Rotor"):
        self._rotor = rotor
        self.reynolds = (
            rotor.tip_speed
            * 0.5
            * rotor.chord
            / 1.5e-5
        )
        self.mach = rotor.tip_speed_mach * 0.5

    def get_polar(self, alfa):
        coeffs = self._rotor.polar_provider.get_coefficients(
            self._rotor.airfoil,
            float(alfa),
            self.reynolds,
            self.mach,
        )
        return coeffs.cl, coeffs.cd


class Rotor:
    """
    Backward-compatible facade for the station-based BEMT solver.

    Existing GUI code can still initialize a rectangular/twisted blade rotor and
    call hover()/forward_flight(). New code should prefer RotorSpec,
    OperatingPoint, and HoverSolver directly because they expose blade station
    loads and calculation settings.
    """

    def __init__(
        self,
        airfoil="naca23012",
        num_blades=5,
        chord=0.52,
        rotor_diameter=21.29,
        tip_speed_mach=0.624,
        washout=-8,
        rotor_root_cutout=0.01,
        new_polar=True,
        polar_provider: PolarProvider | None = None,
        solver_settings: HoverSolverSettings | None = None,
        headspeed_rpm: float | None = None,
        station_count: int = 4,
        speed_of_sound_m_s: float = 343.0,
    ):
        self.airfoil = airfoil
        self.num_blades = num_blades
        self.chord = chord
        self.rotor_diameter = rotor_diameter
        self.tip_speed_mach = tip_speed_mach
        self.washout = washout
        self.rotor_root_cutout = rotor_root_cutout
        self.speed_of_sound = speed_of_sound_m_s

        self.spec = RotorSpec.from_uniform_blade(
            airfoil=airfoil,
            num_blades=num_blades,
            chord_m=chord,
            rotor_diameter_m=rotor_diameter,
            headspeed_rpm=headspeed_rpm,
            tip_speed_mach=None if headspeed_rpm is not None else tip_speed_mach,
            washout_deg=washout,
            root_cutout_ratio=rotor_root_cutout,
            station_count=station_count,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )
        self.settings = solver_settings or HoverSolverSettings()
        self.polar_provider = polar_provider or XfoilPolarProvider(new_polar=new_polar)
        self.solver = HoverSolver(self.polar_provider, self.settings)

        self.r = self.spec.radius_m
        self.omega = self.spec.omega_rad_s
        self.omega_mach = self.omega / self.speed_of_sound
        self.rpm = self.spec.headspeed_rpm
        self.tip_speed = self.spec.tip_speed_m_s
        self.tip_speed_mach = self.spec.tip_speed_mach
        self.rotor_disk_area = self.spec.disk_area_m2
        self.solidity = self.spec.solidity
        self.polar = _LegacyPolarView(self)
        self.is_hovered = False

        print("\nInitializing rotor...")
        print(
            "Tip Speed:",
            self.tip_speed,
            "[m/s] | Rotor Disk Area:",
            self.rotor_disk_area,
            "[m2] | Solidity:",
            self.solidity,
        )

    def calculate(self, density=1.225, theta=8):
        """Deprecated: use hover(trim_mode='fixed_collective') through HoverSolver."""
        operating_point = OperatingPoint(
            density_kg_m3=density,
            collective_pitch_deg=theta,
            trim_mode="fixed_collective",
        )
        return self.solver.solve(self.spec, operating_point)

    def ige(self, thrust: float, rotor_height: float):
        """Return thrust corrected by the classic hover in-ground-effect factor."""
        return thrust / (1 - (self.r**2 / (16 * rotor_height**2)))

    def hover(self, weight=13000, density=1.225, n=None):
        """
        Calculate hover trim for a target aircraft mass in kg.

        The legacy `n` argument now maps to BEMT blade element count when
        provided. Results are exposed both as old attributes and as
        self.hover_result.
        """
        solver = self.solver
        if n is not None and n != self.settings.blade_element_count:
            settings = replace(self.settings, blade_element_count=int(n))
            solver = HoverSolver(self.polar_provider, settings)

        print("\nCalculating Hover Conditions...")
        operating_point = OperatingPoint(gross_mass_kg=weight, density_kg_m3=density)
        self.hover_result = solver.solve(self.spec, operating_point)
        self._apply_hover_result(self.hover_result)
        self.is_hovered = True

        print(
            f"Theta = {self.theta} deg | "
            f"Induced Velocity= {self.hover_induced_vel:.3f}[m/s] | "
            f"Thrust = {self.hover_thrust / 9.81:.3f}[kg]"
        )
        print(
            "SHP Induced:",
            self.hover_power_induced * 0.00134102209,
            "| SHP Profile:",
            self.hover_power_profile * 0.00134102209,
            "| SHP Total:",
            self.hover_power_total * 0.00134102209,
        )
        print(
            "Coeffs:",
            self.ct,
            self.cp,
            "| Merit:",
            self.merit,
            "| Tip Loss:",
            self.tip_loss,
        )

    def forward_flight(self, velocity, density=1.225, flat_plate_area=3.5):
        """
        Legacy low-order forward-flight estimate.

        This remains intentionally secondary. It uses the new hover power and
        mean profile data, but still lacks trim, flapping, and stall modeling.
        """
        if not isinstance(velocity, (float, int)):
            raise ValueError("Forward flight velocity must be a numeric singleton.")
        if not self.is_hovered:
            print("Running hover calculations first...")
            self.hover()

        print("\nCalculating Forward Flight Conditions...")
        self.alfa = 0
        advance_ratio = velocity / (self.omega * self.r)
        self.downwash_velocity_ratio = walds_solver(
            velocity,
            self.hover_induced_vel,
            self.alfa,
        )

        self.power_induced = max(
            0.0,
            self.downwash_velocity_ratio * self.hover_power_induced,
        )
        self.drag_induced = self.power_induced / max(2 * self.hover_induced_vel, 1e-9)

        self.power_profile = self.hover_power_profile * (1 + 4.65 * advance_ratio**2)
        self.drag_profile = self.power_profile / max(self.omega * self.r, 1e-9)
        self.body_drag = density * velocity**2 * flat_plate_area / 2
        self.power_parasite = self.body_drag * velocity
        self.power_total = self.power_induced + self.power_profile + self.power_parasite
        self.horsepower_total = self.power_total * 0.00134102209

        print(
            "SHP Induced:",
            self.power_induced * 0.00134102209,
            "| SHP Profile:",
            self.power_profile * 0.00134102209,
            "| SHP Parasite:",
            self.power_parasite * 0.00134102209,
            "| HP Total:",
            self.horsepower_total,
        )

    def plot(self):
        print("\nPlot debug.")
        print("Ct/sigma:", self.ct / self.solidity)

    def _apply_hover_result(self, result):
        self.theta = result.collective_pitch_deg
        self.hover_induced_vel = result.mean_induced_velocity_m_s
        self.hover_thrust = result.total_thrust_N
        self.ct = result.ct
        self.cp = result.cp
        self.tip_loss = result.mean_loss_factor
        self.hover_power_induced = result.induced_power_W
        self.hover_power_profile = result.profile_power_W
        self.hover_power_total = result.power_W
        self.merit = result.figure_of_merit
        self.merit_max = result.figure_of_merit

        positive_loads = [load for load in result.element_loads if load.dT_N > 0.0]
        if positive_loads:
            weights = np.array([load.dT_N for load in positive_loads])
            self.cl_mean = float(np.average([load.cl for load in positive_loads], weights=weights))
            self.cd_mean = float(np.average([load.cd for load in positive_loads], weights=weights))
            self.alfa = float(np.average([load.alpha_deg for load in positive_loads], weights=weights))
        else:
            self.cl_mean = 0.0
            self.cd_mean = 0.0
            self.alfa = 0.0


class Engine:
    """Legacy placeholder for future propulsion models."""

    def __init__(self, sfc, test_wg, test_shp, test_wf):
        self.sfc = sfc
        self.test_wg = test_wg
        self.test_shp = test_shp
        self.test_wf = test_wf


class Body:
    """Legacy aircraft body container."""

    def __init__(self, dry_weight, fuel_capacity, flat_plate_area):
        self.dry_weight = dry_weight
        self.fuel_capacity = fuel_capacity
        self.fpa = flat_plate_area


class Copter:
    """Legacy placeholder for an aircraft-level model."""

    def __init__(self):
        pass


class RotorImperial(Rotor):
    """Legacy placeholder for imperial-unit wrappers."""

    def __init__(self, a):
        print(a)


class Optimizer:
    """Legacy placeholder for rotor optimization workflows."""

    def __init__(self):
        pass


if __name__ == "__main__":
    rotor = Rotor(new_polar=False)
    rotor.hover(4)
    rotor.forward_flight(40)
    rotor.plot()
