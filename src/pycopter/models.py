"""Typed data contracts for rotor geometry, operating points, and results."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import numpy as np


GRAVITY_M_S2 = 9.81

TrimMode = Literal["target_thrust", "fixed_collective"]
LossModel = Literal["prandtl", "none"]
CoaxialTrimMode = Literal["torque_balance", "equal_thrust", "equal_collective"]


@dataclass(frozen=True)
class BladeStation:
    """Local blade geometry at a nondimensional radial station."""

    r_over_R: float
    chord_m: float
    twist_deg: float
    airfoil: str | None = None
    pitch_axis_frac: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.r_over_R <= 1.0:
            raise ValueError("Blade station r_over_R must be in (0, 1].")
        if self.chord_m <= 0:
            raise ValueError("Blade station chord_m must be positive.")
        if not 0.0 <= self.pitch_axis_frac <= 1.0:
            raise ValueError("Blade station pitch_axis_frac must be in [0, 1].")


@dataclass
class RotorSpec:
    """Complete rotor geometry and speed definition for BEMT calculations."""

    airfoil: str
    num_blades: int
    rotor_diameter_m: float
    stations: list[BladeStation]
    headspeed_rpm: float | None = None
    tip_speed_mach: float | None = None
    root_cutout_ratio: float = 0.0
    speed_of_sound_m_s: float = 343.0
    name: str = "rotor"
    rotation_direction: int = 1

    def __post_init__(self) -> None:
        if self.num_blades < 1:
            raise ValueError("num_blades must be at least 1.")
        if self.rotor_diameter_m <= 0:
            raise ValueError("rotor_diameter_m must be positive.")
        if not 0.0 <= self.root_cutout_ratio < 1.0:
            raise ValueError("root_cutout_ratio must be in [0, 1).")
        if self.speed_of_sound_m_s <= 0:
            raise ValueError("speed_of_sound_m_s must be positive.")
        if self.rotation_direction not in (-1, 1):
            raise ValueError("rotation_direction must be -1 or 1.")
        if not self.stations:
            raise ValueError("At least one blade station is required.")

        sorted_stations = sorted(self.stations, key=lambda station: station.r_over_R)
        radii = [station.r_over_R for station in sorted_stations]
        if radii != [station.r_over_R for station in self.stations]:
            raise ValueError("Blade stations must be monotonic ascending by r_over_R.")
        if len(set(radii)) != len(radii):
            raise ValueError("Blade station r_over_R values must be unique.")

        self.stations = sorted_stations

        if self.headspeed_rpm is None and self.tip_speed_mach is None:
            raise ValueError("Either headspeed_rpm or tip_speed_mach must be provided.")
        if self.headspeed_rpm is None:
            tip_speed_m_s = self.tip_speed_mach * self.speed_of_sound_m_s
            self.headspeed_rpm = tip_speed_m_s / self.radius_m * 30.0 / np.pi
        if self.tip_speed_mach is None:
            self.tip_speed_mach = self.tip_speed_m_s / self.speed_of_sound_m_s
        if self.headspeed_rpm <= 0:
            raise ValueError("headspeed_rpm must be positive.")
        if self.tip_speed_mach <= 0:
            raise ValueError("tip_speed_mach must be positive.")

    @classmethod
    def from_uniform_blade(
        cls,
        airfoil: str,
        num_blades: int,
        chord_m: float,
        rotor_diameter_m: float,
        headspeed_rpm: float | None = None,
        tip_speed_mach: float | None = None,
        washout_deg: float = 0.0,
        root_cutout_ratio: float = 0.0,
        station_count: int = 4,
        speed_of_sound_m_s: float = 343.0,
        name: str = "rotor",
        rotation_direction: int = 1,
    ) -> "RotorSpec":
        """Create a radial station table from the legacy chord/washout inputs."""
        if station_count < 2:
            raise ValueError("station_count must be at least 2.")
        first_station = max(root_cutout_ratio, 0.02)
        radii = np.linspace(first_station, 1.0, station_count)
        stations = []
        for r_over_R in radii:
            span_fraction = (r_over_R - first_station) / (1.0 - first_station)
            stations.append(
                BladeStation(
                    r_over_R=float(r_over_R),
                    chord_m=chord_m,
                    twist_deg=float(washout_deg * span_fraction),
                    airfoil=airfoil,
                )
            )
        return cls(
            airfoil=airfoil,
            num_blades=num_blades,
            rotor_diameter_m=rotor_diameter_m,
            stations=stations,
            headspeed_rpm=headspeed_rpm,
            tip_speed_mach=tip_speed_mach,
            root_cutout_ratio=root_cutout_ratio,
            speed_of_sound_m_s=speed_of_sound_m_s,
            name=name,
            rotation_direction=rotation_direction,
        )

    @property
    def radius_m(self) -> float:
        return self.rotor_diameter_m / 2.0

    @property
    def root_radius_m(self) -> float:
        return self.root_cutout_ratio * self.radius_m

    @property
    def omega_rad_s(self) -> float:
        return self.headspeed_rpm * np.pi / 30.0

    @property
    def tip_speed_m_s(self) -> float:
        return self.omega_rad_s * self.radius_m

    @property
    def disk_area_m2(self) -> float:
        return np.pi * (self.radius_m**2 - self.root_radius_m**2)

    @property
    def solidity(self) -> float:
        blade_start = max(self.root_cutout_ratio, self.stations[0].r_over_R)
        samples = np.linspace(blade_start, 1.0, 80)
        chords = np.array([self.chord_at(r_over_R) for r_over_R in samples])
        mean_chord = float(np.trapezoid(chords, samples) / (samples[-1] - samples[0]))
        return self.num_blades * mean_chord / (np.pi * self.radius_m)

    def chord_at(self, r_over_R: float) -> float:
        return self._interp_station_value(r_over_R, "chord_m")

    def twist_at(self, r_over_R: float) -> float:
        return self._interp_station_value(r_over_R, "twist_deg")

    def pitch_axis_at(self, r_over_R: float) -> float:
        return self._interp_station_value(r_over_R, "pitch_axis_frac")

    def airfoil_at(self, r_over_R: float) -> str:
        if len(self.stations) == 1:
            return self.stations[0].airfoil or self.airfoil

        nearest = min(self.stations, key=lambda station: abs(station.r_over_R - r_over_R))
        return nearest.airfoil or self.airfoil

    def _interp_station_value(self, r_over_R: float, attr_name: str) -> float:
        radii = np.array([station.r_over_R for station in self.stations])
        values = np.array([getattr(station, attr_name) for station in self.stations])
        return float(np.interp(r_over_R, radii, values))


@dataclass(frozen=True)
class OperatingPoint:
    """Air state and trim request for a rotor calculation."""

    density_kg_m3: float = 1.225
    kinematic_viscosity_m2_s: float = 1.5e-5
    speed_of_sound_m_s: float = 343.0
    target_thrust_N: float | None = None
    gross_mass_kg: float | None = None
    collective_pitch_deg: float = 8.0
    trim_mode: TrimMode = "target_thrust"

    def __post_init__(self) -> None:
        if self.density_kg_m3 <= 0:
            raise ValueError("density_kg_m3 must be positive.")
        if self.kinematic_viscosity_m2_s <= 0:
            raise ValueError("kinematic_viscosity_m2_s must be positive.")
        if self.speed_of_sound_m_s <= 0:
            raise ValueError("speed_of_sound_m_s must be positive.")
        if self.trim_mode not in ("target_thrust", "fixed_collective"):
            raise ValueError("trim_mode must be 'target_thrust' or 'fixed_collective'.")
        if self.target_thrust_N is not None and self.target_thrust_N <= 0:
            raise ValueError("target_thrust_N must be positive when provided.")
        if self.gross_mass_kg is not None and self.gross_mass_kg <= 0:
            raise ValueError("gross_mass_kg must be positive when provided.")

    @property
    def required_thrust_N(self) -> float:
        if self.target_thrust_N is not None:
            return self.target_thrust_N
        if self.gross_mass_kg is not None:
            return self.gross_mass_kg * GRAVITY_M_S2
        raise ValueError("target_thrust_N or gross_mass_kg is required for target_thrust trim.")


@dataclass(frozen=True)
class HoverSolverSettings:
    """Numerical settings and low-order correction switches for hover BEMT."""

    blade_element_count: int = 60
    min_collective_deg: float = -5.0
    max_collective_deg: float = 20.0
    collective_tolerance_deg: float = 0.01
    thrust_tolerance: float = 1e-3
    max_trim_iterations: int = 80
    tip_loss_model: LossModel = "prandtl"
    root_loss_model: LossModel = "prandtl"
    induced_power_factor: float = 1.05
    min_loss_factor: float = 1e-3

    def __post_init__(self) -> None:
        if self.blade_element_count < 2:
            raise ValueError("blade_element_count must be at least 2.")
        if self.max_collective_deg <= self.min_collective_deg:
            raise ValueError("max_collective_deg must be greater than min_collective_deg.")
        if self.collective_tolerance_deg <= 0:
            raise ValueError("collective_tolerance_deg must be positive.")
        if self.thrust_tolerance <= 0:
            raise ValueError("thrust_tolerance must be positive.")
        if self.max_trim_iterations < 1:
            raise ValueError("max_trim_iterations must be positive.")
        if self.tip_loss_model not in ("prandtl", "none"):
            raise ValueError("tip_loss_model must be 'prandtl' or 'none'.")
        if self.root_loss_model not in ("prandtl", "none"):
            raise ValueError("root_loss_model must be 'prandtl' or 'none'.")
        if self.induced_power_factor < 1.0:
            raise ValueError("induced_power_factor must be at least 1.0.")
        if not 0.0 < self.min_loss_factor <= 1.0:
            raise ValueError("min_loss_factor must be in (0, 1].")


@dataclass(frozen=True)
class ElementLoad:
    """Per-blade load result for one radial blade element."""

    r_m: float
    r_over_R: float
    dr_m: float
    chord_m: float
    twist_deg: float
    collective_deg: float
    phi_deg: float
    alpha_deg: float
    reynolds: float
    mach: float
    cl: float
    cd: float
    cm: float
    alpha_clamped: bool
    loss_factor: float
    induced_velocity_m_s: float
    external_axial_velocity_m_s: float
    dL_N: float
    dD_N: float
    dT_N: float
    dQ_Nm: float
    dP_W: float
    normal_force_N_per_m: float
    tangential_force_N_per_m: float
    pitch_moment_Nm: float


@dataclass(frozen=True)
class HoverResult:
    """Integrated hover result plus per-blade radial loads."""

    collective_pitch_deg: float
    total_thrust_N: float
    per_blade_thrust_N: float
    total_torque_Nm: float
    per_blade_torque_Nm: float
    aircraft_yaw_torque_Nm: float
    power_W: float
    induced_power_W: float
    profile_power_W: float
    ideal_power_W: float
    figure_of_merit: float
    mean_induced_velocity_m_s: float
    ct: float
    cp: float
    solidity: float
    mean_loss_factor: float
    root_flap_bending_moment_Nm_per_blade: float
    root_lag_moment_Nm_per_blade: float
    aerodynamic_pitching_moment_Nm_per_blade: float
    element_loads: list[ElementLoad] = field(default_factory=list)

    def load_table(self) -> list[dict[str, float]]:
        """Return element loads as GUI/CSV-friendly dictionaries."""
        return [load.__dict__.copy() for load in self.element_loads]


@dataclass(frozen=True)
class CoaxialSpec:
    """Definition for stacked contra-rotating hover calculations."""

    upper_rotor: RotorSpec
    lower_rotor: RotorSpec | None = None
    spacing_ratio: float = 0.25
    trim_mode: CoaxialTrimMode = "torque_balance"
    lower_collective_offset_deg: float = 0.0
    lower_rotor_speed_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.spacing_ratio <= 0:
            raise ValueError("spacing_ratio must be positive.")
        if self.lower_rotor_speed_ratio <= 0:
            raise ValueError("lower_rotor_speed_ratio must be positive.")
        if self.trim_mode not in ("torque_balance", "equal_thrust", "equal_collective"):
            raise ValueError(
                "trim_mode must be 'torque_balance', 'equal_thrust', or 'equal_collective'."
            )

    @property
    def resolved_lower_rotor(self) -> RotorSpec:
        lower = self.lower_rotor or replace(
            self.upper_rotor,
            name="lower",
            rotation_direction=-self.upper_rotor.rotation_direction,
        )
        if self.lower_rotor_speed_ratio == 1.0 and self.lower_rotor is not None:
            return lower
        return replace(
            lower,
            headspeed_rpm=self.upper_rotor.headspeed_rpm * self.lower_rotor_speed_ratio,
            tip_speed_mach=None,
        )


@dataclass(frozen=True)
class CoaxialHoverResult:
    """Integrated coaxial hover result with upper/lower rotor details."""

    upper: HoverResult
    lower: HoverResult
    isolated_upper: HoverResult
    isolated_lower: HoverResult
    total_thrust_N: float
    total_power_W: float
    net_aircraft_yaw_torque_Nm: float
    interference_power_delta_W: float
    interference_loss_ratio: float
    lower_external_velocity_mean_m_s: float
    wake_radius_m: float
    wake_velocity_m_s: float


ExternalVelocityProfile = Callable[[float], float]
