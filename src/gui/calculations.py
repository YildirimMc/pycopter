"""GUI-facing calculation and configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pycopter import (
    BladeStation,
    CoaxialHoverResult,
    CoaxialSpec,
    HoverResult,
    HoverSolver,
    HoverSolverSettings,
    OperatingPoint,
    RotorSpec,
    solve_coaxial_hover,
)
from pycopter.polars import PolarProvider, XfoilPolarProvider
from pycopter.utils import walds_solver
from pycopter.xfoil import ensure_airfoil_coordinates, is_naca_airfoil, normalize_airfoil_name


PropulsionModel = Literal["electric", "fossil"]
RotorSystemType = Literal["single", "coaxial"]
GeometryMode = Literal["uniform", "station_table"]

CONFIG_VERSION = 1
SHP_PER_WATT = 0.00134102209
MPS_TO_KMH = 3.6


DEFAULT_STATION_ROWS: list[dict[str, Any]] = [
    {"r_over_R": 0.12, "chord_m": 0.035, "twist_deg": 10.0, "airfoil": "naca0012", "pitch_axis_frac": 0.25},
    {"r_over_R": 0.45, "chord_m": 0.035, "twist_deg": 6.6, "airfoil": "naca0012", "pitch_axis_frac": 0.25},
    {"r_over_R": 0.75, "chord_m": 0.033, "twist_deg": 3.5, "airfoil": "naca0012", "pitch_axis_frac": 0.25},
    {"r_over_R": 1.00, "chord_m": 0.030, "twist_deg": 1.0, "airfoil": "naca0012", "pitch_axis_frac": 0.25},
]


DEFAULT_CONFIG: dict[str, Any] = {
    "version": CONFIG_VERSION,
    "airfoil": "naca0012",
    "new_polars": True,
    "rotor_system_type": "single",
    "geometry_mode": "uniform",
    "num_blades": 2,
    "chord": 0.035,
    "rotor_diam": 0.70,
    "root_twist_deg": 10.0,
    "tip_twist_deg": 1.0,
    "root_cutout": 0.12,
    "headspeed_input_mode": "rpm",
    "headspeed_rpm": 2500.0,
    "tip_speed_mach": 0.0,
    "gross": 1.0,
    "density": 1.225,
    "kinematic_viscosity_m2_s": 1.5e-5,
    "trim_mode": "target_thrust",
    "collective_pitch_deg": 0.0,
    "min_collective_deg": 0.0,
    "max_collective_deg": 15.0,
    "blade_element_count": 60,
    "tip_loss_model": "prandtl",
    "root_loss_model": "prandtl",
    "induced_power_factor": 1.05,
    "polar_alpha_min_deg": -3.0,
    "polar_alpha_max_deg": 18.0,
    "xfoil_parallel_workers": 8,
    "xfoil_parallel_backend": "mpi",
    "xfoil_cache_directory": "",
    "propulsion_model": "electric",
    "battery_capacity_Wh": 100.0,
    "battery_usable_fraction": 0.80,
    "motor_efficiency": 0.85,
    "esc_efficiency": 0.95,
    "fuel_capacity_kg": 1.5,
    "specific_fuel_consumption_kg_per_kWh": 0.30,
    "transmission_loss": 0.10,
    "velocity_kmh": 40.0,
    "fpa": 0.05,
    "coaxial_spacing_ratio": 0.25,
    "coaxial_trim_mode": "equal_thrust",
    "lower_collective_offset_deg": 0.0,
    "lower_rotor_scale": 1.0,
}


@dataclass(frozen=True)
class HoverCase:
    """Current rotor hover calculation state."""

    rotor: RotorSpec
    result: HoverResult | CoaxialHoverResult
    system_type: RotorSystemType


@dataclass(frozen=True)
class ForwardFlightEstimate:
    """Legacy forward-flight estimate derived from a single-rotor hover result."""

    velocity_m_s: float
    downwash_velocity_ratio: float
    induced_power_W: float
    profile_power_W: float
    parasite_power_W: float
    total_power_W: float
    induced_drag_N: float
    profile_drag_N: float
    body_drag_N: float


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge a partial or legacy config with current defaults."""
    if "version" not in config and any(key in config for key in ("fuel_cap", "battery_cap", "tip_speed_mach")):
        config = _legacy_config_to_current(config)

    merged = DEFAULT_CONFIG.copy()
    for key, value in config.items():
        if key in merged:
            merged[key] = value
    merged["version"] = CONFIG_VERSION
    return merged


def _legacy_config_to_current(config: dict[str, Any]) -> dict[str, Any]:
    rotor_diam = float(config.get("rotor_diam", DEFAULT_CONFIG["rotor_diam"]))
    tip_speed_mach = float(config.get("tip_speed_mach", 0.0))
    headspeed_rpm = DEFAULT_CONFIG["headspeed_rpm"]
    if tip_speed_mach > 0.0 and rotor_diam > 0.0:
        headspeed_rpm = tip_speed_mach * 343.0 / (rotor_diam / 2.0) * 30.0 / pi

    mapped = {
        "airfoil": config.get("airfoil", DEFAULT_CONFIG["airfoil"]),
        "new_polars": config.get("new_polars", config.get("bNewPolars", DEFAULT_CONFIG["new_polars"])),
        "num_blades": config.get("num_blades", DEFAULT_CONFIG["num_blades"]),
        "chord": config.get("chord", DEFAULT_CONFIG["chord"]),
        "rotor_diam": rotor_diam,
        "tip_speed_mach": tip_speed_mach,
        "headspeed_input_mode": "rpm",
        "headspeed_rpm": headspeed_rpm,
        "root_twist_deg": 0.0,
        "tip_twist_deg": config.get("washout", DEFAULT_CONFIG["tip_twist_deg"]),
        "root_cutout": config.get("root_cutout", DEFAULT_CONFIG["root_cutout"]),
        "gross": config.get("gross", DEFAULT_CONFIG["gross"]),
        "density": config.get("density", DEFAULT_CONFIG["density"]),
        "transmission_loss": config.get("transmission_loss", DEFAULT_CONFIG["transmission_loss"]),
        "fuel_capacity_kg": config.get("fuel_cap", DEFAULT_CONFIG["fuel_capacity_kg"]),
        "specific_fuel_consumption_kg_per_kWh": config.get(
            "sfc",
            DEFAULT_CONFIG["specific_fuel_consumption_kg_per_kWh"],
        ),
        "battery_capacity_Wh": float(config.get("battery_cap", DEFAULT_CONFIG["battery_capacity_Wh"])) * 1000.0,
        "velocity_kmh": config.get("velocity", DEFAULT_CONFIG["velocity_kmh"]),
        "fpa": config.get("fpa", DEFAULT_CONFIG["fpa"]),
        "propulsion_model": "fossil",
    }
    return mapped


def validate_airfoil(airfoil: str) -> str:
    normalized = normalize_airfoil_name(airfoil)
    if not normalized:
        raise ValueError("Airfoil must not be empty.")
    if is_naca_airfoil(normalized):
        return normalized
    if normalized.startswith("naca"):
        raise ValueError("Invalid NACA airfoil. Only 4, 5, and 6 digit NACA profiles are supported.")

    success, _, message = ensure_airfoil_coordinates(normalized)
    if not success:
        raise ValueError(message)
    return normalized


def station_rows_from_uniform(config: dict[str, Any], station_count: int = 4) -> list[dict[str, Any]]:
    first_station = max(float(config["root_cutout"]), 0.02)
    radii = np.linspace(first_station, 1.0, station_count)
    root_twist = float(config["root_twist_deg"])
    tip_twist = float(config["tip_twist_deg"])
    rows: list[dict[str, Any]] = []
    for r_over_R in radii:
        span_fraction = (r_over_R - first_station) / (1.0 - first_station)
        rows.append(
            {
                "r_over_R": float(r_over_R),
                "chord_m": float(config["chord"]),
                "twist_deg": root_twist + (tip_twist - root_twist) * float(span_fraction),
                "airfoil": config["airfoil"],
                "pitch_axis_frac": 0.25,
            }
        )
    return rows


def build_stations(rows: list[dict[str, Any]], global_airfoil: str) -> list[BladeStation]:
    if len(rows) < 2:
        raise ValueError("Blade station table must include at least two rows.")

    stations: list[BladeStation] = []
    previous_r = -1.0
    for index, row in enumerate(rows, start=1):
        r_over_R = float(row["r_over_R"])
        if r_over_R <= previous_r:
            raise ValueError("Blade station r/R values must be monotonic ascending and unique.")
        previous_r = r_over_R

        airfoil = str(row.get("airfoil") or global_airfoil).strip().lower()
        stations.append(
            BladeStation(
                r_over_R=r_over_R,
                chord_m=float(row["chord_m"]),
                twist_deg=float(row["twist_deg"]),
                airfoil=airfoil or global_airfoil,
                pitch_axis_frac=float(row.get("pitch_axis_frac", 0.25)),
            )
        )
    return stations


def build_rotor_spec(
    config: dict[str, Any],
    station_rows: list[dict[str, Any]],
    *,
    name: str = "rotor",
    scale: float = 1.0,
    rotation_direction: int = 1,
) -> RotorSpec:
    airfoil = validate_airfoil(str(config["airfoil"]))
    headspeed_rpm = float(config["headspeed_rpm"])
    tip_speed_mach = None

    if config["geometry_mode"] == "uniform":
        uniform_rows = station_rows_from_uniform(config)
        for row in uniform_rows:
            row["chord_m"] = float(row["chord_m"]) * scale
        return RotorSpec(
            airfoil=airfoil,
            num_blades=int(config["num_blades"]),
            rotor_diameter_m=float(config["rotor_diam"]) * scale,
            stations=build_stations(uniform_rows, airfoil),
            headspeed_rpm=headspeed_rpm,
            root_cutout_ratio=float(config["root_cutout"]),
            name=name,
            rotation_direction=rotation_direction,
        )

    scaled_rows = []
    for row in station_rows:
        scaled = dict(row)
        scaled["chord_m"] = float(row["chord_m"]) * scale
        scaled_rows.append(scaled)

    return RotorSpec(
        airfoil=airfoil,
        num_blades=int(config["num_blades"]),
        rotor_diameter_m=float(config["rotor_diam"]) * scale,
        stations=build_stations(scaled_rows, airfoil),
        headspeed_rpm=headspeed_rpm,
        tip_speed_mach=tip_speed_mach,
        root_cutout_ratio=float(config["root_cutout"]),
        name=name,
        rotation_direction=rotation_direction,
    )


def build_operating_point(config: dict[str, Any]) -> OperatingPoint:
    return OperatingPoint(
        density_kg_m3=float(config["density"]),
        kinematic_viscosity_m2_s=float(config["kinematic_viscosity_m2_s"]),
        gross_mass_kg=float(config["gross"]),
        collective_pitch_deg=0.0,
        trim_mode="target_thrust",
    )


def build_solver_settings(config: dict[str, Any]) -> HoverSolverSettings:
    return HoverSolverSettings(
        blade_element_count=int(config["blade_element_count"]),
        min_collective_deg=float(config["min_collective_deg"]),
        max_collective_deg=float(config["max_collective_deg"]),
        tip_loss_model=str(config["tip_loss_model"]),
        root_loss_model=str(config["root_loss_model"]),
        induced_power_factor=float(config["induced_power_factor"]),
    )


def build_xfoil_provider(config: dict[str, Any]) -> XfoilPolarProvider:
    cache_directory = str(config.get("xfoil_cache_directory") or "").strip()
    return XfoilPolarProvider(
        new_polar=bool(config["new_polars"]),
        alpha_min_deg=float(config["polar_alpha_min_deg"]),
        alpha_max_deg=float(config["polar_alpha_max_deg"]),
        parallel_workers=int(config["xfoil_parallel_workers"]),
        parallel_backend=str(config["xfoil_parallel_backend"]),
        cache_directory=Path(cache_directory) if cache_directory else None,
    )


def run_hover_case(
    config: dict[str, Any],
    station_rows: list[dict[str, Any]],
    *,
    polar_provider: PolarProvider | None = None,
) -> HoverCase:
    config = normalize_config(config)
    rotor = build_rotor_spec(config, station_rows, name="upper")
    operating_point = build_operating_point(config)
    settings = build_solver_settings(config)
    provider = polar_provider or build_xfoil_provider(config)

    if config["rotor_system_type"] == "coaxial":
        lower = build_rotor_spec(
            config,
            station_rows,
            name="lower",
            scale=float(config["lower_rotor_scale"]),
            rotation_direction=-1,
        )
        result = solve_coaxial_hover(
            CoaxialSpec(
                upper_rotor=rotor,
                lower_rotor=lower,
                spacing_ratio=float(config["coaxial_spacing_ratio"]),
                trim_mode=str(config["coaxial_trim_mode"]),
                lower_collective_offset_deg=float(config["lower_collective_offset_deg"]),
            ),
            operating_point,
            polar_provider=provider,
            settings=settings,
        )
        return HoverCase(rotor=rotor, result=result, system_type="coaxial")

    solver = HoverSolver(provider, settings)
    return HoverCase(rotor=rotor, result=solver.solve(rotor, operating_point), system_type="single")


def primary_hover_result(case: HoverCase) -> HoverResult:
    if isinstance(case.result, CoaxialHoverResult):
        return case.result.upper
    return case.result


def total_hover_power_W(case: HoverCase) -> float:
    if isinstance(case.result, CoaxialHoverResult):
        return case.result.total_power_W
    return case.result.power_W


def total_hover_thrust_N(case: HoverCase) -> float:
    if isinstance(case.result, CoaxialHoverResult):
        return case.result.total_thrust_N
    return case.result.total_thrust_N


def electric_summary(power_W: float, config: dict[str, Any]) -> dict[str, float]:
    motor_eff = float(config["motor_efficiency"])
    esc_eff = float(config["esc_efficiency"])
    usable_fraction = float(config["battery_usable_fraction"])
    loss = float(config["transmission_loss"])
    if motor_eff <= 0 or esc_eff <= 0:
        raise ValueError("Electric efficiencies must be positive.")
    if not 0.0 <= loss < 1.0:
        raise ValueError("Transmission loss must be in [0, 1).")
    electric_input_W = power_W / ((1.0 - loss) * motor_eff * esc_eff)
    usable_energy_Wh = float(config["battery_capacity_Wh"]) * usable_fraction
    endurance_hr = usable_energy_Wh / electric_input_W if electric_input_W > 0 else 0.0
    return {
        "shaft_power_W": power_W,
        "electric_input_W": electric_input_W,
        "usable_energy_Wh": usable_energy_Wh,
        "hover_endurance_min": endurance_hr * 60.0,
    }


def fossil_summary(power_W: float, config: dict[str, Any]) -> dict[str, float]:
    loss = float(config["transmission_loss"])
    if not 0.0 <= loss < 1.0:
        raise ValueError("Transmission loss must be in [0, 1).")
    engine_input_W = power_W / (1.0 - loss)
    fuel_flow_kg_hr = engine_input_W / 1000.0 * float(config["specific_fuel_consumption_kg_per_kWh"])
    endurance_hr = float(config["fuel_capacity_kg"]) / fuel_flow_kg_hr if fuel_flow_kg_hr > 0 else 0.0
    return {
        "shaft_power_W": power_W,
        "engine_input_W": engine_input_W,
        "fuel_flow_kg_hr": fuel_flow_kg_hr,
        "hover_endurance_hr": endurance_hr,
    }


def propulsion_summary(case: HoverCase, config: dict[str, Any]) -> dict[str, float]:
    if config["propulsion_model"] == "electric":
        return electric_summary(total_hover_power_W(case), config)
    return fossil_summary(total_hover_power_W(case), config)


def estimate_forward_flight(
    rotor: RotorSpec,
    hover_result: HoverResult,
    *,
    velocity_m_s: float,
    density_kg_m3: float,
    flat_plate_area_m2: float,
) -> ForwardFlightEstimate:
    downwash_velocity_ratio = walds_solver(
        velocity_m_s,
        hover_result.mean_induced_velocity_m_s,
        0.0,
    )
    advance_ratio = velocity_m_s / max(rotor.omega_rad_s * rotor.radius_m, 1e-9)
    induced_power_W = max(0.0, downwash_velocity_ratio * hover_result.induced_power_W)
    profile_power_W = hover_result.profile_power_W * (1.0 + 4.65 * advance_ratio**2)
    body_drag_N = density_kg_m3 * velocity_m_s**2 * flat_plate_area_m2 / 2.0
    parasite_power_W = body_drag_N * velocity_m_s
    return ForwardFlightEstimate(
        velocity_m_s=velocity_m_s,
        downwash_velocity_ratio=downwash_velocity_ratio,
        induced_power_W=induced_power_W,
        profile_power_W=profile_power_W,
        parasite_power_W=parasite_power_W,
        total_power_W=induced_power_W + profile_power_W + parasite_power_W,
        induced_drag_N=induced_power_W / max(2.0 * hover_result.mean_induced_velocity_m_s, 1e-9),
        profile_drag_N=profile_power_W / max(rotor.omega_rad_s * rotor.radius_m, 1e-9),
        body_drag_N=body_drag_N,
    )


def velocity_sweep(
    rotor: RotorSpec,
    hover_result: HoverResult,
    *,
    density_kg_m3: float,
    flat_plate_area_m2: float,
    min_velocity_m_s: float = 3.0,
    max_velocity_m_s: float = 30.0,
    points: int = 50,
) -> list[ForwardFlightEstimate]:
    return [
        estimate_forward_flight(
            rotor,
            hover_result,
            velocity_m_s=float(velocity),
            density_kg_m3=density_kg_m3,
            flat_plate_area_m2=flat_plate_area_m2,
        )
        for velocity in np.linspace(min_velocity_m_s, max_velocity_m_s, points)
    ]


def electric_range_sweep(
    estimates: list[ForwardFlightEstimate],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocities_kmh = np.array([estimate.velocity_m_s * MPS_TO_KMH for estimate in estimates])
    input_powers_W = np.array(
        [
            electric_summary(estimate.total_power_W, config)["electric_input_W"]
            for estimate in estimates
        ]
    )
    usable_energy_Wh = float(config["battery_capacity_Wh"]) * float(config["battery_usable_fraction"])
    endurance_hr = usable_energy_Wh / input_powers_W
    return velocities_kmh, endurance_hr, endurance_hr * velocities_kmh


def fossil_range_sweep(
    estimates: list[ForwardFlightEstimate],
    hover_result: HoverResult,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocities_kmh = np.array([estimate.velocity_m_s * MPS_TO_KMH for estimate in estimates])
    effective_drag_N = np.array(
        [
            estimate.induced_drag_N + estimate.profile_drag_N + estimate.body_drag_N
            for estimate in estimates
        ]
    )
    fuel_capacity_kg = float(config["fuel_capacity_kg"])
    gross_kg = float(config["gross"])
    sfc = float(config["specific_fuel_consumption_kg_per_kWh"])
    loss = float(config["transmission_loss"])
    total_power_kW = np.array([estimate.total_power_W / 1000.0 for estimate in estimates])
    fuel_flow_kg_hr = total_power_kW * sfc / (1.0 - loss)
    endurance_hr = fuel_capacity_kg / fuel_flow_kg_hr
    ld = hover_result.total_thrust_N / np.maximum(effective_drag_N, 1e-9)
    fuel_fraction_mass = max(gross_kg - fuel_capacity_kg, 1e-9)
    breguet_range_km = 366.0 / max(sfc, 1e-9) * ld * log(gross_kg / fuel_fraction_mass)
    return velocities_kmh, endurance_hr, breguet_range_km


def load_rows_for_result(result: HoverResult) -> list[dict[str, Any]]:
    return result.load_table()
