"""State and formatting helpers for the Panel dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class RotorSettings:
    airfoil: str = "naca23012"
    num_blades: int = 5
    chord: float = 0.53
    rotor_diameter: float = 21.29
    tip_speed_mach: float = 0.624
    washout: float = -8.0
    rotor_root_cutout: float = 0.02
    new_polar: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RotorSettings":
        return cls(
            airfoil=validate_airfoil_name(str(data.get("airfoil", cls.airfoil))),
            num_blades=int(data.get("num_blades", cls.num_blades)),
            chord=float(data.get("chord", cls.chord)),
            rotor_diameter=float(data.get("rotor_diam", data.get("rotor_diameter", cls.rotor_diameter))),
            tip_speed_mach=float(data.get("tip_speed_mach", cls.tip_speed_mach)),
            washout=float(data.get("washout", cls.washout)),
            rotor_root_cutout=float(
                data.get("root_cutout", data.get("rotor_root_cutout", cls.rotor_root_cutout))
            ),
            new_polar=bool(data.get("new_polar", data.get("b_new_polars", cls.new_polar))),
        )

    def to_rotor_kwargs(self) -> dict[str, Any]:
        return {
            "airfoil": validate_airfoil_name(self.airfoil),
            "num_blades": self.num_blades,
            "chord": self.chord,
            "rotor_diameter": self.rotor_diameter,
            "tip_speed_mach": self.tip_speed_mach,
            "washout": self.washout,
            "rotor_root_cutout": self.rotor_root_cutout,
            "new_polar": self.new_polar,
        }


@dataclass(frozen=True)
class FlightSettings:
    gross_weight: float = 12_500.0
    density: float = 1.225
    transmission_loss: float = 0.10
    fuel_capacity: float = 1_500.0
    specific_fuel_consumption: float = 0.30
    battery_capacity: float = 1_500.0
    velocity: float = 100.0
    flat_plate_area: float = 3.5

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FlightSettings":
        return cls(
            gross_weight=float(data.get("gross", data.get("gross_weight", cls.gross_weight))),
            density=float(data.get("density", cls.density)),
            transmission_loss=float(data.get("transmission_loss", cls.transmission_loss)),
            fuel_capacity=float(data.get("fuel_cap", data.get("fuel_capacity", cls.fuel_capacity))),
            specific_fuel_consumption=float(
                data.get("sfc", data.get("specific_fuel_consumption", cls.specific_fuel_consumption))
            ),
            battery_capacity=float(data.get("battery_cap", data.get("battery_capacity", cls.battery_capacity))),
            velocity=float(data.get("velocity", cls.velocity)),
            flat_plate_area=float(data.get("fpa", data.get("flat_plate_area", cls.flat_plate_area))),
        )

    @property
    def velocity_mps(self) -> float:
        return self.velocity / 3.6


@dataclass(frozen=True)
class DashboardSettings:
    rotor: RotorSettings = RotorSettings()
    flight: FlightSettings = FlightSettings()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DashboardSettings":
        return cls(rotor=RotorSettings.from_mapping(data), flight=FlightSettings.from_mapping(data))

    def to_legacy_mapping(self) -> dict[str, Any]:
        return {
            "airfoil": self.rotor.airfoil,
            "num_blades": self.rotor.num_blades,
            "chord": self.rotor.chord,
            "rotor_diam": self.rotor.rotor_diameter,
            "tip_speed_mach": self.rotor.tip_speed_mach,
            "washout": self.rotor.washout,
            "root_cutout": self.rotor.rotor_root_cutout,
            "new_polar": self.rotor.new_polar,
            "gross": self.flight.gross_weight,
            "density": self.flight.density,
            "transmission_loss": self.flight.transmission_loss,
            "fuel_cap": self.flight.fuel_capacity,
            "sfc": self.flight.specific_fuel_consumption,
            "battery_cap": self.flight.battery_capacity,
            "velocity": self.flight.velocity,
            "fpa": self.flight.flat_plate_area,
        }


MetricRow = tuple[str, str, float | int | str, str]


def validate_airfoil_name(airfoil: str) -> str:
    normalized = airfoil.strip().lower()
    if not normalized.startswith("naca"):
        raise ValueError("Airfoil must be a NACA profile.")

    digits = normalized[4:]
    if not 4 <= len(digits) <= 6 or not digits.isdigit():
        raise ValueError("Only 4, 5, and 6 digit NACA profiles are supported.")

    return normalized


def format_metric_value(value: float | int | str) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def format_metric_rows(rows: list[MetricRow]) -> str:
    body = []
    last_section = None
    for section, label, value, unit in rows:
        if section != last_section:
            body.append(
                f'<tr class="section-row"><th colspan="3">{escape(section)}</th></tr>'
            )
            last_section = section

        body.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f'<td class="metric-value">{escape(format_metric_value(value))}</td>'
            f"<td>{escape(unit)}</td>"
            "</tr>"
        )

    return '<table class="metric-table">' + "".join(body) + "</table>"


def collect_metric_rows(rotor: Any, settings: DashboardSettings) -> list[MetricRow]:
    rows: list[MetricRow] = [
        ("Geometry", "Tip speed", rotor.tip_speed, "m/s"),
        ("Geometry", "Rotor speed", rotor.rpm, "RPM"),
        ("Geometry", "Disk area", rotor.rotor_disk_area, "m2"),
        ("Geometry", "Solidity", rotor.solidity, ""),
        ("Inputs", "Gross weight", settings.flight.gross_weight, "kg"),
        ("Inputs", "Velocity", settings.flight.velocity, "km/hr"),
        ("Inputs", "Flat plate area", settings.flight.flat_plate_area, "m2"),
    ]

    if hasattr(rotor, "hover_thrust"):
        rows.extend(
            [
                ("Hover", "Collective theta", rotor.theta, "deg"),
                ("Hover", "Mean induced velocity", rotor.hover_induced_vel, "m/s"),
                ("Hover", "Thrust", rotor.hover_thrust / 9.81, "kg"),
                ("Hover", "Induced power", rotor.hover_power_induced * 0.00134102209, "SHP"),
                ("Hover", "Profile power", rotor.hover_power_profile * 0.00134102209, "SHP"),
                ("Hover", "Total power", rotor.hover_power_total * 0.00134102209, "SHP"),
                (
                    "Hover",
                    "Brake horsepower",
                    rotor.hover_power_total * 0.00134102209 / (1 - settings.flight.transmission_loss),
                    "BHP",
                ),
                ("Hover", "Ct", rotor.ct, ""),
                ("Hover", "Cp", rotor.cp, ""),
                ("Hover", "Figure of merit", rotor.merit, ""),
                ("Hover", "Max figure of merit", rotor.merit_max, ""),
                ("Hover", "Tip loss", rotor.tip_loss, ""),
            ]
        )

    if hasattr(rotor, "power_total"):
        rows.extend(
            [
                ("Forward", "Downwash velocity ratio", rotor.downwash_velocity_ratio, ""),
                ("Forward", "Induced power", rotor.power_induced * 0.00134102209, "SHP"),
                ("Forward", "Profile power", rotor.power_profile * 0.00134102209, "SHP"),
                ("Forward", "Parasite power", rotor.power_parasite * 0.00134102209, "SHP"),
                ("Forward", "Total power", rotor.horsepower_total, "SHP"),
                (
                    "Forward",
                    "Brake horsepower",
                    rotor.horsepower_total / (1 - settings.flight.transmission_loss),
                    "BHP",
                ),
                ("Forward", "Body drag", rotor.body_drag, "N"),
            ]
        )

    return rows


def settings_to_json(settings: DashboardSettings) -> str:
    import json

    return json.dumps(settings.to_legacy_mapping(), indent=4)
