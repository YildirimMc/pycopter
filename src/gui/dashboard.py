"""Panel browser dashboard for PyCopter."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import panel as pn

from pycopter import Rotor

from gui.state import (
    DashboardSettings,
    FlightSettings,
    RotorSettings,
    collect_metric_rows,
    format_metric_rows,
    settings_to_json,
    validate_airfoil_name,
)


ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = ROOT / "tutorials" / "presets"

PLOT_OPTIONS = [
    "Power vs velocity",
    "Range and endurance",
    "Specific range and L/D",
    "Max FMR vs Ct/sigma",
    "Downwash ratio",
    "Ground effect",
    "Electric range",
    "Cl and Cd vs alpha",
]

ENGINEERING_CSS = """
body {
    background: #f4f4f2;
}
.app-title {
    background: #30383d;
    color: #f5f5f1;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0;
    padding: 10px 14px;
}
.engineering-pane {
    border: 1px solid #9b9b94;
    background: #fbfbfa;
    padding: 10px;
}
.section-title {
    border-bottom: 1px solid #b8b8b0;
    color: #1d1f21;
    font-size: 13px;
    font-weight: 700;
    margin: 2px 0 8px;
    padding-bottom: 4px;
    text-transform: uppercase;
}
.metric-table {
    border-collapse: collapse;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    width: 100%;
}
.metric-table th,
.metric-table td {
    border: 1px solid #bdbdb6;
    padding: 4px 6px;
}
.metric-table th {
    background: #e2e2dc;
    text-align: left;
}
.metric-value {
    text-align: right;
    width: 95px;
}
.bk-input,
.bk-btn {
    border-radius: 2px !important;
}
"""

pn.extension(raw_css=[ENGINEERING_CSS], sizing_mode="stretch_width")


class PyCopterDashboard:
    def __init__(self, rotor_factory: Callable[..., Rotor] = Rotor) -> None:
        self.rotor_factory = rotor_factory
        self.rotor: Rotor | None = None
        self.settings = DashboardSettings()

        self._build_widgets()
        self.metrics = pn.pane.HTML(
            format_metric_rows([]),
            css_classes=["engineering-pane"],
            height=390,
        )
        self.console = pn.widgets.TextAreaInput(
            name="Output",
            value="",
            disabled=True,
            height=190,
            styles={"font-family": "Consolas, monospace", "font-size": "12px"},
        )
        self.plot_pane = pn.pane.Matplotlib(
            self._empty_plot(),
            tight=True,
            sizing_mode="stretch_both",
            min_height=430,
        )
        self.status = pn.pane.Markdown("`ready`", sizing_mode="stretch_width")
        self._run_case(update_plot=True)

    def view(self) -> pn.Column:
        parameter_controls = pn.Column(
            self._section("Rotor"),
            self.airfoil,
            self.new_polar,
            self.num_blades,
            self.chord,
            self.rotor_diameter,
            self.tip_speed_mach,
            self.washout,
            self.rotor_root_cutout,
            self._section("Flight"),
            self.gross_weight,
            self.density,
            self.transmission_loss,
            self.velocity,
            self.flat_plate_area,
            self._section("Energy"),
            self.fuel_capacity,
            self.sfc,
            self.battery_capacity,
            height=430,
            styles={"overflow-y": "auto"},
            sizing_mode="stretch_width",
        )

        controls = pn.Column(
            self._section("Preset"),
            self.preset_select,
            pn.Row(self.load_preset_button, self.download_config),
            self._section("Actions"),
            pn.Row(self.run_button, self.plot_button),
            self.plot_select,
            parameter_controls,
            width=330,
            height=640,
            css_classes=["engineering-pane"],
        )

        header = pn.pane.HTML('<div class="app-title">PyCopter</div>')
        workbench = pn.Row(
            controls,
            pn.Column(self.status, self.metrics, width=410),
            self.plot_pane,
            height=660,
            sizing_mode="stretch_width",
        )
        return pn.Column(header, workbench, self.console, sizing_mode="stretch_both")

    def _build_widgets(self) -> None:
        self.preset_paths = self._preset_paths()
        preset_names = [""] + sorted(self.preset_paths)
        self.preset_select = pn.widgets.Select(name="Preset", options=preset_names, value="")
        self.load_preset_button = pn.widgets.Button(name="Load", button_type="default", width=94)
        self.load_preset_button.on_click(self._load_selected_preset)
        self.download_config = pn.widgets.FileDownload(
            callback=self._download_settings,
            filename="pycopter-config.json",
            button_type="default",
            label="Save",
            width=94,
        )

        rotor = self.settings.rotor
        self.airfoil = pn.widgets.TextInput(name="Airfoil", value=rotor.airfoil)
        self.new_polar = pn.widgets.Checkbox(name="Generate new polar", value=rotor.new_polar)
        self.num_blades = pn.widgets.IntInput(name="Blades", value=rotor.num_blades, start=1, end=20)
        self.chord = pn.widgets.FloatInput(name="Chord [m]", value=rotor.chord, step=0.01)
        self.rotor_diameter = pn.widgets.FloatInput(
            name="Rotor diameter [m]", value=rotor.rotor_diameter, step=0.01
        )
        self.tip_speed_mach = pn.widgets.FloatInput(
            name="Tip speed [Mach]", value=rotor.tip_speed_mach, step=0.001
        )
        self.washout = pn.widgets.FloatInput(name="Washout [deg]", value=rotor.washout, step=0.1)
        self.rotor_root_cutout = pn.widgets.FloatInput(
            name="Root cutout ratio", value=rotor.rotor_root_cutout, step=0.001
        )

        flight = self.settings.flight
        self.gross_weight = pn.widgets.FloatInput(
            name="Gross weight [kg]", value=flight.gross_weight, step=10
        )
        self.density = pn.widgets.FloatInput(name="Density [kg/m3]", value=flight.density, step=0.001)
        self.transmission_loss = pn.widgets.FloatInput(
            name="Transmission loss", value=flight.transmission_loss, step=0.01
        )
        self.velocity = pn.widgets.FloatInput(name="Velocity [km/hr]", value=flight.velocity, step=1)
        self.flat_plate_area = pn.widgets.FloatInput(
            name="Flat plate area [m2]", value=flight.flat_plate_area, step=0.01
        )
        self.fuel_capacity = pn.widgets.FloatInput(
            name="Fuel capacity [kg]", value=flight.fuel_capacity, step=10
        )
        self.sfc = pn.widgets.FloatInput(
            name="Specific fuel consumption", value=flight.specific_fuel_consumption, step=0.01
        )
        self.battery_capacity = pn.widgets.FloatInput(
            name="Battery capacity [kWh]", value=flight.battery_capacity, step=10
        )

        self.run_button = pn.widgets.Button(name="Run case", button_type="primary", width=118)
        self.run_button.on_click(self._run_case)
        self.plot_button = pn.widgets.Button(name="Generate plot", button_type="default", width=118)
        self.plot_button.on_click(self._generate_plot)
        self.plot_select = pn.widgets.Select(
            name="Plot",
            options=PLOT_OPTIONS,
            value="Power vs velocity",
        )

    def _section(self, label: str) -> pn.pane.HTML:
        return pn.pane.HTML(f'<div class="section-title">{label}</div>')

    def _preset_paths(self) -> dict[str, Path]:
        if not PRESETS_DIR.exists():
            return {}
        return {path.stem: path for path in PRESETS_DIR.glob("*.json")}

    def _settings_from_widgets(self) -> DashboardSettings:
        rotor = RotorSettings(
            airfoil=validate_airfoil_name(self.airfoil.value),
            num_blades=int(self.num_blades.value),
            chord=float(self.chord.value),
            rotor_diameter=float(self.rotor_diameter.value),
            tip_speed_mach=float(self.tip_speed_mach.value),
            washout=float(self.washout.value),
            rotor_root_cutout=float(self.rotor_root_cutout.value),
            new_polar=bool(self.new_polar.value),
        )
        flight = FlightSettings(
            gross_weight=float(self.gross_weight.value),
            density=float(self.density.value),
            transmission_loss=float(self.transmission_loss.value),
            fuel_capacity=float(self.fuel_capacity.value),
            specific_fuel_consumption=float(self.sfc.value),
            battery_capacity=float(self.battery_capacity.value),
            velocity=float(self.velocity.value),
            flat_plate_area=float(self.flat_plate_area.value),
        )
        return DashboardSettings(rotor=rotor, flight=flight)

    def _apply_settings(self, settings: DashboardSettings) -> None:
        self.airfoil.value = settings.rotor.airfoil
        self.new_polar.value = settings.rotor.new_polar
        self.num_blades.value = settings.rotor.num_blades
        self.chord.value = settings.rotor.chord
        self.rotor_diameter.value = settings.rotor.rotor_diameter
        self.tip_speed_mach.value = settings.rotor.tip_speed_mach
        self.washout.value = settings.rotor.washout
        self.rotor_root_cutout.value = settings.rotor.rotor_root_cutout
        self.gross_weight.value = settings.flight.gross_weight
        self.density.value = settings.flight.density
        self.transmission_loss.value = settings.flight.transmission_loss
        self.fuel_capacity.value = settings.flight.fuel_capacity
        self.sfc.value = settings.flight.specific_fuel_consumption
        self.battery_capacity.value = settings.flight.battery_capacity
        self.velocity.value = settings.flight.velocity
        self.flat_plate_area.value = settings.flight.flat_plate_area

    def _load_selected_preset(self, _event: object) -> None:
        selected = self.preset_select.value
        if not selected:
            return

        try:
            with self.preset_paths[selected].open("r", encoding="utf-8") as file:
                settings = DashboardSettings.from_mapping(json.load(file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.status.object = f"`preset load failed: {exc}`"
            return

        self.settings = settings
        self._apply_settings(settings)
        self.status.object = f"`loaded {selected}`"

    def _download_settings(self) -> io.StringIO:
        return io.StringIO(settings_to_json(self._settings_from_widgets()))

    def _run_case(self, _event: object | None = None, update_plot: bool = True) -> None:
        try:
            settings = self._settings_from_widgets()
            transcript = io.StringIO()
            with contextlib.redirect_stdout(transcript):
                rotor = self.rotor_factory(**settings.rotor.to_rotor_kwargs())
                rotor.hover(settings.flight.gross_weight, settings.flight.density)
                rotor.forward_flight(
                    settings.flight.velocity_mps,
                    settings.flight.density,
                    settings.flight.flat_plate_area,
                )

            self.settings = settings
            self.rotor = rotor
            self.metrics.object = format_metric_rows(collect_metric_rows(rotor, settings))
            self.console.value = transcript.getvalue()
            self.status.object = "`case complete`"
            if update_plot:
                self._generate_plot(None, rerun=False)
        except Exception as exc:  # noqa: BLE001 - show numerical/model errors in the workbench.
            self.status.object = f"`case failed: {exc}`"
            self.console.value = traceback.format_exc()

    def _generate_plot(self, _event: object | None, rerun: bool = True) -> None:
        if rerun or self.rotor is None:
            self._run_case(update_plot=False)
            if self.rotor is None:
                return

        try:
            transcript = io.StringIO()
            with contextlib.redirect_stdout(transcript):
                figure = self._plot_selected()
            plt.close(self.plot_pane.object)
            self.plot_pane.object = figure
            if transcript.getvalue():
                self.console.value = (
                    self.console.value.rstrip()
                    + f"\n\n[plot]\nGenerated {self.plot_select.value}."
                )
            self.status.object = f"`plot: {self.plot_select.value}`"
        except Exception as exc:  # noqa: BLE001 - show plotting/model errors in the workbench.
            self.status.object = f"`plot failed: {exc}`"
            self.console.value = traceback.format_exc()

    def _plot_selected(self) -> plt.Figure:
        assert self.rotor is not None

        plotter = {
            "Power vs velocity": self._plot_powers_vs_velocity,
            "Range and endurance": self._plot_range_endurance_vs_velocity,
            "Specific range and L/D": self._plot_sr_ld_vs_velocity,
            "Max FMR vs Ct/sigma": self._plot_ctsigma_to_maxmerit,
            "Downwash ratio": self._plot_nfs_to_dvr,
            "Ground effect": self._plot_ground_effect,
            "Electric range": self._plot_range_electric,
            "Cl and Cd vs alpha": self._plot_alpha_cl_cd,
        }[self.plot_select.value]
        return plotter()

    def _velocity_sweep(self, start: float = 10.0, stop: float = 80.0, count: int = 50) -> dict[str, np.ndarray]:
        assert self.rotor is not None

        settings = self.settings
        rotor = self.rotor
        rotor.hover(settings.flight.gross_weight, settings.flight.density)
        velocities = np.linspace(start, stop, count)
        induced_powers = np.zeros(count)
        profile_powers = np.zeros(count)
        parasite_powers = np.zeros(count)
        total_powers = np.zeros(count)
        effective_drags = np.zeros(count)
        downwash_ratios = np.zeros(count)

        for index, velocity in enumerate(velocities):
            rotor.forward_flight(
                velocity,
                settings.flight.density,
                settings.flight.flat_plate_area,
            )
            induced_powers[index] = rotor.power_induced
            profile_powers[index] = rotor.power_profile
            parasite_powers[index] = rotor.power_parasite
            total_powers[index] = rotor.power_total
            effective_drags[index] = rotor.drag_induced + rotor.drag_profile + rotor.body_drag
            downwash_ratios[index] = rotor.downwash_velocity_ratio

        return {
            "velocities_mps": velocities,
            "velocities_kmh": velocities * 3.6,
            "induced_powers": induced_powers,
            "profile_powers": profile_powers,
            "parasite_powers": parasite_powers,
            "total_powers": total_powers,
            "effective_drags": effective_drags,
            "downwash_ratios": downwash_ratios,
            "lift": np.full(count, rotor.hover_thrust),
        }

    def _plot_powers_vs_velocity(self) -> plt.Figure:
        data = self._velocity_sweep()
        fig, ax = plt.subplots()
        hp = 0.00134102209
        ax.plot(data["velocities_kmh"], data["induced_powers"] * hp, label="Induced")
        ax.plot(data["velocities_kmh"], data["profile_powers"] * hp, label="Profile")
        ax.plot(data["velocities_kmh"], data["parasite_powers"] * hp, label="Parasite")
        ax.plot(data["velocities_kmh"], data["total_powers"] * hp, label="Total", color="black")
        ax.set_title("Forward flight powers")
        ax.set_xlabel("Free-stream velocity [km/hr]")
        ax.set_ylabel("Shaft horsepower [SHP]")
        ax.grid(True)
        ax.legend()
        return fig

    def _plot_range_endurance_vs_velocity(self) -> plt.Figure:
        data = self._velocity_sweep()
        settings = self.settings
        total_powers_kw = data["total_powers"] / 1000
        engine_powers = 1.13 * total_powers_kw / (1 - settings.flight.transmission_loss)
        sfc = 1.13 * settings.flight.specific_fuel_consumption / (1 - settings.flight.transmission_loss)
        lift_drag = data["lift"] / data["effective_drags"]
        fuel_consumptions = engine_powers * sfc
        endurance = settings.flight.fuel_capacity / fuel_consumptions
        ranges = 366 / sfc * lift_drag * np.log(
            settings.flight.gross_weight / (settings.flight.gross_weight - settings.flight.fuel_capacity)
        )

        fig, ax1 = plt.subplots()
        ax1.plot(data["velocities_kmh"], endurance, color="tab:red", label="Endurance")
        ax1.set_title("Range and endurance")
        ax1.set_xlabel("Free-stream velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")
        ax2 = ax1.twinx()
        ax2.plot(data["velocities_kmh"], ranges, label="Range")
        ax2.set_ylabel("Range [km]")
        ax1.grid(True)
        fig.legend(loc="upper right")
        return fig

    def _plot_sr_ld_vs_velocity(self) -> plt.Figure:
        data = self._velocity_sweep()
        settings = self.settings
        total_powers_kw = data["total_powers"] / 1000
        engine_powers = 1.13 * total_powers_kw / (1 - settings.flight.transmission_loss)
        lift_drag = data["lift"] / data["effective_drags"]
        fuel_consumptions = engine_powers * settings.flight.specific_fuel_consumption
        specific_range = data["velocities_kmh"] / fuel_consumptions

        fig, ax1 = plt.subplots()
        ax1.plot(data["velocities_kmh"], specific_range, color="tab:orange", label="Specific range")
        ax1.set_title("Specific range and L/D")
        ax1.set_xlabel("Free-stream velocity [km/hr]")
        ax1.set_ylabel("Specific range [km/kg]")
        ax2 = ax1.twinx()
        ax2.plot(data["velocities_kmh"], lift_drag, label="L/D")
        ax2.set_ylabel("Lift / drag")
        ax1.grid(True)
        fig.legend(loc="upper right")
        return fig

    def _plot_ctsigma_to_maxmerit(self) -> plt.Figure:
        assert self.rotor is not None
        self.rotor.hover(self.settings.flight.gross_weight, self.settings.flight.density)
        ct_range = np.linspace(self.rotor.ct - self.rotor.ct / 2, self.rotor.ct + self.rotor.ct / 2, 50)
        merit_max = 0.707 * ct_range**1.5 / (
            ct_range**1.5 / (np.sqrt(2) * self.rotor.tip_loss) + self.rotor.solidity * self.rotor.cd_mean / 8
        )
        fig, ax = plt.subplots()
        ax.plot(ct_range / self.rotor.solidity, merit_max)
        ax.set_title("Max figure of merit")
        ax.set_xlabel("Ct / sigma")
        ax.set_ylabel("Maximum figure of merit")
        ax.grid(True)
        return fig

    def _plot_nfs_to_dvr(self) -> plt.Figure:
        data = self._velocity_sweep(start=0.0, stop=50.0)
        normalized = data["velocities_mps"] / self.rotor.hover_induced_vel
        fig, ax = plt.subplots()
        ax.plot(normalized, data["downwash_ratios"])
        ax.set_title("Wald equation")
        ax.set_xlabel("Normalized flight speed [V/v0]")
        ax.set_ylabel("Downwash velocity ratio [v/v0]")
        ax.grid(True)
        return fig

    def _plot_ground_effect(self) -> plt.Figure:
        assert self.rotor is not None
        self.rotor.hover(self.settings.flight.gross_weight, self.settings.flight.density)
        heights = np.linspace(max(self.rotor.r * 0.55, 0.1), self.rotor.r * 5, 50)
        thrusts = np.array([self.rotor.ige(self.rotor.hover_thrust, height) for height in heights])
        fig, ax = plt.subplots()
        ax.plot(thrusts / 9.81, heights, label="In ground effect")
        ax.axvline(self.rotor.hover_thrust / 9.81, color="black", label="Out of ground effect")
        ax.set_title(f"Ground effect at {self.rotor.theta} deg collective")
        ax.set_xlabel("Thrust [kg]")
        ax.set_ylabel("Rotor height [m]")
        ax.grid(True)
        ax.legend()
        return fig

    def _plot_range_electric(self) -> plt.Figure:
        data = self._velocity_sweep()
        settings = self.settings
        motor_powers = 1.13 * data["total_powers"] / 1000 / (1 - settings.flight.transmission_loss)
        endurance = settings.flight.battery_capacity / motor_powers
        ranges = endurance * data["velocities_kmh"]
        fig, ax1 = plt.subplots()
        ax1.plot(data["velocities_kmh"], endurance, label="Endurance")
        ax1.set_title("Electric range and endurance")
        ax1.set_xlabel("Velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")
        ax2 = ax1.twinx()
        ax2.plot(data["velocities_kmh"], ranges, color="tab:orange", label="Range")
        ax2.set_ylabel("Range [km]")
        ax1.grid(True)
        fig.legend(loc="upper right")
        return fig

    def _plot_alpha_cl_cd(self) -> plt.Figure:
        assert self.rotor is not None
        alpha = np.arange(-5, 20)
        cl_cd = np.empty((len(alpha), 2))
        for index, alpha_i in enumerate(alpha):
            cl_cd[index] = self.rotor.polar.get_polar(alpha_i)

        fig, ax1 = plt.subplots()
        ax1.plot(alpha, cl_cd[:, 0], label="Cl")
        ax1.set_title(f"{self.settings.rotor.airfoil}, Re={self.rotor.polar.reynolds:.0f}")
        ax1.set_xlabel("Angle of attack [deg]")
        ax1.set_ylabel("Cl")
        ax2 = ax1.twinx()
        ax2.plot(alpha, cl_cd[:, 1], color="tab:orange", label="Cd")
        ax2.set_ylabel("Cd")
        ax1.grid(True)
        fig.legend(loc="upper left")
        return fig

    def _empty_plot(self) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.set_title("No plot generated")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        return fig


def create_dashboard() -> pn.Column:
    return PyCopterDashboard().view()


def main() -> None:
    pn.serve(create_dashboard(), title="PyCopter", show=True, port=5006)


if pn.state.served:
    create_dashboard().servable()
