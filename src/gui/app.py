"""Panel Web UI for PyCopter."""

from __future__ import annotations

import io
import json
from html import escape
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn

from pycopter import CoaxialHoverResult, HoverResult

from .calculations import (
    DEFAULT_CONFIG,
    DEFAULT_STATION_ROWS,
    CONFIG_VERSION,
    HoverCase,
    build_rotor_spec,
    electric_summary,
    electric_range_sweep,
    estimate_forward_flight,
    fossil_range_sweep,
    load_rows_for_result,
    normalize_config,
    primary_hover_result,
    propulsion_summary,
    run_hover_case,
    station_rows_from_uniform,
    total_hover_power_W,
    total_hover_thrust_N,
    velocity_sweep,
)


PLOT_FIGSIZE = (7.0, 8.0)
RESULT_PANE_HEIGHT = 780
RESULT_TABS_HEIGHT = 820
THEME = {
    "page": "#111315",
    "panel": "#1b1d20",
    "panel_alt": "#23262b",
    "field": "#151719",
    "field_hover": "#202329",
    "border": "#3a4048",
    "border_strong": "#59616b",
    "text": "#e7ecf2",
    "muted": "#aeb7c2",
    "accent": "#4db6ac",
    "accent_hover": "#63c8bf",
    "warning": "#d6a858",
    "plot": "#141619",
    "plot_grid": "#353a42",
}


RAW_CSS = """
html,
body {
    background: #111315;
}
.pycopter-shell {
    font-family: Arial, Helvetica, sans-serif;
    color: #e7ecf2;
}
.pycopter-panel {
    border: 1px solid #3a4048;
    background: #1b1d20;
    padding: 8px;
}
.pycopter-panel-title {
    border: 1px solid #3a4048;
    background: #23262b;
    color: #e7ecf2;
    padding: 7px 8px;
    text-align: center;
    font-size: 12px;
    font-weight: 600;
}
.pycopter-log {
    height: 150px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    color: #d9efe9;
    background: #121416;
    border: 1px solid #3a4048;
    padding: 8px;
    user-select: text;
}
.pycopter-log-text {
    margin: 0;
    white-space: pre-wrap;
}
button.bk-btn,
.bk-btn {
    border: 1px solid #59616b !important;
    border-radius: 2px !important;
    background: #23262b !important;
    color: #e7ecf2 !important;
    min-height: 26px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 1px 0 rgba(0, 0, 0, 0.35);
}
button.bk-btn:hover,
.bk-btn:hover {
    background: #2c3036 !important;
    border-color: #4db6ac !important;
}
button.bk-btn:disabled,
.bk-btn:disabled,
.bk-btn.bk-disabled {
    color: #737d88 !important;
    background: #1b1d20 !important;
    border-color: #333840 !important;
    box-shadow: none;
}
.pycopter-plot-frame {
    background: #141619;
    border: 1px solid #3a4048;
}
.bk-root,
.bk {
    color: #e7ecf2;
}
.bk-input-group label,
.bk-input-group .bk-input-group-label,
.bk-checkbox-label,
.bk-radio-group label {
    color: #d7dde5 !important;
}
.bk-input,
input.bk-input,
select.bk-input,
textarea.bk-input {
    background: #151719 !important;
    border-color: #3a4048 !important;
    color: #e7ecf2 !important;
}
.bk-input:hover,
input.bk-input:hover,
select.bk-input:hover,
textarea.bk-input:hover {
    background: #202329 !important;
    border-color: #59616b !important;
}
.bk-input:focus,
input.bk-input:focus,
select.bk-input:focus,
textarea.bk-input:focus {
    border-color: #4db6ac !important;
    box-shadow: 0 0 0 1px #4db6ac !important;
}
.bk-divider {
    border-color: #3a4048 !important;
}
.bk-tab {
    background: #1b1d20 !important;
    border-color: #3a4048 !important;
    color: #aeb7c2 !important;
}
.bk-tab.bk-active {
    background: #23262b !important;
    color: #e7ecf2 !important;
    border-top-color: #4db6ac !important;
}
.bk-accordion,
.bk-accordion > .bk-card {
    background: #1b1d20 !important;
    border-color: #3a4048 !important;
    color: #e7ecf2 !important;
}
.bk-card-header {
    background: #23262b !important;
    border-color: #3a4048 !important;
    color: #e7ecf2 !important;
}
.bk-card-header *,
.bk-accordion .bk-header,
.bk-accordion .bk-header *,
.bk-accordion button,
.bk-accordion button * {
    color: #e7ecf2 !important;
    opacity: 1 !important;
}
.pycopter-shell h3,
h3 {
    color: #e7ecf2 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
.pycopter-shell h3::before,
h3::before {
    color: #aeb7c2 !important;
}
input[type="file"]::file-selector-button {
    background: #23262b !important;
    border: 1px solid #59616b !important;
    color: #e7ecf2 !important;
    border-radius: 2px !important;
    padding: 4px 10px !important;
}
input[type="file"]::file-selector-button:hover {
    background: #2c3036 !important;
    border-color: #4db6ac !important;
}
.tabulator {
    background: #151719 !important;
    border-color: #3a4048 !important;
    color: #e7ecf2 !important;
}
.tabulator .tabulator-header {
    background: #23262b !important;
    border-color: #3a4048 !important;
    color: #e7ecf2 !important;
}
.tabulator .tabulator-header .tabulator-col {
    background: #23262b !important;
    border-color: #3a4048 !important;
    color: #e7ecf2 !important;
}
.tabulator .tabulator-row {
    background: #151719 !important;
    border-color: #303640 !important;
    color: #e7ecf2 !important;
}
.tabulator .tabulator-row.tabulator-row-even {
    background: #1a1d21 !important;
}
.tabulator .tabulator-row:hover {
    background: #253037 !important;
}
.tabulator .tabulator-cell {
    border-color: #303640 !important;
}
.tabulator .tabulator-footer {
    background: #23262b !important;
    border-color: #3a4048 !important;
    color: #aeb7c2 !important;
}
"""


class PycopterWebApp:
    """Stateful Panel application."""

    def __init__(self) -> None:
        pn.extension("tabulator", raw_css=[RAW_CSS])
        self.current_case: HoverCase | None = None
        self.initialized_rotor = None
        self.output_lines: list[str] = []
        self._log_render_count = 0
        self.plot_fig = self._blank_figure("Initialize rotor, then calculate hover.")

        self._build_widgets()
        self._wire_events()
        self._sync_enabled_state()
        self._update_plot_options()

    def view(self):
        toolbar = pn.Row(
            self.new_btn,
            self.save_download,
            self.load_file,
            self.load_btn,
            self.clear_output_btn,
            sizing_mode="stretch_width",
        )

        left = pn.Column(
            self._panel("Rotor Parameters", self._rotor_controls()),
            width=305,
        )
        middle = pn.Column(
            self._panel("Calculation Parameters", self._calculation_controls()),
            self._panel("Propulsion Estimation", self._propulsion_controls()),
            self._panel("Plots", self._plot_controls()),
            width=430,
        )
        right = pn.Column(
            self.result_tabs,
            width=700,
        )
        bottom = pn.Column(
            self.output_log,
            sizing_mode="stretch_width",
            css_classes=["pycopter-panel"],
        )

        return pn.Column(
            toolbar,
            pn.Row(left, middle, right),
            bottom,
            css_classes=["pycopter-shell"],
            width=1440,
        )

    def _build_widgets(self) -> None:
        cfg = DEFAULT_CONFIG
        self.new_btn = pn.widgets.Button(label="New", width=70)
        self.save_download = pn.widgets.FileDownload(
            callback=self._save_config_callback,
            filename="pycopter_config.json",
            label="Save Config",
            width=120,
        )
        self.load_file = pn.widgets.FileInput(accept=".json", width=230)
        self.load_btn = pn.widgets.Button(label="Load Config", width=120)
        self.clear_output_btn = pn.widgets.Button(label="Clear Outputs", width=120)

        self.airfoil = pn.widgets.TextInput(label="Airfoil", value=cfg["airfoil"], width=270)
        self.new_polars = pn.widgets.Checkbox(label="Generate New XFOIL Polars", value=cfg["new_polars"])
        self.rotor_system_type = pn.widgets.Select(
            label="Rotor System",
            options={"Single rotor": "single", "Coaxial": "coaxial"},
            value=cfg["rotor_system_type"],
            width=135,
        )
        self.num_blades = pn.widgets.IntInput(label="Blades per Rotor", value=cfg["num_blades"], start=1, end=12, step=1, width=135)
        self.chord = pn.widgets.FloatInput(label="Blade Chord [m]", value=cfg["chord"], start=0.001, end=2.5, step=0.001, width=135)
        self.rotor_diam = pn.widgets.FloatInput(label="Rotor Diameter [m]", value=cfg["rotor_diam"], start=0.05, end=30.0, step=0.01, width=135)
        self.root_twist = pn.widgets.FloatInput(label="Root Twist [deg]", value=cfg["root_twist_deg"], start=-30.0, end=30.0, step=0.25, width=135)
        self.tip_twist = pn.widgets.FloatInput(label="Tip Twist [deg]", value=cfg["tip_twist_deg"], start=-30.0, end=30.0, step=0.25, width=135)
        self.root_cutout = pn.widgets.FloatInput(label="Root Cutout r/R", value=cfg["root_cutout"], start=0.0, end=0.95, step=0.01, width=135)
        self.headspeed_input_mode = pn.widgets.Select(
            label="Speed Input",
            options={"RPM": "rpm", "Tip Mach": "tip_mach"},
            value=cfg["headspeed_input_mode"],
            width=115,
        )
        self.headspeed_rpm = pn.widgets.FloatInput(label="Headspeed [rpm]", value=cfg["headspeed_rpm"], start=100.0, end=100000.0, step=50.0, width=135)
        self.tip_speed_mach = pn.widgets.FloatInput(label="Tip Speed Mach", value=cfg["tip_speed_mach"], start=0.02, end=0.9, step=0.01, width=135)
        self.init_rotor_btn = pn.widgets.Button(label="Initialize Rotor", width=270)

        self.geometry_mode = pn.widgets.Select(
            label="Geometry Input",
            options={"Uniform blade": "uniform", "Station table": "station_table"},
            value=cfg["geometry_mode"],
            width=160,
        )
        self.station_table = pn.widgets.Tabulator(
            pd.DataFrame(DEFAULT_STATION_ROWS),
            show_index=False,
            height=190,
            width=400,
            layout="fit_columns",
            editors={
                "r_over_R": {"type": "number", "min": 0.02, "max": 1.0, "step": 0.01},
                "chord_m": {"type": "number", "min": 0.001, "max": 2.5, "step": 0.001},
                "twist_deg": {"type": "number", "min": -30.0, "max": 30.0, "step": 0.25},
                "airfoil": "input",
                "pitch_axis_frac": {"type": "number", "min": 0.0, "max": 1.0, "step": 0.01},
            },
        )
        self.reset_stations_btn = pn.widgets.Button(label="Reset Geometry Points From Root/Tip Twist", width=300)

        self.gross = pn.widgets.FloatInput(label="Gross Mass [kg]", value=cfg["gross"], start=0.01, end=100000.0, step=0.1, width=185)
        self.density = pn.widgets.FloatInput(label="Density [kg/m3]", value=cfg["density"], start=0.01, end=2.0, step=0.001, width=185)
        self.kinematic_viscosity = pn.widgets.FloatInput(label="Kinematic Viscosity [m2/s]", value=cfg["kinematic_viscosity_m2_s"], start=1.0e-5, end=2.5e-5, step=1.0e-7, width=185)
        self.trim_mode = pn.widgets.Select(
            label="Trim Mode",
            options={"Target thrust": "target_thrust", "Fixed collective": "fixed_collective"},
            value=cfg["trim_mode"],
            width=140,
        )
        self.collective_pitch = pn.widgets.FloatInput(label="Collective [deg]", value=cfg["collective_pitch_deg"], start=-5.0, end=25.0, step=0.25, width=185)
        self.min_collective = pn.widgets.FloatInput(label="Search From [deg]", value=cfg["min_collective_deg"], start=-10.0, end=10.0, step=0.25, width=185)
        self.max_collective = pn.widgets.FloatInput(label="Search To [deg]", value=cfg["max_collective_deg"], start=0.0, end=35.0, step=0.25, width=185)
        self.blade_element_count = pn.widgets.IntInput(label="Blade Elements", value=cfg["blade_element_count"], start=20, end=200, step=1, width=185)
        self.tip_loss_model = pn.widgets.Select(label="Tip Loss", options={"Prandtl": "prandtl", "None": "none"}, value=cfg["tip_loss_model"], width=185)
        self.root_loss_model = pn.widgets.Select(label="Root Loss", options={"Prandtl": "prandtl", "None": "none"}, value=cfg["root_loss_model"], width=185)
        self.induced_power_factor = pn.widgets.FloatInput(label="Induced Factor", value=cfg["induced_power_factor"], start=1.0, end=1.3, step=0.01, width=185)
        self.polar_alpha_min = pn.widgets.FloatInput(label="XFOIL Alpha Min [deg]", value=cfg["polar_alpha_min_deg"], start=-20.0, end=5.0, step=1.0, width=185)
        self.polar_alpha_max = pn.widgets.FloatInput(label="XFOIL Alpha Max [deg]", value=cfg["polar_alpha_max_deg"], start=10.0, end=18.0, step=1.0, width=185)
        self.xfoil_workers = pn.widgets.IntInput(label="XFOIL Workers", value=cfg["xfoil_parallel_workers"], start=2, end=16, step=1, width=185)
        self.xfoil_backend = pn.widgets.Select(label="XFOIL Backend", options={"MPI": "mpi", "Serial debug": "serial"}, value=cfg["xfoil_parallel_backend"], width=185)
        self.xfoil_cache_dir = pn.widgets.TextInput(label="XFOIL Cache Dir", value=cfg["xfoil_cache_directory"], placeholder="optional ignored path", width=390)
        self.calc_hover_btn = pn.widgets.Button(label="Calculate Hover", width=390, disabled=True)

        self.coaxial_spacing = pn.widgets.FloatInput(label="Spacing z/R", value=cfg["coaxial_spacing_ratio"], start=0.05, end=1.5, step=0.05, width=185)
        self.coaxial_trim_mode = pn.widgets.Select(
            label="Coaxial Trim",
            options={"Equal thrust": "equal_thrust", "Equal collective": "equal_collective"},
            value=cfg["coaxial_trim_mode"],
            width=185,
        )
        self.lower_collective_offset = pn.widgets.FloatInput(label="Lower Offset [deg]", value=cfg["lower_collective_offset_deg"], start=-10.0, end=10.0, step=0.25, width=185)
        self.lower_rotor_scale = pn.widgets.FloatInput(label="Lower Scale", value=cfg["lower_rotor_scale"], start=0.5, end=1.5, step=0.01, width=185)

        self.battery_capacity = pn.widgets.FloatInput(label="Battery [Wh]", value=cfg["battery_capacity_Wh"], start=0.1, end=100000.0, step=10.0, width=185)
        self.battery_usable = pn.widgets.FloatInput(label="Usable Fraction", value=cfg["battery_usable_fraction"], start=0.05, end=1.0, step=0.01, width=185)
        self.motor_efficiency = pn.widgets.FloatInput(label="Motor Efficiency", value=cfg["motor_efficiency"], start=0.01, end=1.0, step=0.01, width=185)
        self.esc_efficiency = pn.widgets.FloatInput(label="ESC Efficiency", value=cfg["esc_efficiency"], start=0.01, end=1.0, step=0.01, width=185)

        self.fuel_capacity = pn.widgets.FloatInput(label="Fuel Capacity [kg]", value=cfg["fuel_capacity_kg"], start=0.0, end=999999.0, step=0.1, width=185)
        self.sfc = pn.widgets.FloatInput(label="SFC [kg/kWh]", value=cfg["specific_fuel_consumption_kg_per_kWh"], start=0.001, end=10.0, step=0.01, width=185)
        self.transmission_loss = pn.widgets.FloatInput(label="Transmission Loss [-]", value=cfg["transmission_loss"], start=0.0, end=0.5, step=0.01, width=185)
        self.propulsion_tabs = pn.Tabs(
            ("Electric", pn.Column(self._two_col(self.battery_capacity, self.battery_usable), self._two_col(self.motor_efficiency, self.esc_efficiency))),
            ("Fossil fuel", pn.Column(self._two_col(self.fuel_capacity, self.sfc))),
            dynamic=False,
            width=405,
            height=145,
        )

        self.velocity = pn.widgets.FloatInput(label="Velocity [km/hr]", value=cfg["velocity_kmh"], start=0.0, end=9999.0, step=1.0, width=185)
        self.fpa = pn.widgets.FloatInput(label="Flat Plate Area [m2]", value=cfg["fpa"], start=0.0, end=9999.0, step=0.01, width=185)
        self.calc_forward_btn = pn.widgets.Button(label="Calculate Forward Flight", width=390, disabled=True)

        self.plot_select = pn.widgets.Select(label="Selected Plot", options=[], width=400)
        self.generate_plot_btn = pn.widgets.Button(label="Generate Plot", width=400, disabled=True)

        self.output_log = pn.pane.HTML("", height=150, sizing_mode="stretch_width", css_classes=["pycopter-log"])
        self.summary_table = pn.widgets.Tabulator(pd.DataFrame(columns=["Metric", "Value"]), show_index=False, height=RESULT_PANE_HEIGHT, layout="fit_data_stretch")
        self.load_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=RESULT_PANE_HEIGHT, layout="fit_data_stretch")
        self.plot_pane = pn.pane.Matplotlib(
            self.plot_fig,
            height=RESULT_PANE_HEIGHT,
            width=690,
            align="start",
            css_classes=["pycopter-plot-frame"],
        )
        self.result_tabs = pn.Tabs(
            ("Plot", self.plot_pane),
            ("Summary", self.summary_table),
            ("Blade Element Loads", self.load_table),
            dynamic=False,
            height=RESULT_TABS_HEIGHT,
            width=700,
        )

    def _wire_events(self) -> None:
        self.new_btn.on_click(lambda _: self._new_config())
        self.load_btn.on_click(lambda _: self._load_config())
        self.clear_output_btn.on_click(lambda _: self._clear_output())
        self.init_rotor_btn.on_click(lambda _: self._initialize_rotor())
        self.calc_hover_btn.on_click(lambda _: self._calculate_hover())
        self.calc_forward_btn.on_click(lambda _: self._calculate_forward_flight())
        self.generate_plot_btn.on_click(lambda _: self._generate_plot())
        self.reset_stations_btn.on_click(lambda _: self._reset_station_rows_from_uniform())
        self.propulsion_tabs.param.watch(lambda _: self._on_propulsion_changed(), "active")
        self.rotor_system_type.param.watch(lambda _: self._sync_enabled_state(), "value")
        self.headspeed_input_mode.param.watch(lambda _: self._sync_enabled_state(), "value")
        self.geometry_mode.param.watch(lambda _: self._sync_enabled_state(), "value")

    def _panel(self, title: str, content) -> pn.Column:
        return pn.Column(
            pn.pane.HTML(f'<div class="pycopter-panel-title">{title}</div>'),
            content,
            css_classes=["pycopter-panel"],
            sizing_mode="stretch_width",
        )

    def _two_col(self, *widgets) -> pn.Row:
        return pn.Row(*widgets, width=400)

    def _rotor_controls(self) -> pn.Column:
        return pn.Column(
            self.airfoil,
            self._two_col(self.rotor_system_type, self.num_blades),
            self._two_col(self.chord, self.rotor_diam),
            self._two_col(self.root_twist, self.tip_twist),
            self._two_col(self.root_cutout, self.headspeed_rpm),
            pn.layout.Divider(),
            self.init_rotor_btn,
            width=285,
        )

    def _coaxial_controls(self) -> pn.Column:
        return pn.Column(
            self._two_col(self.coaxial_spacing, self.coaxial_trim_mode),
            self._two_col(self.lower_collective_offset, self.lower_rotor_scale),
            width=300,
        )

    def _hover_controls(self) -> pn.Column:
        return pn.Column(
            self._two_col(self.gross, self.density),
            self._two_col(self.kinematic_viscosity, self.blade_element_count),
            self._two_col(self.min_collective, self.max_collective),
            pn.layout.Divider(),
            self.calc_hover_btn,
            width=300,
        )

    def _solver_controls(self) -> pn.Column:
        return pn.Column(
            self._two_col(self.tip_loss_model, self.root_loss_model),
            self._two_col(self.induced_power_factor),
            width=300,
        )

    def _xfoil_controls(self) -> pn.Column:
        return pn.Column(
            self.new_polars,
            self._two_col(self.polar_alpha_min, self.polar_alpha_max),
            self._two_col(self.xfoil_workers, self.xfoil_backend),
            self.xfoil_cache_dir,
            width=300,
        )

    def _calculation_controls(self) -> pn.Column:
        return pn.Column(
            self._hover_controls(),
            pn.Accordion(
                ("Background Solver Settings", self._solver_controls()),
                ("XFOIL Polar Generation", self._xfoil_controls()),
                ("Blade Geometry Control Points", self._geometry_controls()),
                ("Coaxial Settings", self._coaxial_controls()),
                active=[],
                width=410,
            ),
            width=410,
        )

    def _geometry_controls(self) -> pn.Column:
        return pn.Column(
            self.geometry_mode,
            self.station_table,
            self.reset_stations_btn,
            width=405,
        )

    def _propulsion_controls(self) -> pn.Column:
        return pn.Column(
            self._two_col(self.transmission_loss),
            self.propulsion_tabs,
            self._two_col(self.velocity, self.fpa),
            self.calc_forward_btn,
            width=405,
        )

    def _plot_controls(self) -> pn.Column:
        return pn.Column(
            self.plot_select,
            self.generate_plot_btn,
            width=405,
        )

    def _current_config(self) -> dict[str, Any]:
        return normalize_config(
            {
                "version": CONFIG_VERSION,
                "airfoil": self.airfoil.value,
                "new_polars": self.new_polars.value,
                "rotor_system_type": self.rotor_system_type.value,
                "geometry_mode": self.geometry_mode.value,
                "num_blades": self.num_blades.value,
                "chord": self.chord.value,
                "rotor_diam": self.rotor_diam.value,
                "root_twist_deg": self.root_twist.value,
                "tip_twist_deg": self.tip_twist.value,
                "root_cutout": self.root_cutout.value,
                "headspeed_input_mode": "rpm",
                "headspeed_rpm": self.headspeed_rpm.value,
                "tip_speed_mach": 0.0,
                "gross": self.gross.value,
                "density": self.density.value,
                "kinematic_viscosity_m2_s": self.kinematic_viscosity.value,
                "trim_mode": "target_thrust",
                "collective_pitch_deg": 0.0,
                "min_collective_deg": self.min_collective.value,
                "max_collective_deg": self.max_collective.value,
                "blade_element_count": self.blade_element_count.value,
                "tip_loss_model": self.tip_loss_model.value,
                "root_loss_model": self.root_loss_model.value,
                "induced_power_factor": self.induced_power_factor.value,
                "polar_alpha_min_deg": self.polar_alpha_min.value,
                "polar_alpha_max_deg": self.polar_alpha_max.value,
                "xfoil_parallel_workers": self.xfoil_workers.value,
                "xfoil_parallel_backend": self.xfoil_backend.value,
                "xfoil_cache_directory": self.xfoil_cache_dir.value,
                "propulsion_model": self._propulsion_model(),
                "battery_capacity_Wh": self.battery_capacity.value,
                "battery_usable_fraction": self.battery_usable.value,
                "motor_efficiency": self.motor_efficiency.value,
                "esc_efficiency": self.esc_efficiency.value,
                "fuel_capacity_kg": self.fuel_capacity.value,
                "specific_fuel_consumption_kg_per_kWh": self.sfc.value,
                "transmission_loss": self.transmission_loss.value,
                "velocity_kmh": self.velocity.value,
                "fpa": self.fpa.value,
                "coaxial_spacing_ratio": self.coaxial_spacing.value,
                "coaxial_trim_mode": self.coaxial_trim_mode.value,
                "lower_collective_offset_deg": self.lower_collective_offset.value,
                "lower_rotor_scale": self.lower_rotor_scale.value,
            }
        )

    def _station_rows(self) -> list[dict[str, Any]]:
        frame = self.station_table.value.copy()
        return frame.to_dict(orient="records")

    def _apply_config(self, config: dict[str, Any], station_rows: list[dict[str, Any]] | None = None) -> None:
        cfg = normalize_config(config)
        self.airfoil.value = str(cfg["airfoil"])
        self.new_polars.value = bool(cfg["new_polars"])
        self.rotor_system_type.value = cfg["rotor_system_type"]
        self.geometry_mode.value = cfg["geometry_mode"]
        self.num_blades.value = int(cfg["num_blades"])
        self.chord.value = float(cfg["chord"])
        self.rotor_diam.value = float(cfg["rotor_diam"])
        self.root_twist.value = float(cfg["root_twist_deg"])
        self.tip_twist.value = float(cfg["tip_twist_deg"])
        self.root_cutout.value = float(cfg["root_cutout"])
        self.headspeed_input_mode.value = cfg["headspeed_input_mode"]
        self.headspeed_rpm.value = float(cfg["headspeed_rpm"])
        self.tip_speed_mach.value = float(cfg["tip_speed_mach"])
        self.gross.value = float(cfg["gross"])
        self.density.value = float(cfg["density"])
        self.kinematic_viscosity.value = float(cfg["kinematic_viscosity_m2_s"])
        self.trim_mode.value = cfg["trim_mode"]
        self.collective_pitch.value = float(cfg["collective_pitch_deg"])
        self.min_collective.value = float(cfg["min_collective_deg"])
        self.max_collective.value = float(cfg["max_collective_deg"])
        self.blade_element_count.value = int(cfg["blade_element_count"])
        self.tip_loss_model.value = cfg["tip_loss_model"]
        self.root_loss_model.value = cfg["root_loss_model"]
        self.induced_power_factor.value = float(cfg["induced_power_factor"])
        self.polar_alpha_min.value = float(cfg["polar_alpha_min_deg"])
        self.polar_alpha_max.value = float(cfg["polar_alpha_max_deg"])
        self.xfoil_workers.value = int(cfg["xfoil_parallel_workers"])
        self.xfoil_backend.value = cfg["xfoil_parallel_backend"]
        self.xfoil_cache_dir.value = str(cfg["xfoil_cache_directory"])
        self.propulsion_tabs.active = 0 if cfg["propulsion_model"] == "electric" else 1
        self.battery_capacity.value = float(cfg["battery_capacity_Wh"])
        self.battery_usable.value = float(cfg["battery_usable_fraction"])
        self.motor_efficiency.value = float(cfg["motor_efficiency"])
        self.esc_efficiency.value = float(cfg["esc_efficiency"])
        self.fuel_capacity.value = float(cfg["fuel_capacity_kg"])
        self.sfc.value = float(cfg["specific_fuel_consumption_kg_per_kWh"])
        self.transmission_loss.value = float(cfg["transmission_loss"])
        self.velocity.value = float(cfg["velocity_kmh"])
        self.fpa.value = float(cfg["fpa"])
        self.coaxial_spacing.value = float(cfg["coaxial_spacing_ratio"])
        self.coaxial_trim_mode.value = cfg["coaxial_trim_mode"]
        self.lower_collective_offset.value = float(cfg["lower_collective_offset_deg"])
        self.lower_rotor_scale.value = float(cfg["lower_rotor_scale"])
        self.station_table.value = pd.DataFrame(station_rows or station_rows_from_uniform(cfg))
        self.current_case = None
        self.initialized_rotor = None
        self._sync_enabled_state()
        self._update_plot_options()

    def _propulsion_model(self) -> str:
        return "electric" if self.propulsion_tabs.active == 0 else "fossil"

    def _save_config_callback(self) -> io.BytesIO:
        payload = {
            "version": CONFIG_VERSION,
            "config": self._current_config(),
            "station_rows": self._station_rows(),
        }
        data = io.BytesIO(json.dumps(payload, indent=4).encode("utf-8"))
        data.seek(0)
        return data

    def _load_config(self) -> None:
        if not self.load_file.value:
            self._log("No configuration file selected.")
            return
        try:
            payload = json.loads(self.load_file.value.decode("utf-8"))
            if "config" in payload:
                self._apply_config(payload["config"], payload.get("station_rows"))
            else:
                self._apply_config(payload)
            self._log("Configuration loaded. Re-initialize rotor before calculating.")
        except Exception as err:
            self._log(f"ERROR - Could not load configuration: {err}")

    def _new_config(self) -> None:
        self._apply_config(DEFAULT_CONFIG, DEFAULT_STATION_ROWS)
        self._clear_results()
        self._log("New configuration loaded.")

    def _clear_output(self) -> None:
        self.output_lines = []
        self.output_log.object = ""

    def _clear_results(self) -> None:
        self.current_case = None
        self.initialized_rotor = None
        self.plot_fig = self._blank_figure("Initialize rotor, then calculate hover.")
        self.plot_pane.object = self.plot_fig
        self.summary_table.value = pd.DataFrame(columns=["Metric", "Value"])
        self.load_table.value = pd.DataFrame()
        self._sync_enabled_state()
        self._update_plot_options()

    def _initialize_rotor(self) -> None:
        try:
            config = self._current_config()
            rows = self._station_rows()
            self.initialized_rotor = build_rotor_spec(config, rows, name="upper")
            self.current_case = None
            self._log("Initializing rotor...")
            self._log(
                "Tip Speed: "
                f"{self.initialized_rotor.tip_speed_m_s:.3f} [m/s] | "
                f"Tip Mach: {self.initialized_rotor.tip_speed_mach:.3f} | "
                f"Rotor Disk Area: {self.initialized_rotor.disk_area_m2:.4f} [m2] | "
                f"Solidity: {self.initialized_rotor.solidity:.4f}"
            )
            total_blades = self.initialized_rotor.num_blades * (2 if config["rotor_system_type"] == "coaxial" else 1)
            self._log(f"Blades per rotor: {self.initialized_rotor.num_blades} | Total blades: {total_blades}")
            self._log(f"Geometry control points: {len(rows)} | Solver blade elements per rotor: {config['blade_element_count']}")
        except Exception as err:
            self.initialized_rotor = None
            self.current_case = None
            self._log(f"ERROR - {err}")
        self._sync_enabled_state()
        self._update_plot_options()

    def _calculate_hover(self) -> None:
        try:
            config = self._current_config()
            case = run_hover_case(config, self._station_rows())
            self.current_case = case
            self._log("")
            self._log("Calculating Hover Conditions...")
            self._log_hover_result(case, config)
            self._update_summary_table(case, config)
            self._update_load_table(case)
            self._generate_plot()
        except Exception as err:
            self.current_case = None
            self._log(f"ERROR - {err}")
        self._sync_enabled_state()
        self._update_plot_options()

    def _calculate_forward_flight(self) -> None:
        if self.current_case is None:
            self._log("Calculate hover before forward flight.")
            return
        if self.current_case.system_type != "single":
            self._log("Legacy forward-flight estimates are available for single-rotor cases only.")
            return
        try:
            config = self._current_config()
            hover = primary_hover_result(self.current_case)
            estimate = estimate_forward_flight(
                self.current_case.rotor,
                hover,
                velocity_m_s=float(config["velocity_kmh"]) / 3.6,
                density_kg_m3=float(config["density"]),
                flat_plate_area_m2=float(config["fpa"]),
            )
            self._log("")
            self._log("Calculating Forward Flight Conditions...")
            self._log(
                f"Velocity: {config['velocity_kmh']:.2f} [km/hr] | "
                f"Downwash Velocity Ratio: {estimate.downwash_velocity_ratio:.3f}"
            )
            self._log(
                f"SHP Induced: {estimate.induced_power_W * 0.00134102209:.3f} | "
                f"SHP Profile: {estimate.profile_power_W * 0.00134102209:.3f} | "
                f"SHP Parasite: {estimate.parasite_power_W * 0.00134102209:.3f} | "
                f"SHP Total: {estimate.total_power_W * 0.00134102209:.3f}"
            )
            if config["propulsion_model"] == "electric":
                electric = {
                    "propulsion_model": "electric",
                    **config,
                }
                summary = electric_summary(estimate.total_power_W, electric)
                input_power_W = summary["electric_input_W"]
                endurance_min = summary["hover_endurance_min"]
                self._log(f"Electric Input: {input_power_W / 1000.0:.3f} [kW] | Endurance: {endurance_min:.2f} [min]")
            else:
                engine_input_W = estimate.total_power_W / (1.0 - float(config["transmission_loss"]))
                fuel_flow = engine_input_W / 1000.0 * float(config["specific_fuel_consumption_kg_per_kWh"])
                endurance_hr = float(config["fuel_capacity_kg"]) / fuel_flow if fuel_flow > 0 else 0.0
                self._log(f"Engine Input: {engine_input_W / 1000.0:.3f} [kW] | Fuel Flow: {fuel_flow:.3f} [kg/hr] | Endurance: {endurance_hr:.3f} [hr]")
        except Exception as err:
            self._log(f"ERROR - {err}")

    def _log_hover_result(self, case: HoverCase, config: dict[str, Any]) -> None:
        result = case.result
        if isinstance(result, CoaxialHoverResult):
            self._log(
                f"Total Thrust: {result.total_thrust_N:.3f} [N] | "
                f"Total Shaft Power: {result.total_power_W / 1000.0:.3f} [kW] | "
                f"Interference Loss: {result.interference_loss_ratio:.3f}"
            )
            self._log(
                f"Upper Required Collective: {result.upper.collective_pitch_deg:.3f} [deg] | "
                f"Lower Required Collective: {result.lower.collective_pitch_deg:.3f} [deg] | "
                f"Wake Velocity: {result.wake_velocity_m_s:.3f} [m/s]"
            )
        else:
            self._log(
                f"Required Collective: {result.collective_pitch_deg:.3f} [deg] | "
                f"Induced Velocity: {result.mean_induced_velocity_m_s:.3f} [m/s] | "
                f"Thrust: {result.total_thrust_N:.3f} [N]"
            )
            self._log(
                f"Shaft Power: {result.power_W / 1000.0:.3f} [kW] | "
                f"Induced: {result.induced_power_W / 1000.0:.3f} [kW] | "
                f"Profile: {result.profile_power_W / 1000.0:.3f} [kW]"
            )
            self._log(
                f"Ct: {result.ct:.5f} | Cp: {result.cp:.5f} | "
                f"Figure of Merit: {result.figure_of_merit:.3f} | Mean Loss: {result.mean_loss_factor:.3f}"
            )
        prop = propulsion_summary(case, config)
        if config["propulsion_model"] == "electric":
            self._log(
                f"Electric Input: {prop['electric_input_W'] / 1000.0:.3f} [kW] | "
                f"Usable Battery: {prop['usable_energy_Wh']:.1f} [Wh] | "
                f"Hover Endurance: {prop['hover_endurance_min']:.2f} [min]"
            )
        else:
            self._log(
                f"Engine Input: {prop['engine_input_W'] / 1000.0:.3f} [kW] | "
                f"Fuel Flow: {prop['fuel_flow_kg_hr']:.3f} [kg/hr] | "
                f"Hover Endurance: {prop['hover_endurance_hr']:.3f} [hr]"
            )

    def _update_summary_table(self, case: HoverCase, config: dict[str, Any]) -> None:
        rows: list[dict[str, str]] = [
            {"Metric": "Rotor System", "Value": "Coaxial" if case.system_type == "coaxial" else "Single rotor"},
            {"Metric": "Total Thrust [N]", "Value": f"{total_hover_thrust_N(case):.3f}"},
            {"Metric": "Shaft Power [kW]", "Value": f"{total_hover_power_W(case) / 1000.0:.3f}"},
        ]

        result = case.result
        if isinstance(result, CoaxialHoverResult):
            rows.extend(
                [
                    {"Metric": "Upper Power [kW]", "Value": f"{result.upper.power_W / 1000.0:.3f}"},
                    {"Metric": "Lower Power [kW]", "Value": f"{result.lower.power_W / 1000.0:.3f}"},
                    {"Metric": "Interference Delta [W]", "Value": f"{result.interference_power_delta_W:.3f}"},
                    {"Metric": "Interference Loss", "Value": f"{result.interference_loss_ratio:.4f}"},
                    {"Metric": "Wake Radius [m]", "Value": f"{result.wake_radius_m:.4f}"},
                    {"Metric": "Wake Velocity [m/s]", "Value": f"{result.wake_velocity_m_s:.4f}"},
                ]
            )
            hover_results = [("Upper", result.upper), ("Lower", result.lower)]
        else:
            hover_results = [("Rotor", result)]

        for label, hover in hover_results:
            rows.extend(
                [
                    {"Metric": f"{label} Required Collective [deg]", "Value": f"{hover.collective_pitch_deg:.3f}"},
                    {"Metric": f"{label} Torque [Nm]", "Value": f"{hover.total_torque_Nm:.4f}"},
                    {"Metric": f"{label} Figure of Merit", "Value": f"{hover.figure_of_merit:.4f}"},
                    {"Metric": f"{label} Ct", "Value": f"{hover.ct:.6f}"},
                    {"Metric": f"{label} Cp", "Value": f"{hover.cp:.6f}"},
                    {"Metric": f"{label} Solidity", "Value": f"{hover.solidity:.5f}"},
                    {"Metric": f"{label} Root Flap Moment [Nm/blade]", "Value": f"{hover.root_flap_bending_moment_Nm_per_blade:.4f}"},
                    {"Metric": f"{label} Root Lag Moment [Nm/blade]", "Value": f"{hover.root_lag_moment_Nm_per_blade:.4f}"},
                    {"Metric": f"{label} Pitch Moment [Nm/blade]", "Value": f"{hover.aerodynamic_pitching_moment_Nm_per_blade:.4f}"},
                ]
            )

        prop = propulsion_summary(case, config)
        if config["propulsion_model"] == "electric":
            rows.extend(
                [
                    {"Metric": "Propulsion Model", "Value": "Electric"},
                    {"Metric": "Transmission Loss", "Value": f"{float(config['transmission_loss']):.3f}"},
                    {"Metric": "Electric Input [kW]", "Value": f"{prop['electric_input_W'] / 1000.0:.3f}"},
                    {"Metric": "Usable Battery [Wh]", "Value": f"{prop['usable_energy_Wh']:.1f}"},
                    {"Metric": "Hover Endurance [min]", "Value": f"{prop['hover_endurance_min']:.2f}"},
                ]
            )
        else:
            rows.extend(
                [
                    {"Metric": "Propulsion Model", "Value": "Fossil fuel"},
                    {"Metric": "Transmission Loss", "Value": f"{float(config['transmission_loss']):.3f}"},
                    {"Metric": "Engine Input [kW]", "Value": f"{prop['engine_input_W'] / 1000.0:.3f}"},
                    {"Metric": "Fuel Flow [kg/hr]", "Value": f"{prop['fuel_flow_kg_hr']:.3f}"},
                    {"Metric": "Hover Endurance [hr]", "Value": f"{prop['hover_endurance_hr']:.3f}"},
                ]
            )
        self.summary_table.value = pd.DataFrame(rows)

    def _update_load_table(self, case: HoverCase) -> None:
        result = case.result
        if isinstance(result, CoaxialHoverResult):
            upper = pd.DataFrame(load_rows_for_result(result.upper))
            upper.insert(0, "rotor", "upper")
            lower = pd.DataFrame(load_rows_for_result(result.lower))
            lower.insert(0, "rotor", "lower")
            self.load_table.value = pd.concat([upper, lower], ignore_index=True)
        else:
            self.load_table.value = pd.DataFrame(load_rows_for_result(result))

    def _generate_plot(self) -> None:
        if not self.plot_select.value:
            return
        try:
            plot_name = self.plot_select.value
            if plot_name == "Blade Geometry":
                fig = self._plot_blade_geometry()
            elif self.current_case is None:
                self._log("Calculate hover before generating this plot.")
                return
            elif plot_name == "Power Breakdown":
                fig = self._plot_power_breakdown(self.current_case)
            elif plot_name == "Radial Loads":
                fig = self._plot_radial_loads(self.current_case)
            elif plot_name == "Alpha, Re, Mach vs Radius":
                fig = self._plot_alpha_re_mach(self.current_case)
            elif plot_name == "Induced Velocity and Loss":
                fig = self._plot_induced_loss(self.current_case)
            elif plot_name == "Section Coefficients vs Radius":
                fig = self._plot_section_coefficients(self.current_case)
            elif plot_name == "Pitch Moment vs Radius":
                fig = self._plot_pitch_moment(self.current_case)
            elif plot_name == "Cumulative Thrust and Power":
                fig = self._plot_cumulative_loads(self.current_case)
            elif plot_name == "Coaxial Interference":
                fig = self._plot_coaxial_interference(self.current_case)
            elif plot_name == "Electric Range vs Velocity":
                fig = self._plot_electric_range(self.current_case)
            elif plot_name == "Fuel Range, Endurance vs Velocity":
                fig = self._plot_fuel_range(self.current_case)
            elif plot_name == "Forward Flight Powers vs Velocity":
                fig = self._plot_forward_powers(self.current_case)
            else:
                fig = self._blank_figure("No plot selected.")
            self.plot_fig = fig
            self.plot_pane.object = fig
            self.result_tabs.active = 0
        except Exception as err:
            self._log(f"ERROR - {err}")

    def _plot_blade_geometry(self):
        frame = self.station_table.value.copy()
        fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)
        ax1.plot(frame["r_over_R"], frame["chord_m"], marker="o", label="Chord [m]")
        ax1.set_xlabel("r/R")
        ax1.set_ylabel("Chord [m]")
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(frame["r_over_R"], frame["twist_deg"], marker="s", color="tab:red", label="Twist [deg]")
        ax2.set_ylabel("Twist [deg]")
        fig.suptitle("Blade Geometry")
        fig.legend(loc="upper right")
        return self._finish_plot(fig)

    def _plot_power_breakdown(self, case: HoverCase):
        result = case.result
        labels = []
        induced = []
        profile = []
        if isinstance(result, CoaxialHoverResult):
            for label, hover in (("Upper", result.upper), ("Lower", result.lower)):
                labels.append(label)
                induced.append(hover.induced_power_W / 1000.0)
                profile.append(hover.profile_power_W / 1000.0)
        else:
            labels = ["Rotor"]
            induced = [result.induced_power_W / 1000.0]
            profile = [result.profile_power_W / 1000.0]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.bar(x, induced, label="Induced")
        ax.bar(x, profile, bottom=induced, label="Profile")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Shaft Power [kW]")
        ax.set_title("Hover Power Breakdown")
        ax.grid(True, axis="y")
        ax.legend()
        return self._finish_plot(fig)

    def _plot_radial_loads(self, case: HoverCase):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax.plot(rows["r_over_R"], rows["dT_N"], label=f"{label} dT [N/blade]")
            ax.plot(rows["r_over_R"], rows["dQ_Nm"], linestyle="--", label=f"{label} dQ [Nm/blade]")
        ax.set_title("Per-Blade Radial Loads")
        ax.set_xlabel("r/R")
        ax.set_ylabel("Element Load [N/blade, Nm/blade]")
        ax.grid(True)
        ax.legend()
        return self._finish_plot(fig)

    def _plot_alpha_re_mach(self, case: HoverCase):
        fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax1.plot(rows["r_over_R"], rows["alpha_deg"], label=f"{label} alpha")
        ax1.set_xlabel("r/R")
        ax1.set_ylabel("Alpha [deg]")
        ax1.grid(True)
        ax2 = ax1.twinx()
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax2.plot(rows["r_over_R"], rows["reynolds"], linestyle="--", label=f"{label} Re")
            ax2.plot(rows["r_over_R"], rows["mach"], linestyle=":", label=f"{label} Mach")
        ax2.set_ylabel("Reynolds / Mach")
        fig.suptitle("Section Flow Conditions")
        fig.legend(loc="upper right")
        return self._finish_plot(fig)

    def _plot_induced_loss(self, case: HoverCase):
        fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax1.plot(rows["r_over_R"], rows["induced_velocity_m_s"], label=f"{label} vi")
        ax1.set_xlabel("r/R")
        ax1.set_ylabel("Induced Velocity [m/s]")
        ax1.grid(True)
        ax2 = ax1.twinx()
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax2.plot(rows["r_over_R"], rows["loss_factor"], linestyle="--", label=f"{label} loss")
        ax2.set_ylabel("Loss Factor")
        fig.suptitle("Induced Velocity and Root/Tip Loss")
        fig.legend(loc="upper right")
        return self._finish_plot(fig)

    def _plot_section_coefficients(self, case: HoverCase):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax.plot(rows["r_over_R"], rows["cl"], label=f"{label} Cl")
            ax.plot(rows["r_over_R"], rows["cd"], linestyle="--", label=f"{label} Cd")
            ax.plot(rows["r_over_R"], rows["cm"], linestyle=":", label=f"{label} Cm")
        ax.set_title("Section Coefficients")
        ax.set_xlabel("r/R")
        ax.set_ylabel("Coefficient")
        ax.grid(True)
        ax.legend()
        return self._finish_plot(fig)

    def _plot_pitch_moment(self, case: HoverCase):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax.plot(rows["r_over_R"], rows["pitch_moment_Nm"], label=label)
        ax.axhline(0.0, color=THEME["muted"], linewidth=0.8)
        ax.set_title("Element Pitch Moment")
        ax.set_xlabel("r/R")
        ax.set_ylabel("Pitch Moment [Nm]")
        ax.grid(True)
        ax.legend()
        return self._finish_plot(fig)

    def _plot_cumulative_loads(self, case: HoverCase):
        fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax1.plot(rows["r_over_R"], rows["dT_N"].cumsum(), label=f"{label} thrust")
        ax1.set_xlabel("r/R")
        ax1.set_ylabel("Cumulative Thrust [N/blade]")
        ax1.grid(True)
        ax2 = ax1.twinx()
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            ax2.plot(rows["r_over_R"], rows["dP_W"].cumsum(), linestyle="--", label=f"{label} power")
        ax2.set_ylabel("Cumulative Power [W/blade]")
        fig.suptitle("Cumulative Per-Blade Load Build-Up")
        fig.legend(loc="upper right")
        return self._finish_plot(fig)

    def _plot_coaxial_interference(self, case: HoverCase):
        if not isinstance(case.result, CoaxialHoverResult):
            raise ValueError("Coaxial interference plot requires a coaxial result.")
        result = case.result
        labels = ["Isolated Pair", "Coaxial Pair", "Delta"]
        values = [
            (result.isolated_upper.power_W + result.isolated_lower.power_W) / 1000.0,
            result.total_power_W / 1000.0,
            result.interference_power_delta_W / 1000.0,
        ]
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:red"])
        ax.set_ylabel("Power [kW]")
        ax.set_title("Coaxial Interference Power")
        ax.grid(True, axis="y")
        return self._finish_plot(fig)

    def _plot_forward_powers(self, case: HoverCase):
        self._require_single_rotor(case)
        hover = primary_hover_result(case)
        cfg = self._current_config()
        estimates = velocity_sweep(case.rotor, hover, density_kg_m3=float(cfg["density"]), flat_plate_area_m2=float(cfg["fpa"]))
        velocities = [estimate.velocity_m_s * 3.6 for estimate in estimates]
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.plot(velocities, [estimate.induced_power_W * 0.00134102209 for estimate in estimates], label="Induced")
        ax.plot(velocities, [estimate.profile_power_W * 0.00134102209 for estimate in estimates], label="Profile")
        ax.plot(velocities, [estimate.parasite_power_W * 0.00134102209 for estimate in estimates], label="Parasite")
        ax.plot(velocities, [estimate.total_power_W * 0.00134102209 for estimate in estimates], label="Total", color=THEME["text"])
        ax.set_title("Forward Flight Powers")
        ax.set_xlabel("Free Stream Velocity [km/hr]")
        ax.set_ylabel("Shaft Horsepower [SHP]")
        ax.grid(True)
        ax.legend()
        return self._finish_plot(fig)

    def _plot_electric_range(self, case: HoverCase):
        self._require_single_rotor(case)
        cfg = self._current_config()
        hover = primary_hover_result(case)
        estimates = velocity_sweep(case.rotor, hover, density_kg_m3=float(cfg["density"]), flat_plate_area_m2=float(cfg["fpa"]))
        velocities, endurance, flight_range = electric_range_sweep(estimates, cfg)
        fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)
        ax1.plot(velocities, endurance, label="Endurance")
        ax1.set_xlabel("Velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(velocities, flight_range, color="tab:orange", label="Range")
        ax2.set_ylabel("Range [km]")
        fig.suptitle("Electric Range and Endurance")
        fig.legend(loc="upper right")
        return self._finish_plot(fig)

    def _plot_fuel_range(self, case: HoverCase):
        self._require_single_rotor(case)
        cfg = self._current_config()
        hover = primary_hover_result(case)
        estimates = velocity_sweep(case.rotor, hover, density_kg_m3=float(cfg["density"]), flat_plate_area_m2=float(cfg["fpa"]))
        velocities, endurance, flight_range = fossil_range_sweep(estimates, hover, cfg)
        fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)
        ax1.plot(velocities, endurance, color="tab:red", label="Endurance")
        ax1.set_xlabel("Free Stream Velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(velocities, flight_range, label="Range")
        ax2.set_ylabel("Range [km]")
        fig.suptitle("Fuel Range and Endurance")
        fig.legend(loc="upper right")
        return self._finish_plot(fig)

    def _iter_hover_results(self, case: HoverCase):
        if isinstance(case.result, CoaxialHoverResult):
            return [("Upper", case.result.upper), ("Lower", case.result.lower)]
        return [("Rotor", case.result)]

    def _require_single_rotor(self, case: HoverCase) -> None:
        if case.system_type != "single":
            raise ValueError("Legacy forward-flight plots are available for single-rotor cases only.")

    def _blank_figure(self, text: str):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.text(0.5, 0.5, text, ha="center", va="center", color=THEME["muted"])
        ax.set_axis_off()
        return self._finish_plot(fig)

    def _finish_plot(self, fig):
        fig.patch.set_facecolor(THEME["plot"])
        for ax in fig.axes:
            ax.set_facecolor(THEME["plot"])
            ax.tick_params(colors=THEME["muted"])
            ax.xaxis.label.set_color(THEME["text"])
            ax.yaxis.label.set_color(THEME["text"])
            ax.title.set_color(THEME["text"])
            for spine in ax.spines.values():
                spine.set_color(THEME["border"])
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color(THEME["muted"])
            ax.grid(color=THEME["plot_grid"], alpha=0.55, linewidth=0.8)
            self._style_legend(ax.get_legend())

        if fig._suptitle is not None:
            fig._suptitle.set_color(THEME["text"])
        for legend in fig.legends:
            self._style_legend(legend)

        fig.tight_layout()
        return fig

    def _style_legend(self, legend) -> None:
        if legend is None:
            return
        legend.get_frame().set_facecolor(THEME["panel_alt"])
        legend.get_frame().set_edgecolor(THEME["border"])
        legend.get_frame().set_alpha(0.95)
        for text in legend.get_texts():
            text.set_color(THEME["text"])

    def _on_propulsion_changed(self) -> None:
        self._update_plot_options()
        if self.current_case is not None:
            self._update_summary_table(self.current_case, self._current_config())

    def _update_plot_options(self) -> None:
        options = [
            "Blade Geometry",
            "Power Breakdown",
            "Radial Loads",
            "Alpha, Re, Mach vs Radius",
            "Induced Velocity and Loss",
            "Section Coefficients vs Radius",
            "Pitch Moment vs Radius",
            "Cumulative Thrust and Power",
        ]
        if self.current_case and self.current_case.system_type == "coaxial":
            options.append("Coaxial Interference")
        if self.current_case and self.current_case.system_type == "single":
            options.append("Forward Flight Powers vs Velocity")
            if self._propulsion_model() == "electric":
                options.append("Electric Range vs Velocity")
            else:
                options.append("Fuel Range, Endurance vs Velocity")
        current = self.plot_select.value
        self.plot_select.options = options
        self.plot_select.value = current if current in options else options[0]

    def _sync_enabled_state(self) -> None:
        is_coaxial = self.rotor_system_type.value == "coaxial"
        for widget in (self.coaxial_spacing, self.coaxial_trim_mode, self.lower_collective_offset, self.lower_rotor_scale):
            widget.disabled = not is_coaxial
        self.headspeed_rpm.disabled = self.headspeed_input_mode.value != "rpm"
        self.tip_speed_mach.disabled = self.headspeed_input_mode.value != "tip_mach"
        self.station_table.disabled = self.geometry_mode.value != "station_table"
        self.calc_hover_btn.disabled = self.initialized_rotor is None
        self.generate_plot_btn.disabled = self.initialized_rotor is None and self.current_case is None
        self.calc_forward_btn.disabled = self.current_case is None or self.current_case.system_type != "single"

    def _reset_station_rows_from_uniform(self) -> None:
        cfg = self._current_config()
        self.station_table.value = pd.DataFrame(station_rows_from_uniform(cfg))
        self._log("Geometry control points reset from root/tip twist inputs.")

    def _log(self, text: str) -> None:
        prefix = datetime.now().strftime("%H:%M:%S")
        self.output_lines.append("" if text == "" else f"[{prefix}] {text}")
        self.output_log.object = self._render_output_log()

    def _render_output_log(self) -> str:
        self._log_render_count += 1
        anchor_id = f"pycopter-log-end-{self._log_render_count}"
        escaped_text = escape("\n".join(self.output_lines[-300:]))
        return (
            f'<pre class="pycopter-log-text">{escaped_text}</pre>'
            f'<span id="{anchor_id}"></span>'
            "<script>"
            "requestAnimationFrame(function(){"
            f'var anchor=document.getElementById("{anchor_id}");'
            "if(!anchor){return;}"
            'var log=anchor.closest(".pycopter-log");'
            "if(log){log.scrollTop=log.scrollHeight;}"
            'anchor.scrollIntoView({block:"end"});'
            "});"
            "</script>"
        )


def create_app():
    """Create a fresh Panel app session."""
    return PycopterWebApp().view()
