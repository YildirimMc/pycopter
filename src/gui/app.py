"""Panel Web UI for PyCopter.

The dashboard is a run manager rather than a form: one tab-gated inspector
column holds the settings of the subsystem being worked on, the canvas shows
the active run, and the right-hand rail keeps every run of the session as a
named, comparable record with a delta against a chosen baseline.
"""

from __future__ import annotations

import io
import json
import queue
import threading
import traceback
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import param

from pycopter import (
    CoaxialHoverResult,
    CoaxialSpec,
    ElementLoad,
    HoverResult,
    HoverSolver,
    OperatingPoint,
    solve_coaxial_hover,
    sweep_interference_loss,
)

from .calculations import (
    DEFAULT_CONFIG,
    DEFAULT_STATION_ROWS,
    CONFIG_VERSION,
    HoverCase,
    build_coaxial_spec,
    build_operating_point,
    build_solver_settings,
    build_rotor_spec,
    build_xfoil_provider,
    derived_geometry,
    electric_summary,
    electric_range_sweep,
    estimate_forward_flight,
    fossil_range_sweep,
    isa_atmosphere,
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
from .runs import RunRecord, RunStore, diff_inputs


# ---------------------------------------------------------------------------
# Layout contract
# ---------------------------------------------------------------------------
# Fixed-height app shell. The command bar and the log are pinned; the middle
# band divides the remaining height, and each of its regions scrolls
# internally, so changing tabs or expanding the log never resizes the page.
COMMAND_BAR_HEIGHT = 34
WARNING_BAR_HEIGHT = 28
ICON_RAIL_WIDTH = 60
INSPECTOR_WIDTH = 320
RUN_RAIL_WIDTH = 384
# 60 + 320 + 384 fixed regions plus this minimum canvas is the ~1420 px below
# which the body scrolls horizontally instead of clipping.
RESULT_MIN_WIDTH = 656
METRIC_STRIP_HEIGHT = 52
# The app menu is an absolutely positioned popup. Bokeh sizes a layout
# container from its model rather than from its content, so the popup carries
# an explicit size that grows when Preferences opens.
MENU_WIDTH = 268
MENU_HEIGHT = 440
MENU_PREFERENCES_HEIGHT = 240
# The log is a status strip first and a terminal second: it starts at the strip
# height and is dragged upwards to reveal the scrollback.
LOG_MIN_HEIGHT = 30
LOG_EXPANDED_HEIGHT = 196
LOG_STRIP_HEIGHT = 30
LOG_GRIP_HEIGHT = 7
LOG_TERMINAL_INSET = 10
# The result area never shrinks below this, so dragging cannot hide the canvas.
RESULT_MIN_HEIGHT = 300
# Figure inches are derived from CSS pixels at the 96 dpi CSS reference, while
# the pane rasterizes at PLOT_RENDER_DPI. The 1.5x ratio supersamples the PNG so
# it stays sharp on high-density displays while axis text keeps its true size.
PLOT_CSS_DPI = 96.0
PLOT_RENDER_DPI = 144
# Fallback plot size in inches, used before the browser reports the result-area
# size and in headless contexts such as the unit tests.
PLOT_FIGSIZE = (7.6, 6.4)
PLOT_MIN_SIZE_IN = (4.6, 3.6)
PLOT_MAX_SIZE_IN = (19.0, 13.0)
RESULT_TABLE_CONFIGURATION = {
    "clipboard": True,
    "clipboardCopyRowRange": "selected",
    "columnDefaults": {"editable": False},
}
COAXIAL_SPACING_SWEEP_MIN = 0.05
COAXIAL_SPACING_SWEEP_MAX = 1.50
DESIGN_SWEEP_POINTS = 17
CONTROL_SWEEP_POINTS = 31
MIN_SWEEP_GROSS_MASS_KG = 0.05
# How often the UI drains progress from the calculation thread.
WORKER_POLL_MS = 200
# Axis layout rules for multi-quantity plots, applied to the measured data:
#
# 1. Two quantities share a y-axis when their magnitudes are within this factor,
#    so the smaller one still spans at least a fifth of the axis.
# 2. Beyond that they get the left and right y-axis of the same panel.
# 3. A third, differently scaled quantity moves to its own stacked panel rather
#    than a third offset spine, which is hard to read.
SHARED_AXIS_MAX_RATIO = 5.0
MAX_PLOT_PANELS = 3
# Two curves on opposite autoscaled axes draw on top of each other when their
# shapes match; Reynolds and Mach are both linear in radius and coincide
# exactly. Such a pair is split across panels instead, because a single visible
# line silently hiding a second quantity is worse than an extra panel.
COINCIDENT_SHAPE_GAP = 0.03
# One colour per plotted quantity. Rotors are separated by line style and
# overlaid runs by opacity, so a quantity keeps its colour in every case.
SERIES_COLORS = ("#77b6ea", "#e0a24f", "#7fb069", "#a68ade")
SERIES_MARKERS = ("o", "s", "^", "D")
SERIES_MARKER_COUNT = 9
OVERLAY_DASHES = ((6, 3), (2, 2), (8, 2, 2, 2), (1, 2))
OVERLAY_ALPHA = 0.55

# Industry visual system: steel-blue accent, condensed headings over a
# humanist body face, square corners and hairline borders on a dark ground.
THEME = {
    "page": "#101215",
    "panel": "#171a1e",
    "panel_alt": "#1e2227",
    "field": "#121417",
    "field_hover": "#1c2026",
    "border": "#2c3238",
    "border_strong": "#454d56",
    "text": "#e4e9ef",
    "muted": "#8b96a3",
    "dim": "#616c78",
    "accent": "#5980a6",
    "accent_hi": "#7aa3cc",
    "accent_hover": "#6b93b8",
    "accent_tint": "rgba(89, 128, 166, 0.14)",
    "warning": "#c9973f",
    "danger": "#b4574d",
    "good": "#5f9e6a",
    "plot": "#131619",
    "plot_grid": "#2b3138",
}
# Barlow if the machine has it, otherwise the nearest stock condensed and
# humanist faces. The app must not depend on a font download.
HEADING_FONT = '"Barlow Condensed", "Roboto Condensed", "Arial Narrow", sans-serif'
BODY_FONT = 'Barlow, Inter, "Segoe UI", Arial, sans-serif'
MONO_FONT = 'Consolas, "Cascadia Mono", "Courier New", monospace'

# Unit suffixes rendered inside the right edge of an inspector field. Fields
# take a token from this map instead of putting the unit in the label, which
# keeps the label column readable at 320 px.
FIELD_UNITS = {
    "m": "m",
    "m2": "m²",
    "mm": "mm",
    "deg": "deg",
    "rpm": "rpm",
    "kg": "kg",
    "kgm3": "kg/m³",
    "m2s": "m²/s",
    "ms": "m/s",
    "s": "s",
    "n": "N",
    "w": "W",
    "wh": "Wh",
    "kgkwh": "kg/kWh",
    "kmh": "km/hr",
    "rr": "r/R",
    "zr": "z/R",
    "c": "c",
    "mach": "M",
    "re": "Re",
    "ratio": "-",
    "count": "n",
    "calc": "calc",
    "pct": "%",
    "px": "px",
    "dpi": "dpi",
    "celsius": "°C",
}

INSPECTOR_TABS = ("ROTOR", "BLADE", "OPER", "SOLVER", "XFOIL", "PROP")
SHOWING_OPTIONS = {
    "UPPER": "upper",
    "LOWER": "lower",
    "TOTAL": "total",
    "Δ ISOLATED": "delta",
}
LOADS_SHADING_OPTIONS = {"ALPHA": "alpha", "CL/CD": "clcd", "OFF": "off"}
LOG_SEVERITIES = ("ALL", "WARN", "ERROR", "SOLVER")
# Column groups of the blade element loads table, in reading order.
LOAD_COLUMN_GROUPS = (
    ("GEOMETRY", ("r_m", "dr_m", "chord_m", "twist_deg", "collective_deg")),
    (
        "LOCAL FLOW",
        (
            "phi_deg",
            "alpha_deg",
            "reynolds",
            "mach",
            "loss_factor",
            "induced_velocity_m_s",
            "external_axial_velocity_m_s",
        ),
    ),
    ("SECTION COEFFICIENTS", ("cl", "cd", "cm", "alpha_clamped", "CLAMPED")),
    (
        "INTEGRATED LOADS",
        (
            "dL_N",
            "dD_N",
            "dT_N",
            "dQ_Nm",
            "dP_W",
            "normal_force_N_per_m",
            "tangential_force_N_per_m",
            "pitch_moment_Nm",
        ),
    ),
)
# Metric strip cells: (key, label, unit, goodness). Goodness decides the delta
# colour: 'up' means a rise is an improvement, 'down' means a rise is worse,
# 'zero' means closer to zero is better, and 'none' stays neutral.
SINGLE_ROTOR_METRICS = (
    ("total_thrust_N", "TOTAL THRUST", "N", "none"),
    ("power_W", "SHAFT POWER", "W", "down"),
    ("collective_pitch_deg", "COLLECTIVE", "deg", "none"),
    ("figure_of_merit", "FIGURE OF MERIT", "", "up"),
    ("ct_cp", "Ct / Cp", "", "none"),
    ("mean_loss_factor", "MEAN LOSS", "", "up"),
    ("aircraft_yaw_torque_Nm", "YAW TORQUE", "Nm", "zero"),
    ("endurance", "ENDURANCE", "", "up"),
)
COAXIAL_METRICS = (
    ("total_thrust_N", "TOTAL THRUST", "N", "none"),
    ("total_power_W", "TOTAL POWER", "W", "down"),
    ("interference_loss_ratio", "INTERFERENCE", "%", "down"),
    ("active_collective_deg", "COLLECTIVE", "deg", "none"),
    ("net_aircraft_yaw_torque_Nm", "NET YAW", "Nm", "zero"),
    ("active_figure_of_merit", "FIGURE OF MERIT", "", "up"),
    ("wake_velocity_m_s", "WAKE VELOCITY", "m/s", "none"),
    ("wake_radius_m", "WAKE RADIUS", "m", "none"),
)


# Panel renders every component into its own shadow root, so a rule written as
# ".container .bk-input" never reaches the widget: the container is outside the
# boundary. Rules that have to touch a widget's internals are therefore passed
# to that widget as its own stylesheet, where plain selectors resolve. Rules
# that only style a host element, or that stay inside one component, live in
# RAW_CSS and are inherited into every shadow root.
FIELD_CSS = """
label,
.bk-input-group > label {
    font-family: %(heading_font)s;
    font-size: 10px;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: %(muted)s;
    margin-bottom: 1px;
}
.bk-input-group { position: relative; }
.bk-input {
    height: 26px;
    min-height: 26px;
    padding: 2px 34px 2px 7px;
    font-size: 12px;
    border-radius: 0 !important;
}
select.bk-input { padding-right: 22px; }
/* Direct entry only: the spinner buttons would sit under the unit chip. */
.bk-spin-btn,
.bk-spin-wrapper > .bk-spin-btn-up,
.bk-spin-wrapper > .bk-spin-btn-down { display: none !important; }
""" % {**THEME, "heading_font": HEADING_FONT}

EDITED_FIELD_CSS = """
.bk-input {
    border-color: %(accent)s !important;
    box-shadow: inset 0 0 0 1px %(accent)s !important;
}
""" % THEME

DERIVED_FIELD_CSS = """
.bk-input {
    color: %(muted)s !important;
    background: %(panel)s !important;
}
""" % THEME

RAIL_TABS_CSS = """
.bk-btn-group { flex-direction: column; width: 100%%; gap: 0; }
.bk-btn-group > button.bk-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 5px;
    height: 52px;
    width: 100%%;
    font-family: %(heading_font)s;
    font-size: 10px !important;
    letter-spacing: .1em;
    color: %(muted)s !important;
    background: transparent !important;
    border: 0 !important;
    border-left: 2px solid transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
.bk-btn-group > button.bk-btn::before {
    content: "";
    width: 16px;
    height: 16px;
    border: 1px solid %(border_strong)s;
    background: transparent;
}
.bk-btn-group > button.bk-btn:hover {
    background: %(panel_alt)s !important;
    color: %(text)s !important;
}
.bk-btn-group > button.bk-btn.bk-active {
    background: %(accent_tint)s !important;
    border-left: 2px solid %(accent)s !important;
    color: %(text)s !important;
}
.bk-btn-group > button.bk-btn.bk-active::before {
    background: %(accent)s;
    border-color: %(accent)s;
}
""" % {**THEME, "heading_font": HEADING_FONT}

SEGMENT_CSS = """
.bk-btn-group { gap: 0; }
.bk-btn-group > button.bk-btn {
    font-family: %(heading_font)s;
    font-size: 10px !important;
    letter-spacing: .1em;
    border-radius: 0 !important;
    color: %(muted)s !important;
    background: %(panel)s !important;
    border-color: %(border)s !important;
    min-height: 22px;
}
.bk-btn-group > button.bk-btn.bk-active {
    background: %(accent_tint)s !important;
    color: %(accent_hi)s !important;
    border-color: %(accent)s !important;
}
""" % {**THEME, "heading_font": HEADING_FONT}

LOG_FILTER_CSS = """
.bk-btn-group { flex-direction: column; gap: 0; }
.bk-btn-group > button.bk-btn {
    font-family: %(heading_font)s;
    font-size: 10px !important;
    letter-spacing: .1em;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: %(muted)s !important;
    min-height: 20px;
    padding: 1px 8px !important;
}
.bk-btn-group > button.bk-btn.bk-active {
    color: %(accent_hi)s !important;
    background: %(accent_tint)s !important;
}
""" % {**THEME, "heading_font": HEADING_FONT}

BRAND_CSS = """
button.bk-btn {
    font-family: %(heading_font)s;
    font-size: 14px !important;
    letter-spacing: .16em;
    text-transform: uppercase;
    background: transparent !important;
    border: 0 !important;
    border-right: 1px solid %(border)s !important;
    box-shadow: none !important;
    height: %(command_bar)dpx;
    border-radius: 0 !important;
    color: %(text)s !important;
}
button.bk-btn:hover { background: %(panel_alt)s !important; }
""" % {**THEME, "heading_font": HEADING_FONT, "command_bar": COMMAND_BAR_HEIGHT}

MENU_ITEM_CSS = """
button.bk-btn,
a.bk-btn,
.bk-btn {
    width: 100%%;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-size: 12px !important;
    padding: 5px 12px !important;
    color: %(text)s !important;
    white-space: pre;
}
button.bk-btn:hover,
a.bk-btn:hover,
.bk-btn:hover { background: %(accent_tint)s !important; }
""" % THEME

PRIMARY_BUTTON_CSS = """
button.bk-btn {
    background: %(accent)s !important;
    border-color: %(accent)s !important;
    color: #0e1216 !important;
    font-family: %(heading_font)s;
    font-size: 11px !important;
    letter-spacing: .12em;
    font-weight: 600;
}
button.bk-btn:hover { background: %(accent_hover)s !important; }
button.bk-btn:disabled {
    background: %(panel_alt)s !important;
    color: %(dim)s !important;
    border-color: %(border)s !important;
}
""" % {**THEME, "heading_font": HEADING_FONT}

CONDENSED_BUTTON_CSS = """
button.bk-btn,
a.bk-btn {
    font-family: %(heading_font)s;
    font-size: 10px !important;
    letter-spacing: .12em;
}
""" % {**THEME, "heading_font": HEADING_FONT}

DANGER_BUTTON_CSS = CONDENSED_BUTTON_CSS + (
    """
button.bk-btn {
    color: %(danger)s !important;
    border-color: %(danger)s !important;
}
"""
    % THEME
)

COLLAPSE_BUTTON_CSS = """
button.bk-btn {
    font-family: %(heading_font)s;
    font-size: 12px !important;
    letter-spacing: .12em;
    color: %(muted)s !important;
    background: transparent !important;
    border: 0 !important;
    border-top: 1px solid %(border)s !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    width: 100%%;
}
""" % {**THEME, "heading_font": HEADING_FONT}

WARNING_BUTTON_CSS = """
button.bk-btn {
    font-family: %(heading_font)s;
    font-size: 10px !important;
    letter-spacing: .1em;
    min-height: 20px;
    border-color: %(warning)s !important;
    color: %(warning)s !important;
    background: transparent !important;
}
""" % {**THEME, "heading_font": HEADING_FONT}

# The :host prefix raises specificity above the inherited ".bk-input-group
# label" rule, which paints every widget label muted.
CHECKBOX_CSS = """
:host,
:host *,
:host label,
:host .bk-checkbox-label,
:host .bk-input-group label,
:host .bk-input-group > label {
    color: %(text)s !important;
    opacity: 1 !important;
    font-size: 11px;
    text-transform: none;
    letter-spacing: 0;
}
input[type="checkbox"] { accent-color: %(accent)s; }
""" % THEME

TERMINAL_CSS = """
.xterm {
    height: 100%% !important;
    font-family: %(mono_font)s;
    font-size: 12px;
    background: %(field)s;
    border: 1px solid %(border)s;
    padding: 2px 6px;
}
.xterm-viewport,
.xterm-screen { background: %(field)s !important; }
.xterm-viewport { overflow-y: auto !important; }
.xterm-rows { user-select: text; }
""" % {**THEME, "mono_font": MONO_FONT}

PLOT_FRAME_CSS = """
img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}
"""

RESULT_TABLE_CSS = """
.tabulator-tableholder { overflow-x: auto !important; }
.tabulator-col-group .tabulator-col-title {
    font-family: %(heading_font)s;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: %(accent_hi)s !important;
}
""" % {**THEME, "heading_font": HEADING_FONT}

LOAD_TABLE_CSS = RESULT_TABLE_CSS + """
.tabulator-table { min-width: max-content; }
"""


def _unit_stylesheet(token: str) -> str:
    """Right-aligned unit chip drawn inside a field's own shadow root."""
    text = FIELD_UNITS.get(token, token)
    return (
        ".bk-input-group::after {"
        f' content: "{text}";'
        " position: absolute;"
        " right: 7px;"
        " bottom: 6px;"
        " font-size: 10px;"
        f" color: {THEME['dim']};"
        " pointer-events: none;"
        " line-height: 1;"
        " }"
    )


_BASE_CSS = """
html,
body {
    background: %(page)s;
    /* The dashboard is a fixed-height app shell: every region scrolls
       internally so expanding a section never grows or clips the page. */
    height: 100%%;
    overflow: hidden;
    font-family: %(body_font)s;
}
.pycopter-shell {
    font-family: %(body_font)s;
    color: %(text)s;
    height: 100%%;
    min-height: 0;
    background: %(page)s;
}
/* Panel wraps each child in its own div; these keep the flex chain able to
   shrink so the inner scroll containers, not the page, absorb overflow. */
.pycopter-shell > div,
.pycopter-body > div,
.pycopter-results > div,
.pycopter-inspector > div,
.pycopter-runrail > div {
    min-height: 0;
}
.pycopter-body {
    min-height: 0;
    overflow-x: auto;
    overflow-y: hidden;
    gap: 0;
}
/* Internally scrolling regions: they take the height their parent column has
   left over, and absorb their own overflow instead of growing the page. */
.pycopter-column {
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    flex: 1 1 auto;
}
.pycopter-results {
    min-width: %(result_min_width)dpx;
    min-height: 0;
    flex: 1 1 auto;
    border-left: 1px solid %(border)s;
    border-right: 1px solid %(border)s;
}
.pycopter-inspector,
.pycopter-runrail {
    background: %(panel)s;
}

/* ---- command bar ---- */
.pycopter-cmdbar {
    flex: 0 0 auto;
    height: %(command_bar)dpx;
    background: %(panel)s;
    border-bottom: 1px solid %(border)s;
    overflow: visible;
    align-items: center;
    gap: 0;
}
.pycopter-crumb {
    font-size: 11px;
    color: %(muted)s;
    padding: 0 12px;
    white-space: nowrap;
}
.pycopter-crumb .active {
    color: %(text)s;
    font-weight: 600;
}
.pycopter-tag {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    font-family: %(heading_font)s;
    font-size: 10px;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: %(accent_hi)s;
    border: 1px solid %(border_strong)s;
    background: %(accent_tint)s;
}
.pycopter-status {
    font-size: 11px;
    color: %(muted)s;
    text-align: right;
    padding: 0 12px;
    white-space: nowrap;
}
.pycopter-status .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 5px;
    background: %(dim)s;
}
.pycopter-status .dot.idle { background: %(good)s; }
.pycopter-status .dot.running { background: %(accent_hi)s; }
.pycopter-status .dot.failed { background: %(danger)s; }
.pycopter-showing-label {
    font-family: %(heading_font)s;
    font-size: 10px;
    letter-spacing: .12em;
    color: %(muted)s;
    text-transform: uppercase;
    padding: 0 8px 0 4px;
}

/* ---- app menu ---- */
.pycopter-menu-sep {
    height: 1px;
    background: %(border)s;
    margin: 4px 0;
}
.pycopter-menu-note {
    font-size: 10px;
    color: %(dim)s;
    padding: 2px 12px 6px;
}
.pycopter-menu input[type="file"] {
    font-size: 11px;
    padding: 2px 10px;
    color: %(muted)s;
}

/* ---- warning bar ---- */
.pycopter-warnbar {
    flex: 0 0 auto;
    height: %(warning_bar)dpx;
    align-items: center;
    background: rgba(201, 151, 63, .14);
    border-bottom: 1px solid %(warning)s;
    gap: 0;
}
.pycopter-warnbar-text {
    font-size: 11px;
    color: %(warning)s;
    padding: 0 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
/* ---- icon rail ---- */
.pycopter-rail {
    width: %(rail_width)dpx;
    flex: 0 0 auto;
    background: %(panel)s;
    border-right: 1px solid %(border)s;
}

/* ---- inspector ---- */
.pycopter-inspector {
    width: %(inspector_width)dpx;
    border-right: 1px solid %(border)s;
}
.pycopter-section-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: %(panel_alt)s;
    border-bottom: 1px solid %(border)s;
    padding: 6px 10px;
    font-family: %(heading_font)s;
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: %(text)s;
}
.pycopter-section-head .meta {
    font-family: %(body_font)s;
    font-size: 10px;
    letter-spacing: 0;
    text-transform: none;
    color: %(accent_hi)s;
}
.pycopter-subhead {
    font-family: %(heading_font)s;
    font-size: 10px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: %(muted)s;
    border-bottom: 1px solid %(border)s;
    padding-bottom: 3px;
    margin: 4px 0 2px;
}
.pycopter-inspector-foot {
    flex: 0 0 auto;
    border-top: 1px solid %(border)s;
    background: %(panel)s;
    padding: 6px;
    gap: 6px;
}
/* ---- MPI worker strip ---- */
.pycopter-mpi {
    display: flex;
    gap: 3px;
    margin: 2px 0 4px;
}
.pycopter-mpi div {
    flex: 1 1 0;
    text-align: center;
    font-family: %(mono_font)s;
    font-size: 10px;
    padding: 3px 0;
    border: 1px solid %(border_strong)s;
    color: %(muted)s;
    background: %(field)s;
}
.pycopter-mpi div.running {
    background: %(accent_tint)s;
    border-color: %(accent)s;
    color: %(accent_hi)s;
}
.pycopter-mpi div.ok {
    border-color: %(good)s;
    color: %(good)s;
}
.pycopter-mpi div.failed {
    border-color: %(danger)s;
    color: %(danger)s;
}
.pycopter-mpi-meta {
    font-size: 10px;
    color: %(dim)s;
    margin-bottom: 6px;
}

/* ---- metric strip ---- */
.pycopter-metrics {
    flex: 0 0 auto;
    display: flex;
    height: %(metric_height)dpx;
    border-bottom: 1px solid %(border)s;
    background: %(panel)s;
    overflow-x: auto;
}
.pycopter-metrics .cell {
    flex: 1 1 0;
    min-width: 96px;
    padding: 5px 10px;
    border-right: 1px solid %(border)s;
    white-space: nowrap;
}
.pycopter-metrics .cell:last-child { border-right: 0; }
.pycopter-metrics .label {
    display: block;
    font-family: %(heading_font)s;
    font-size: 9px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: %(muted)s;
}
.pycopter-metrics .value {
    font-size: 17px;
    font-weight: 600;
    color: %(text)s;
}
.pycopter-metrics .unit {
    font-size: 10px;
    color: %(dim)s;
    margin-left: 3px;
}
.pycopter-metrics .delta { font-size: 10px; margin-left: 6px; }
.pycopter-metrics .delta.good { color: %(good)s; }
.pycopter-metrics .delta.bad { color: %(danger)s; }
.pycopter-metrics .delta.flat { color: %(dim)s; }

/* ---- canvas ---- */
.pycopter-plot-frame {
    background: %(plot)s;
    border: 1px solid %(border)s;
    min-height: 0;
    overflow: hidden;
}
.pycopter-canvas-controls {
    flex: 0 0 auto;
    align-items: center;
    gap: 6px;
    padding: 4px 6px;
    border-bottom: 1px solid %(border)s;
    background: %(panel)s;
}
.pycopter-table-scroll {
    overflow-x: auto;
    overflow-y: hidden;
    min-height: 0;
}
.pycopter-table-foot {
    flex: 0 0 auto;
    font-size: 11px;
    color: %(muted)s;
    border-top: 1px solid %(border)s;
    padding: 4px 8px;
    background: %(panel)s;
}
.pycopter-table-foot .warn { color: %(warning)s; }
.pycopter-table-foot .bad { color: %(danger)s; }

/* ---- run rail ---- */
.pycopter-runrail {
    width: %(runrail_width)dpx;
    flex: 0 0 auto;
    border-left: 1px solid %(border)s;
}
.pycopter-runrail-head {
    display: flex;
    justify-content: space-between;
    background: %(panel_alt)s;
    border-bottom: 1px solid %(border)s;
    padding: 6px 10px;
    font-family: %(heading_font)s;
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: %(text)s;
}
.pycopter-runrail-head .meta {
    font-family: %(body_font)s;
    font-size: 10px;
    letter-spacing: 0;
    text-transform: none;
    color: %(accent_hi)s;
}
.pycopter-runrail-list {
    height: 100%%;
    overflow-y: auto;
    overflow-x: hidden;
    font-size: 11px;
}
.pycopter-runrail-cols {
    display: flex;
    padding: 3px 8px;
    border-bottom: 1px solid %(border)s;
    font-family: %(heading_font)s;
    font-size: 9px;
    letter-spacing: .12em;
    color: %(dim)s;
    text-transform: uppercase;
}
.pycopter-run {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    padding: 5px 8px;
    border-bottom: 1px solid %(border)s;
    cursor: pointer;
}
.pycopter-run:hover { background: %(panel_alt)s; }
.pycopter-run.current { background: %(accent_tint)s; }
.pycopter-run.child { padding-left: 22px; }
.pycopter-run.queued .run-name,
.pycopter-run.queued .run-meta { color: %(dim)s; }
.pycopter-run.failed { background: rgba(180, 87, 77, .12); }
.pycopter-run input[type="checkbox"] {
    margin-top: 3px;
    accent-color: %(accent)s;
}
.pycopter-run .run-main { flex: 1 1 auto; min-width: 0; }
.pycopter-run .run-name {
    color: %(text)s;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pycopter-run .run-name .state {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 6px;
    background: %(dim)s;
}
.pycopter-run .run-name .state.done { background: %(good)s; }
.pycopter-run .run-name .state.running { background: %(accent_hi)s; }
.pycopter-run .run-name .state.failed { background: %(danger)s; }
.pycopter-run .run-name .state.baseline { background: %(accent_hi)s; box-shadow: 0 0 0 2px %(accent_tint)s; }
.pycopter-run .run-name .state.cancelled { background: %(warning)s; }
.pycopter-run .run-meta {
    color: %(muted)s;
    font-size: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pycopter-run .run-value {
    width: 56px;
    text-align: right;
    font-family: %(mono_font)s;
    font-size: 11px;
    color: %(text)s;
    flex: 0 0 auto;
}
.pycopter-run .run-delta {
    width: 62px;
    text-align: right;
    font-family: %(mono_font)s;
    font-size: 11px;
    flex: 0 0 auto;
}
.pycopter-run .run-delta.good { color: %(good)s; }
.pycopter-run .run-delta.bad { color: %(danger)s; }
.pycopter-run .run-delta.flat { color: %(dim)s; }
.pycopter-run-progress {
    height: 4px;
    background: %(field)s;
    border: 1px solid %(border)s;
    margin-top: 3px;
}
.pycopter-run-progress span {
    display: block;
    height: 100%%;
    background: %(accent)s;
}
.pycopter-run-error {
    font-family: %(mono_font)s;
    font-size: 10px;
    color: %(danger)s;
    background: rgba(180, 87, 77, .1);
    border-left: 2px solid %(danger)s;
    padding: 5px 8px;
    margin: 0 8px 6px 26px;
    white-space: pre-wrap;
    word-break: break-word;
}
.pycopter-run-toggle {
    color: %(muted)s;
    width: 12px;
    flex: 0 0 auto;
    text-align: center;
    user-select: none;
}
.pycopter-runrail-foot {
    flex: 0 0 auto;
    border-top: 1px solid %(border)s;
    background: %(panel)s;
    padding: 6px;
    gap: 6px;
}

/* ---- log ---- */
.pycopter-log-region {
    flex: 0 0 auto;
    background: %(panel)s;
    border-top: 1px solid %(border)s;
    overflow: hidden;
}
.pycopter-log-grip {
    flex: 0 0 auto;
    height: %(grip_height)dpx;
    cursor: ns-resize;
    background: %(panel)s;
    border-top: 1px solid %(border)s;
    display: flex;
    align-items: center;
    justify-content: center;
    touch-action: none;
    user-select: none;
}
.pycopter-log-grip::after {
    content: "";
    width: 54px;
    height: 3px;
    background: %(border_strong)s;
}
.pycopter-log-grip:hover::after,
.pycopter-log-grip.pycopter-grip-active::after {
    background: %(accent)s;
}
.pycopter-log-strip {
    height: %(log_strip)dpx;
    display: flex;
    align-items: center;
    gap: 0;
    flex: 0 0 auto;
}
.pycopter-log-line {
    font-family: %(mono_font)s;
    font-size: 11px;
    color: %(muted)s;
    padding: 0 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pycopter-log-line .tag {
    font-family: %(heading_font)s;
    letter-spacing: .14em;
    color: %(dim)s;
    margin-right: 8px;
}
.pycopter-log-counts {
    font-size: 11px;
    text-align: right;
    padding: 0 8px;
    white-space: nowrap;
}
.pycopter-log-counts .warn { color: %(warning)s; }
.pycopter-log-counts .bad { color: %(danger)s; }
.pycopter-log-counts .hint { color: %(dim)s; }
.pycopter-log-counts-col {
    font-family: %(mono_font)s;
    font-size: 10px;
    text-align: right;
    padding-right: 8px;
}
.pycopter-log-counts-col .pycopter-log-count {
    height: 20px;
    line-height: 20px;
    color: %(dim)s;
}
.pycopter-log-counts-col .warn { color: %(warning)s; }
.pycopter-log-counts-col .error { color: %(danger)s; }
.pycopter-log-body {
    min-height: 0;
    gap: 0;
}
/* ---- shared widget chrome ---- */
button.bk-btn,
.bk-btn {
    border: 1px solid %(border_strong)s !important;
    border-radius: 0 !important;
    background: %(panel_alt)s !important;
    color: %(text)s !important;
    min-height: 24px;
    font-size: 11px !important;
}
button.bk-btn:hover,
.bk-btn:hover {
    background: %(field_hover)s !important;
    border-color: %(accent)s !important;
}
button.bk-btn:disabled,
.bk-btn:disabled,
.bk-btn.bk-disabled {
    color: %(dim)s !important;
    background: %(panel)s !important;
    border-color: %(border)s !important;
}
.bk-root,
.bk { color: %(text)s; }
.bk-input-group label,
.bk-input-group .bk-input-group-label,
.bk-checkbox-label,
.bk-radio-group label { color: %(muted)s !important; }
.bk-input,
input.bk-input,
select.bk-input,
textarea.bk-input {
    background: %(field)s !important;
    border-color: %(border)s !important;
    border-radius: 0 !important;
    color: %(text)s !important;
}
.bk-input:hover,
input.bk-input:hover,
select.bk-input:hover,
textarea.bk-input:hover {
    background: %(field_hover)s !important;
    border-color: %(border_strong)s !important;
}
.bk-input:focus,
input.bk-input:focus,
select.bk-input:focus,
textarea.bk-input:focus {
    border-color: %(accent)s !important;
    box-shadow: 0 0 0 1px %(accent)s !important;
}
.bk-divider { border-color: %(border)s !important; }
.bk-tab {
    background: %(panel)s !important;
    border-color: %(border)s !important;
    border-radius: 0 !important;
    color: %(muted)s !important;
    font-family: %(heading_font)s;
    font-size: 11px !important;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.bk-tab.bk-active {
    background: %(panel_alt)s !important;
    color: %(text)s !important;
    border-top-color: %(accent)s !important;
}
input[type="file"]::file-selector-button {
    background: %(panel_alt)s !important;
    border: 1px solid %(border_strong)s !important;
    color: %(text)s !important;
    border-radius: 0 !important;
    padding: 3px 8px !important;
}
.tabulator {
    background: %(field)s !important;
    border-color: %(border)s !important;
    color: %(text)s !important;
    font-size: 11px;
}
.tabulator .tabulator-header,
.tabulator .tabulator-header .tabulator-col {
    background: %(panel_alt)s !important;
    border-color: %(border)s !important;
    color: %(text)s !important;
}
.tabulator .tabulator-row {
    background: %(field)s !important;
    border-color: %(border)s !important;
    color: %(text)s !important;
}
.tabulator .tabulator-row.tabulator-row-even { background: %(panel)s !important; }
.tabulator .tabulator-row:hover { background: %(field_hover)s !important; }
.tabulator .tabulator-cell { border-color: %(border)s !important; }
.tabulator .tabulator-footer {
    background: %(panel_alt)s !important;
    border-color: %(border)s !important;
    color: %(muted)s !important;
}
"""


RAW_CSS = (
    _BASE_CSS
    % {  # noqa: E501 - one substitution table for the whole sheet
        **THEME,
        "body_font": BODY_FONT,
        "heading_font": HEADING_FONT,
        "mono_font": MONO_FONT,
        "command_bar": COMMAND_BAR_HEIGHT,
        "warning_bar": WARNING_BAR_HEIGHT,
        "rail_width": ICON_RAIL_WIDTH,
        "inspector_width": INSPECTOR_WIDTH,
        "runrail_width": RUN_RAIL_WIDTH,
        "result_min_width": RESULT_MIN_WIDTH,
        "metric_height": METRIC_STRIP_HEIGHT,
        "grip_height": LOG_GRIP_HEIGHT,
        "log_strip": LOG_STRIP_HEIGHT,
    }
)


class ResultAreaProbe(pn.reactive.ReactiveHTML):
    """Reports the on-screen size of the plot frame back to Python.

    Panel cannot tell the server how large a pane ended up in the browser, so
    matplotlib figures would otherwise be drawn at a guessed size and then
    rescaled by the browser, which shrinks axis labels on small monitors. This
    zero-height probe measures the plot frame and syncs its pixel size so
    figures can be drawn at their true display size.
    """

    width_px = param.Integer(default=0)
    height_px = param.Integer(default=0)

    _template = '<div id="probe" style="display:none"></div>'
    _scripts = {
        "render": """
            const SELECTOR = '.pycopter-plot-frame';
            // Panel renders each component into its own shadow root, so the
            // plot frame is invisible to a plain document.querySelector.
            function findFrame(root) {
                for (const el of root.querySelectorAll('*')) {
                    if (el.matches && el.matches(SELECTOR)) { return el; }
                    if (el.shadowRoot) {
                        const found = findFrame(el.shadowRoot);
                        if (found) { return found; }
                    }
                }
                return null;
            }
            function report() {
                const frame = findFrame(document);
                if (!frame) { return false; }
                const box = frame.getBoundingClientRect();
                if (box.width < 80 || box.height < 80) { return false; }
                const width = Math.round(box.width);
                const height = Math.round(box.height);
                if (width !== data.width_px || height !== data.height_px) {
                    data.width_px = width;
                    data.height_px = height;
                }
                return true;
            }
            function observe(attempt) {
                const frame = findFrame(document);
                if (frame && window.ResizeObserver) {
                    new ResizeObserver(report).observe(frame);
                    report();
                    return;
                }
                if (attempt < 40) { setTimeout(() => observe(attempt + 1), 150); }
            }
            window.addEventListener('resize', report);
            observe(0);
        """
    }


class LogSplitter(pn.reactive.ReactiveHTML):
    """Drag handle that resizes the output log against the result area.

    Dragging updates the log element's inline height directly so the flex
    layout reflows every frame, and commits the final height to Python on
    release. The commit matters because the terminal only re-flows its rows
    when Panel resizes the widget itself.
    """

    height_px = param.Integer(default=LOG_MIN_HEIGHT)

    _template = (
        '<div id="grip" class="pycopter-log-grip"'
        ' title="Drag to resize the output log. Double-click to reset."></div>'
    )
    _scripts = {
        "render": """
            const MIN_HEIGHT = %(min_height)d;
            const RESULT_MIN = %(result_min)d;
            const GRIP_HEIGHT = %(grip_height)d;
            const STRIP_HEIGHT = %(strip_height)d;
            const TERMINAL_INSET = %(terminal_inset)d;

            // Panel renders each component into its own shadow root, so these
            // elements cannot be reached with a plain document.querySelector.
            function findDeep(selector, root) {
                for (const el of (root || document).querySelectorAll('*')) {
                    if (el.matches && el.matches(selector)) { return el; }
                    if (el.shadowRoot) {
                        const found = findDeep(selector, el.shadowRoot);
                        if (found) { return found; }
                    }
                }
                return null;
            }

            // xterm only re-flows its rows when the Panel view refits it, and
            // Bokeh does not run a layout pass for a plain height change.
            let terminalView = null;
            function findTerminalView(view, depth) {
                if (!view || depth > 12) { return null; }
                if (view.constructor && view.constructor.__name__ === 'TerminalView') { return view; }
                for (const child of (view.child_views || [])) {
                    const found = findTerminalView(child, depth + 1);
                    if (found) { return found; }
                }
                return null;
            }
            function refitTerminal() {
                try {
                    if (!terminalView) {
                        const roots = (window.Bokeh && window.Bokeh.index) || {};
                        for (const root of Object.values(roots)) {
                            terminalView = findTerminalView(root, 0);
                            if (terminalView) { break; }
                        }
                    }
                    if (terminalView && typeof terminalView.fit === 'function') {
                        terminalView.fit();
                    }
                } catch (err) {
                    // Leave the log at its current row count rather than break the drag.
                }
            }

            function applyHeight(height) {
                const region = findDeep('.pycopter-log-region');
                if (region) { region.style.height = height + 'px'; }
                // Resize the terminal box too, so the rows follow the drag
                // instead of snapping only once Python commits the height.
                const inner = Math.max(20, height - STRIP_HEIGHT - TERMINAL_INSET);
                const body = findDeep('.pycopter-log-body');
                if (body) { body.style.height = inner + 'px'; }
                const host = findDeep('.pycopter-log');
                if (host) { host.style.height = inner + 'px'; }
                const container = findDeep('.terminal-container');
                if (container) { container.style.height = inner + 'px'; }
            }

            function maxHeight() {
                return Math.max(MIN_HEIGHT, window.innerHeight - RESULT_MIN - GRIP_HEIGHT);
            }
            function clamp(value) {
                return Math.round(Math.min(Math.max(value, MIN_HEIGHT), maxHeight()));
            }

            let dragging = false;
            let startY = 0;
            let startHeight = MIN_HEIGHT;
            let liveHeight = MIN_HEIGHT;
            let refitQueued = false;
            function queueRefit() {
                if (refitQueued) { return; }
                refitQueued = true;
                window.requestAnimationFrame(() => {
                    refitQueued = false;
                    refitTerminal();
                });
            }

            grip.addEventListener('pointerdown', (event) => {
                const region = findDeep('.pycopter-log-region');
                if (!region) { return; }
                dragging = true;
                startY = event.clientY;
                startHeight = region.getBoundingClientRect().height;
                liveHeight = startHeight;
                grip.classList.add('pycopter-grip-active');
                grip.setPointerCapture(event.pointerId);
                event.preventDefault();
            });

            grip.addEventListener('pointermove', (event) => {
                if (!dragging) { return; }
                // Dragging up (a smaller clientY) makes the log taller.
                liveHeight = clamp(startHeight + (startY - event.clientY));
                applyHeight(liveHeight);
                queueRefit();
                event.preventDefault();
            });

            function endDrag(event) {
                if (!dragging) { return; }
                dragging = false;
                grip.classList.remove('pycopter-grip-active');
                if (event && event.pointerId !== undefined && grip.hasPointerCapture(event.pointerId)) {
                    grip.releasePointerCapture(event.pointerId);
                }
                if (liveHeight !== data.height_px) { data.height_px = liveHeight; }
                refitTerminal();
            }
            grip.addEventListener('pointerup', endDrag);
            grip.addEventListener('pointercancel', endDrag);

            grip.addEventListener('dblclick', () => {
                liveHeight = MIN_HEIGHT;
                applyHeight(MIN_HEIGHT);
                if (data.height_px !== MIN_HEIGHT) { data.height_px = MIN_HEIGHT; }
                refitTerminal();
            });

            // Shrinking the window must not leave the log taller than allowed.
            window.addEventListener('resize', () => {
                const region = findDeep('.pycopter-log-region');
                if (!region || dragging) { return; }
                const corrected = clamp(region.getBoundingClientRect().height);
                applyHeight(corrected);
                if (corrected !== data.height_px) { data.height_px = corrected; }
                queueRefit();
            });
        """
        % {
            "min_height": LOG_MIN_HEIGHT,
            "result_min": RESULT_MIN_HEIGHT,
            "grip_height": LOG_GRIP_HEIGHT,
            "strip_height": LOG_STRIP_HEIGHT,
            "terminal_inset": LOG_TERMINAL_INSET,
        }
    }


class RunRail(pn.reactive.ReactiveHTML):
    """Scrollable list of session runs with per-row state.

    Rendered by hand rather than as a table because a row is not uniform: a
    running sweep carries a progress bar and an ETA, a failed run carries its
    exception text inline, and sweep points are indented under their parent.
    Row interactions are reported back through a single ``event`` string so a
    repeated click on the same run still reaches Python.
    """

    rows = param.List(default=[])
    event = param.String(default="")

    _template = '<div id="rail" class="pycopter-runrail-list"></div>'
    _scripts = {
        "render": """
            state.counter = 0;
            state.send = (kind, runId) => {
                state.counter += 1;
                data.event = kind + '|' + runId + '|' + state.counter;
            };
            state.escape = (text) => String(text === undefined || text === null ? '' : text)
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            state.draw = () => {
                const parts = [];
                for (const row of (data.rows || [])) {
                    const classes = ['pycopter-run', row.status || 'queued'];
                    if (row.current) { classes.push('current'); }
                    if (row.child) { classes.push('child'); }
                    const checked = row.checked ? ' checked' : '';
                    const toggle = row.has_children
                        ? '<span class="pycopter-run-toggle" data-toggle="' + row.run_id + '">'
                          + (row.collapsed ? '\\u25b8' : '\\u25be') + '</span>'
                        : '<span class="pycopter-run-toggle"></span>';
                    let extra = '';
                    if (row.progress_pct !== null && row.progress_pct !== undefined) {
                        extra = '<div class="pycopter-run-progress"><span style="width:'
                            + row.progress_pct + '%"></span></div>';
                    }
                    parts.push(
                        '<div class="' + classes.join(' ') + '" data-run="' + row.run_id + '">'
                        + '<input type="checkbox" data-check="' + row.run_id + '"' + checked + '>'
                        + toggle
                        + '<div class="run-main">'
                        + '<div class="run-name"><span class="state ' + (row.dot || '')
                        + '"></span>' + state.escape(row.run_id) + ' ' + state.escape(row.label)
                        + '</div>'
                        + '<div class="run-meta">' + state.escape(row.meta) + '</div>'
                        + extra
                        + '</div>'
                        + '<div class="run-value">' + state.escape(row.power) + '</div>'
                        + '<div class="run-value">' + state.escape(row.fom) + '</div>'
                        + '<div class="run-delta ' + (row.delta_class || 'flat') + '">'
                        + state.escape(row.delta) + '</div>'
                        + '</div>'
                    );
                    if (row.error) {
                        parts.push('<div class="pycopter-run-error">'
                            + state.escape(row.error) + '</div>');
                    }
                }
                rail.innerHTML = parts.join('');
            };
            rail.addEventListener('click', (ev) => {
                const check = ev.target.closest('[data-check]');
                if (check) {
                    state.send(check.checked ? 'check' : 'uncheck',
                               check.getAttribute('data-check'));
                    ev.stopPropagation();
                    return;
                }
                const toggle = ev.target.closest('[data-toggle]');
                if (toggle) {
                    state.send('toggle', toggle.getAttribute('data-toggle'));
                    ev.stopPropagation();
                    return;
                }
                const row = ev.target.closest('[data-run]');
                if (row) { state.send('select', row.getAttribute('data-run')); }
            });
            state.draw();
        """,
        "rows": "state.draw();",
    }


class KeyBindings(pn.reactive.ReactiveHTML):
    """Global keyboard shortcuts for the app menu commands.

    Save actions are browser downloads, so the shortcut clicks the hidden
    download anchor directly instead of round-tripping through Python, which
    cannot start a download on its own.
    """

    command = param.String(default="")

    _template = '<div id="keys" style="display:none"></div>'
    _scripts = {
        "render": """
            state.counter = 0;
            function findDeep(selector) {
                function walk(root) {
                    for (const el of root.querySelectorAll('*')) {
                        if (el.matches && el.matches(selector)) { return el; }
                        if (el.shadowRoot) {
                            const found = walk(el.shadowRoot);
                            if (found) { return found; }
                        }
                    }
                    return null;
                }
                return walk(document);
            }
            window.addEventListener('keydown', (event) => {
                if (!event.ctrlKey && !event.metaKey) { return; }
                const key = event.key.toLowerCase();
                if (!['n', 'o', 's'].includes(key)) { return; }
                event.preventDefault();
                if (key === 's') {
                    const selector = event.shiftKey
                        ? '.pycopter-save-config-as a' : '.pycopter-save-config a';
                    const anchor = findDeep(selector) || findDeep('.pycopter-save-config a');
                    if (anchor) { anchor.click(); }
                    return;
                }
                state.counter += 1;
                data.command = (key === 'n' ? 'new' : 'open') + '|' + state.counter;
            });
        """
    }

class PycopterWebApp:
    """Stateful Panel application."""

    def __init__(self) -> None:
        pn.extension("tabulator", "terminal", raw_css=[RAW_CSS])
        self.runs = RunStore(max_runs=int(DEFAULT_CONFIG["max_session_runs"]))
        self.current_case: HoverCase | None = None
        self.active_run_id: str | None = None
        self.showing = "total"
        self.checked_run_ids: list[str] = []
        self.collapsed_parents: set[str] = set()
        self.inspector_tab = "ROTOR"
        self.recent_configs: list[tuple[str, dict[str, Any]]] = []
        self.config_filename = "untitled.json"

        self._xfoil_provider = None
        self._xfoil_provider_key: tuple[Any, ...] | None = None
        self._xfoil_state = {"workers": 0, "state": "idle", "jobs": 0, "error": ""}
        self.output_lines: list[str] = []
        self._log_filter = "ALL"
        self._log_counts = {"WARN": 0, "ERROR": 0, "SOLVER": 0}
        self._plot_area_size_px: tuple[int, int] | None = None
        self._current_plot_name = ""
        self.plot_fig = self._blank_figure("Solve a run to see results.")
        self._plot_save_available = False

        self._field_wrappers: dict[str, pn.Column] = {}
        self._field_stylesheets: dict[str, tuple[list[str], list[str]]] = {}
        self._bindings: dict[str, Any] = {}
        self._edited_count = 0
        self._applying = False
        self._baseline_config = normalize_config(DEFAULT_CONFIG)

        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._active_job: dict[str, Any] | None = None
        self._periodic = None
        self._background_enabled = False

        self._build_widgets()
        self._wire_events()
        self._update_plot_options()
        self._sync_enabled_state()
        self._refresh_all()

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    def view(self):
        body = pn.Row(
            self.icon_rail,
            self.inspector,
            self.canvas,
            self.run_rail_region,
            sizing_mode="stretch_both",
            css_classes=["pycopter-body"],
            margin=0,
        )
        shell = pn.Column(
            self.command_bar,
            self.warning_bar,
            body,
            self.log_splitter,
            self.log_region,
            css_classes=["pycopter-shell"],
            sizing_mode="stretch_both",
            margin=0,
        )
        self._start_background_polling()
        return shell

    def _start_background_polling(self) -> None:
        """Drain calculation-thread progress on the Panel event loop.

        Without a running server there is no loop to attach to, so the app
        falls back to solving on the calling thread. That is what the unit
        tests exercise.
        """
        if self._periodic is not None:
            return
        try:
            self._periodic = pn.state.add_periodic_callback(
                self._drain_worker_queue,
                period=WORKER_POLL_MS,
            )
            self._background_enabled = True
        except Exception:
            self._periodic = None
            self._background_enabled = False

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        cfg = normalize_config(DEFAULT_CONFIG)
        self._build_inspector_widgets(cfg)
        self._build_menu(cfg)
        self._build_command_bar()
        self._build_canvas()
        self._build_run_rail()
        self._build_log()
        self._build_icon_rail()
        self._build_inspector()

    # ---- shared field helpers ----------------------------------------

    def _field(self, widget, unit: str = "", key: str | None = None, derived: bool = False):
        # Field chrome lives in the widget's own stylesheet: label, input size
        # and the unit chip are all inside the widget's shadow root, which a
        # rule written against the wrapper could never reach.
        sheets = [FIELD_CSS]
        if unit:
            sheets.append(_unit_stylesheet(unit))
        if derived:
            sheets.append(DERIVED_FIELD_CSS)
        widget.stylesheets = list(sheets)

        classes = ["pycopter-field"]
        if unit:
            classes.append(f"pycopter-unit-{unit}")
        if derived:
            classes.append("pycopter-derived")
        wrapper = pn.Column(
            widget,
            css_classes=list(classes),
            sizing_mode="stretch_width",
            margin=(0, 0, 5, 0),
        )
        if key is not None:
            self._bindings[key] = widget
            self._field_wrappers[key] = wrapper
            self._field_stylesheets[key] = (list(sheets), list(sheets) + [EDITED_FIELD_CSS])
            wrapper._pycopter_classes = classes
        return wrapper

    def _pair(self, left, right=None):
        """Two-column field row; a missing right cell keeps the grid aligned."""
        return pn.Row(
            left,
            right if right is not None else pn.Spacer(sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            margin=0,
        )

    def _subhead(self, text: str) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div class="pycopter-subhead">{text}</div>',
            sizing_mode="stretch_width",
            margin=0,
        )

    def _text_display(self, label: str) -> pn.widgets.TextInput:
        return pn.widgets.TextInput(
            label=label, value="", disabled=True, sizing_mode="stretch_width"
        )

    # ---- inspector widgets -------------------------------------------

    def _build_inspector_widgets(self, cfg: dict[str, Any]) -> None:
        # ROTOR ---------------------------------------------------------
        self.rotor_system_type = pn.widgets.Select(
            label="System",
            options={"Single rotor": "single", "Coaxial": "coaxial"},
            value=cfg["rotor_system_type"],
            sizing_mode="stretch_width",
        )
        self.num_blades = pn.widgets.IntInput(
            label="Blades / rotor", value=cfg["num_blades"], start=1, end=12, step=1,
            sizing_mode="stretch_width",
        )
        self.rotor_diam = pn.widgets.FloatInput(
            label="Diameter", value=cfg["rotor_diam"], start=0.05, end=30.0, step=0.01,
            sizing_mode="stretch_width",
        )
        self.root_cutout = pn.widgets.FloatInput(
            label="Root cutout", value=cfg["root_cutout"], start=0.0, end=0.95, step=0.01,
            sizing_mode="stretch_width",
        )
        self.headspeed_rpm = pn.widgets.FloatInput(
            label="Upper RPM", value=cfg["headspeed_rpm"], start=100.0, end=100000.0,
            step=50.0, sizing_mode="stretch_width",
        )
        self.lower_rotor_speed_ratio = pn.widgets.FloatInput(
            label="RPM ratio", value=cfg["lower_rotor_speed_ratio"], start=0.25, end=3.0,
            step=0.01, sizing_mode="stretch_width",
        )
        self.coaxial_spacing = pn.widgets.FloatInput(
            label="Spacing", value=cfg["coaxial_spacing_ratio"], start=0.05, end=1.5,
            step=0.05, sizing_mode="stretch_width",
        )
        self.lower_collective_offset = pn.widgets.FloatInput(
            label="Coll. bias", value=cfg["lower_collective_offset_deg"], start=-10.0,
            end=10.0, step=0.25, sizing_mode="stretch_width",
        )
        self.lower_rotor_scale = pn.widgets.FloatInput(
            label="Lower scale", value=cfg["lower_rotor_scale"], start=0.5, end=1.5,
            step=0.01, sizing_mode="stretch_width",
        )
        self.lower_headspeed_rpm = pn.widgets.FloatInput(
            label="Lower RPM",
            value=float(cfg["headspeed_rpm"]) * float(cfg["lower_rotor_speed_ratio"]),
            start=0.0, end=300000.0, step=1.0, disabled=True,
            sizing_mode="stretch_width",
        )
        self.tip_speed_display = self._text_display("Tip speed")
        self.tip_mach_display = self._text_display("Tip Mach")
        self.disk_area_display = self._text_display("Disk area")

        # BLADE ---------------------------------------------------------
        self.airfoil = pn.widgets.TextInput(
            label="Airfoil", value=cfg["airfoil"], sizing_mode="stretch_width"
        )
        self.geometry_mode = pn.widgets.Select(
            label="Geometry source",
            options={"Uniform": "uniform", "Station table": "station_table"},
            value=cfg["geometry_mode"],
            sizing_mode="stretch_width",
        )
        self.chord = pn.widgets.FloatInput(
            label="Chord", value=cfg["chord"], start=0.001, end=2.5, step=0.001,
            sizing_mode="stretch_width",
        )
        self.root_twist = pn.widgets.FloatInput(
            label="Root twist", value=cfg["root_twist_deg"], start=-30.0, end=30.0,
            step=0.25, sizing_mode="stretch_width",
        )
        self.tip_twist = pn.widgets.FloatInput(
            label="Tip twist", value=cfg["tip_twist_deg"], start=-30.0, end=30.0,
            step=0.25, sizing_mode="stretch_width",
        )
        self.station_pitch_axis = pn.widgets.FloatInput(
            label="Moment axis", value=cfg["station_pitch_axis_frac"], start=0.0, end=1.0,
            step=0.01, sizing_mode="stretch_width",
        )
        self.station_count = pn.widgets.IntInput(
            label="Station count", value=cfg["station_count"], start=2, end=40, step=1,
            sizing_mode="stretch_width",
        )
        self.solidity_display = self._text_display("Solidity")
        self.blade_area_display = self._text_display("Blade area")
        self.aspect_ratio_display = self._text_display("Aspect ratio")
        self.station_table = pn.widgets.Tabulator(
            pd.DataFrame(DEFAULT_STATION_ROWS),
            show_index=False,
            height=190,
            sizing_mode="stretch_width",
            layout="fit_columns",
            titles={
                "r_over_R": "r/R",
                "chord_m": "Chord [m]",
                "twist_deg": "Twist [deg]",
                "airfoil": "Airfoil",
                "pitch_axis_frac": "Axis [c]",
            },
            editors={
                "r_over_R": {"type": "number", "min": 0.02, "max": 1.0, "step": 0.01},
                "chord_m": {"type": "number", "min": 0.001, "max": 2.5, "step": 0.001},
                "twist_deg": {"type": "number", "min": -30.0, "max": 30.0, "step": 0.25},
                "airfoil": "input",
                "pitch_axis_frac": {"type": "number", "min": 0.0, "max": 1.0, "step": 0.01},
            },
        )
        self.reset_stations_btn = pn.widgets.Button(
            label="Reset from root/tip twist",
            sizing_mode="stretch_width",
            stylesheets=[CONDENSED_BUTTON_CSS],
        )

        # OPER ----------------------------------------------------------
        self.gross = pn.widgets.FloatInput(
            label="Gross mass", value=cfg["gross"], start=0.01, end=100000.0, step=0.1,
            sizing_mode="stretch_width",
        )
        self.density = pn.widgets.FloatInput(
            label="Density", value=cfg["density"], start=0.01, end=2.0, step=0.001,
            sizing_mode="stretch_width",
        )
        self.kinematic_viscosity = pn.widgets.FloatInput(
            label="Kin. viscosity", value=cfg["kinematic_viscosity_m2_s"], start=1.0e-5,
            end=2.5e-5, step=1.0e-7, sizing_mode="stretch_width",
        )
        self.speed_of_sound = pn.widgets.FloatInput(
            label="Speed of sound", value=cfg["speed_of_sound_m_s"], start=250.0, end=400.0,
            step=0.5, sizing_mode="stretch_width",
        )
        self.altitude = pn.widgets.FloatInput(
            label="Altitude", value=cfg["altitude_m"], start=-500.0, end=11000.0, step=50.0,
            sizing_mode="stretch_width",
        )
        self.temperature = pn.widgets.FloatInput(
            label="Temperature", value=cfg["temperature_C"], start=-60.0, end=60.0, step=0.5,
            sizing_mode="stretch_width",
        )
        self.flight_mode = pn.widgets.Select(
            label="Flight mode",
            options={"Hover": "hover", "Forward flight": "forward_flight"},
            value=cfg["flight_mode"],
            sizing_mode="stretch_width",
        )
        self.apply_isa_btn = pn.widgets.Button(
            label="Apply ISA to density / viscosity",
            sizing_mode="stretch_width",
            stylesheets=[CONDENSED_BUTTON_CSS],
        )
        self.target_thrust_display = self._text_display("Target thrust")
        self.disk_loading_display = self._text_display("Disk loading")

        # SOLVER --------------------------------------------------------
        self.trim_mode = pn.widgets.Select(
            label="Trim mode",
            options={"Target thrust": "target_thrust", "Fixed collective": "fixed_collective"},
            value=cfg["trim_mode"],
            sizing_mode="stretch_width",
        )
        self.collective_pitch = pn.widgets.FloatInput(
            label="Collective", value=cfg["collective_pitch_deg"], start=-5.0, end=25.0,
            step=0.25, sizing_mode="stretch_width",
        )
        self.blade_element_count = pn.widgets.IntInput(
            label="Blade elements", value=cfg["blade_element_count"], start=20, end=200,
            step=1, sizing_mode="stretch_width",
        )
        self.element_spacing = pn.widgets.Select(
            label="Spacing law",
            options={"Uniform": "uniform", "Cosine": "cosine"},
            value=cfg["element_spacing"],
            sizing_mode="stretch_width",
        )
        self.min_collective = pn.widgets.FloatInput(
            label="Search from", value=cfg["min_collective_deg"], start=-10.0, end=10.0,
            step=0.25, sizing_mode="stretch_width",
        )
        self.max_collective = pn.widgets.FloatInput(
            label="Search to", value=cfg["max_collective_deg"], start=0.0, end=35.0,
            step=0.25, sizing_mode="stretch_width",
        )
        self.thrust_tolerance = pn.widgets.FloatInput(
            label="Thrust tol.", value=cfg["thrust_tolerance"], start=1.0e-6, end=1.0e-1,
            step=1.0e-4, format="0.000000", sizing_mode="stretch_width",
        )
        self.max_trim_iterations = pn.widgets.IntInput(
            label="Max iter", value=cfg["max_trim_iterations"], start=5, end=400, step=5,
            sizing_mode="stretch_width",
        )
        self.tip_loss_model = pn.widgets.Select(
            label="Tip loss", options={"Prandtl": "prandtl", "None": "none"},
            value=cfg["tip_loss_model"], sizing_mode="stretch_width",
        )
        self.root_loss_model = pn.widgets.Select(
            label="Root loss", options={"Prandtl": "prandtl", "None": "none"},
            value=cfg["root_loss_model"], sizing_mode="stretch_width",
        )
        self.induced_power_factor = pn.widgets.FloatInput(
            label="Induced factor", value=cfg["induced_power_factor"], start=1.0, end=1.3,
            step=0.01, sizing_mode="stretch_width",
        )
        self.coaxial_trim_mode = pn.widgets.Select(
            label="Coaxial trim",
            options={
                "Torque balance": "torque_balance",
                "Equal thrust": "equal_thrust",
                "Equal collective": "equal_collective",
            },
            value=cfg["coaxial_trim_mode"],
            sizing_mode="stretch_width",
        )

        # XFOIL ---------------------------------------------------------
        self.new_polars = pn.widgets.Select(
            label="Mode",
            options={"Cache + gen": True, "Cache only": False},
            value=bool(cfg["new_polars"]),
            sizing_mode="stretch_width",
        )
        self.xfoil_workers = pn.widgets.IntInput(
            label="Workers", value=cfg["xfoil_parallel_workers"], start=2, end=16, step=1,
            sizing_mode="stretch_width",
        )
        self.polar_alpha_min = pn.widgets.FloatInput(
            label="Alpha min", value=cfg["polar_alpha_min_deg"], start=-20.0, end=5.0,
            step=0.5, sizing_mode="stretch_width",
        )
        self.polar_alpha_max = pn.widgets.FloatInput(
            label="Alpha max", value=cfg["polar_alpha_max_deg"], start=10.0, end=18.0,
            step=0.5, sizing_mode="stretch_width",
        )
        self.polar_alpha_step = pn.widgets.FloatInput(
            label="Alpha step", value=cfg["polar_alpha_step_deg"], start=0.05, end=5.0,
            step=0.05, sizing_mode="stretch_width",
        )
        self.polar_reynolds_bin = pn.widgets.FloatInput(
            label="Re bin", value=cfg["polar_reynolds_bin"], start=1000.0, end=1000000.0,
            step=10000.0, sizing_mode="stretch_width",
        )
        self.polar_mach_bin = pn.widgets.FloatInput(
            label="Mach bin", value=cfg["polar_mach_bin"], start=0.01, end=0.5, step=0.01,
            sizing_mode="stretch_width",
        )
        self.polar_n_crit = pn.widgets.FloatInput(
            label="N crit", value=cfg["polar_n_crit"], start=1.0, end=14.0, step=0.5,
            sizing_mode="stretch_width",
        )
        self.xfoil_max_iterations = pn.widgets.IntInput(
            label="XFOIL iter", value=cfg["xfoil_max_iterations"], start=20, end=2000,
            step=20, sizing_mode="stretch_width",
        )
        self.xfoil_timeout = pn.widgets.IntInput(
            label="Timeout", value=cfg["xfoil_timeout_s"], start=5, end=600, step=5,
            sizing_mode="stretch_width",
        )
        self.xfoil_backend = pn.widgets.Select(
            label="Backend", options={"MPI": "mpi", "Serial debug": "serial"},
            value=cfg["xfoil_parallel_backend"], sizing_mode="stretch_width",
        )
        self.xfoil_cache_dir = pn.widgets.TextInput(
            label="Cache directory", value=cfg["xfoil_cache_directory"],
            placeholder="optional ignored path", sizing_mode="stretch_width",
        )
        self.mpi_strip = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)

        # PROP ----------------------------------------------------------
        self.propulsion_model = pn.widgets.Select(
            label="Model",
            options={"Electric": "electric", "Fossil fuel": "fossil"},
            value=cfg["propulsion_model"],
            sizing_mode="stretch_width",
        )
        self.transmission_loss = pn.widgets.FloatInput(
            label="Transmission loss", value=cfg["transmission_loss"], start=0.0, end=0.5,
            step=0.01, sizing_mode="stretch_width",
        )
        self.battery_capacity = pn.widgets.FloatInput(
            label="Battery", value=cfg["battery_capacity_Wh"], start=0.1, end=100000.0,
            step=10.0, sizing_mode="stretch_width",
        )
        self.battery_usable = pn.widgets.FloatInput(
            label="Usable fraction", value=cfg["battery_usable_fraction"], start=0.05,
            end=1.0, step=0.01, sizing_mode="stretch_width",
        )
        self.motor_efficiency = pn.widgets.FloatInput(
            label="Motor eff.", value=cfg["motor_efficiency"], start=0.01, end=1.0,
            step=0.01, sizing_mode="stretch_width",
        )
        self.esc_efficiency = pn.widgets.FloatInput(
            label="ESC eff.", value=cfg["esc_efficiency"], start=0.01, end=1.0, step=0.01,
            sizing_mode="stretch_width",
        )
        self.fuel_capacity = pn.widgets.FloatInput(
            label="Fuel capacity", value=cfg["fuel_capacity_kg"], start=0.0, end=999999.0,
            step=0.1, sizing_mode="stretch_width",
        )
        self.sfc = pn.widgets.FloatInput(
            label="SFC", value=cfg["specific_fuel_consumption_kg_per_kWh"], start=0.001,
            end=10.0, step=0.01, sizing_mode="stretch_width",
        )
        self.velocity = pn.widgets.FloatInput(
            label="Velocity", value=cfg["velocity_kmh"], start=0.0, end=9999.0, step=1.0,
            sizing_mode="stretch_width",
        )
        self.fpa = pn.widgets.FloatInput(
            label="Flat plate area", value=cfg["fpa"], start=0.0, end=9999.0, step=0.01,
            sizing_mode="stretch_width",
        )
        self.propulsion_display = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)

        # Preferences ---------------------------------------------------
        self.sweep_points = pn.widgets.IntInput(
            label="Sweep points", value=cfg["sweep_points"], start=3, end=200, step=1,
            sizing_mode="stretch_width",
        )
        self.plot_render_dpi = pn.widgets.IntInput(
            label="Figure render DPI", value=cfg["plot_render_dpi"], start=72, end=300,
            step=12, sizing_mode="stretch_width",
        )
        self.log_scrollback = pn.widgets.IntInput(
            label="Log scrollback", value=cfg["log_scrollback_lines"], start=100, end=20000,
            step=100, sizing_mode="stretch_width",
        )
        self.max_session_runs = pn.widgets.IntInput(
            label="Max session runs", value=cfg["max_session_runs"], start=10, end=2000,
            step=10, sizing_mode="stretch_width",
        )

        # Footer --------------------------------------------------------
        self.revert_btn = pn.widgets.Button(
            label="REVERT",
            sizing_mode="stretch_width",
            stylesheets=[CONDENSED_BUTTON_CSS],
        )
        self.solve_btn = pn.widgets.Button(
            label="SOLVE AS NEW RUN",
            sizing_mode="stretch_width",
            css_classes=["pycopter-primary"],
            stylesheets=[PRIMARY_BUTTON_CSS],
        )
        self.cancel_btn = pn.widgets.Button(
            label="CANCEL",
            sizing_mode="stretch_width",
            visible=False,
            css_classes=["pycopter-danger"],
            stylesheets=[DANGER_BUTTON_CSS],
        )

    # ---- app menu ----------------------------------------------------

    def _build_menu(self, cfg: dict[str, Any]) -> None:
        self.new_btn = pn.widgets.Button(
            label="New session            Ctrl N",
            sizing_mode="stretch_width",
            stylesheets=[MENU_ITEM_CSS],
            margin=0,
        )
        self.load_file = pn.widgets.FileInput(
            accept=".json", sizing_mode="stretch_width", margin=(2, 10)
        )
        self.load_btn = pn.widgets.Button(
            label="Open selected config    Ctrl O",
            sizing_mode="stretch_width",
            stylesheets=[MENU_ITEM_CSS],
            margin=0,
        )
        self.save_download = pn.widgets.FileDownload(
            callback=self._save_config_callback,
            filename="pycopter_config.json",
            label="Save config             Ctrl S",
            sizing_mode="stretch_width",
            css_classes=["pycopter-save-config"],
            stylesheets=[MENU_ITEM_CSS],
        )
        self.save_as_name = pn.widgets.TextInput(
            value="pycopter_config.json",
            placeholder="save as filename",
            sizing_mode="stretch_width",
            margin=(2, 10),
        )
        self.save_as_download = pn.widgets.FileDownload(
            callback=self._save_config_callback,
            filename="pycopter_config.json",
            label="Save config as...      Ctrl ⇧ S",
            sizing_mode="stretch_width",
            css_classes=["pycopter-save-config-as"],
            stylesheets=[MENU_ITEM_CSS],
        )
        self.recent_menu = pn.Column(sizing_mode="stretch_width", margin=0)
        self.save_plot_download = pn.widgets.FileDownload(
            callback=self._save_plot_callback,
            filename="pycopter_plot.png",
            label="Export plot PNG",
            sizing_mode="stretch_width",
            stylesheets=[MENU_ITEM_CSS],
            margin=0,
        )
        self.save_summary_download = pn.widgets.FileDownload(
            callback=self._save_summary_callback,
            filename="pycopter_summary.csv",
            label="Export summary CSV",
            sizing_mode="stretch_width",
            stylesheets=[MENU_ITEM_CSS],
            margin=0,
        )
        self.save_loads_download = pn.widgets.FileDownload(
            callback=self._save_loads_callback,
            filename="pycopter_blade_loads.csv",
            label="Export loads CSV",
            sizing_mode="stretch_width",
            stylesheets=[MENU_ITEM_CSS],
            margin=0,
        )
        self.save_runs_download = pn.widgets.FileDownload(
            callback=self._save_runs_callback,
            filename="pycopter_session_runs.json",
            label="Export session runs",
            sizing_mode="stretch_width",
            stylesheets=[MENU_ITEM_CSS],
            margin=0,
        )
        self.preferences_btn = pn.widgets.Button(
            label="Preferences...",
            sizing_mode="stretch_width",
            stylesheets=[MENU_ITEM_CSS],
            margin=0,
        )
        self.preferences_panel = pn.Column(
            self._subhead("Preferences"),
            self._pair(
                self._field(self.sweep_points, "count", "sweep_points"),
                self._field(self.plot_render_dpi, "dpi", "plot_render_dpi"),
            ),
            self._pair(
                self._field(self.log_scrollback, "count", "log_scrollback_lines"),
                self._field(self.max_session_runs, "count", "max_session_runs"),
            ),
            pn.pane.HTML(
                '<div class="pycopter-menu-note">Session settings. They change how much'
                " work the UI asks for, never the solver equations.</div>",
                sizing_mode="stretch_width",
            ),
            visible=False,
            sizing_mode="stretch_width",
            margin=(0, 8),
        )

        separator = lambda: pn.pane.HTML(  # noqa: E731 - one-line layout helper
            '<div class="pycopter-menu-sep"></div>', sizing_mode="stretch_width", margin=0
        )
        self.app_menu = pn.Column(
            self.new_btn,
            pn.pane.HTML(
                '<div class="pycopter-menu-note">Open config...        Ctrl O</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            self.load_file,
            self.load_btn,
            self.save_download,
            self.save_as_name,
            self.save_as_download,
            separator(),
            pn.pane.HTML(
                '<div class="pycopter-menu-note">Recent configs</div>',
                sizing_mode="stretch_width", margin=0,
            ),
            self.recent_menu,
            separator(),
            self.save_plot_download,
            self.save_summary_download,
            self.save_loads_download,
            self.save_runs_download,
            separator(),
            self.preferences_btn,
            self.preferences_panel,
            css_classes=["pycopter-menu"],
            visible=False,
            width=MENU_WIDTH,
            # Bokeh sizes a layout container from its own model, not from the
            # content, so an auto-height popup collapses to one row. The size
            # is therefore explicit and grows when Preferences opens.
            height=MENU_HEIGHT,
            margin=0,
            styles={
                "position": "absolute",
                "top": f"{COMMAND_BAR_HEIGHT}px",
                "left": "0",
                "z-index": "40",
                "background": THEME["panel_alt"],
                "border": f"1px solid {THEME['border_strong']}",
                "box-shadow": "0 12px 28px rgba(0, 0, 0, .55)",
                "padding": "4px 0",
                "overflow-y": "auto",
            },
        )
        self._refresh_recent_menu()

    # ---- command bar -------------------------------------------------

    def _build_command_bar(self) -> None:
        self.brand_btn = pn.widgets.Button(
            label="◇ PYCOPTER  ▾",
            width=150,
            css_classes=["pycopter-brand"],
            stylesheets=[BRAND_CSS],
            margin=0,
        )
        self.breadcrumb = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
        self.showing_toggle = pn.widgets.RadioButtonGroup(
            options=list(SHOWING_OPTIONS),
            value="TOTAL",
            width=280,
            margin=(4, 0),
            stylesheets=[SEGMENT_CSS],
        )
        self.showing_label = pn.pane.HTML(
            '<span class="pycopter-showing-label">Showing</span>', margin=0
        )
        self.status_pane = pn.pane.HTML("", width=340, margin=0)
        self.plot_area_probe = ResultAreaProbe(width=0, height=0, margin=0)
        self.key_bindings = KeyBindings(width=0, height=0, margin=0)

        self.command_bar = pn.Row(
            pn.Column(self.brand_btn, self.app_menu, width=150, margin=0),
            self.breadcrumb,
            self.showing_label,
            self.showing_toggle,
            self.status_pane,
            self.plot_area_probe,
            self.key_bindings,
            sizing_mode="stretch_width",
            height=COMMAND_BAR_HEIGHT,
            css_classes=["pycopter-cmdbar"],
            margin=0,
        )
        self.warning_text = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
        self.show_clamped_btn = pn.widgets.Button(
            label="SHOW CLAMPED ROWS",
            width=170,
            margin=(3, 8),
            stylesheets=[WARNING_BUTTON_CSS],
        )
        self.warning_bar = pn.Row(
            self.warning_text,
            self.show_clamped_btn,
            sizing_mode="stretch_width",
            height=WARNING_BAR_HEIGHT,
            css_classes=["pycopter-warnbar"],
            visible=False,
            margin=0,
        )

    # ---- icon rail ---------------------------------------------------

    def _build_icon_rail(self) -> None:
        self.tab_selector = pn.widgets.RadioButtonGroup(
            options=list(INSPECTOR_TABS),
            value="ROTOR",
            orientation="vertical",
            sizing_mode="stretch_width",
            margin=0,
            stylesheets=[RAIL_TABS_CSS],
        )
        self.collapse_btn = pn.widgets.Button(
            label="‹",
            sizing_mode="stretch_width",
            css_classes=["pycopter-collapse"],
            stylesheets=[COLLAPSE_BUTTON_CSS],
            margin=0,
        )
        self.icon_rail = pn.Column(
            self.tab_selector,
            pn.Spacer(sizing_mode="stretch_both"),
            self.collapse_btn,
            width=ICON_RAIL_WIDTH,
            sizing_mode="stretch_height",
            css_classes=["pycopter-rail"],
            margin=0,
        )

    # ---- inspector ---------------------------------------------------

    def _build_inspector(self) -> None:
        self.inspector_head = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
        self.tab_panels = {
            "ROTOR": self._rotor_panel(),
            "BLADE": self._blade_panel(),
            "OPER": self._oper_panel(),
            "SOLVER": self._solver_panel(),
            "XFOIL": self._xfoil_panel(),
            "PROP": self._prop_panel(),
        }
        for name, panel in self.tab_panels.items():
            panel.visible = name == self.inspector_tab

        self.inspector_scroll = pn.Column(
            *self.tab_panels.values(),
            sizing_mode="stretch_both",
            scroll=True,
            css_classes=["pycopter-column"],
            margin=0,
        )
        self.inspector = pn.Column(
            self.inspector_head,
            self.inspector_scroll,
            pn.Row(
                self.revert_btn,
                self.solve_btn,
                self.cancel_btn,
                sizing_mode="stretch_width",
                css_classes=["pycopter-inspector-foot"],
                margin=0,
            ),
            width=INSPECTOR_WIDTH,
            sizing_mode="stretch_height",
            css_classes=["pycopter-inspector"],
            margin=0,
        )

    def _panel_column(self, *objects) -> pn.Column:
        return pn.Column(
            *objects,
            sizing_mode="stretch_width",
            margin=(6, 8),
        )

    def _rotor_panel(self) -> pn.Column:
        return self._panel_column(
            self._pair(
                self._field(self.rotor_system_type, key="rotor_system_type"),
                self._field(self.num_blades, "count", "num_blades"),
            ),
            self._pair(
                self._field(self.rotor_diam, "m", "rotor_diam"),
                self._field(self.root_cutout, "rr", "root_cutout"),
            ),
            self._pair(
                self._field(self.headspeed_rpm, "rpm", "headspeed_rpm"),
                self._field(self.lower_rotor_speed_ratio, "ratio", "lower_rotor_speed_ratio"),
            ),
            self._pair(
                self._field(self.lower_headspeed_rpm, "calc", derived=True),
                self._field(self.coaxial_spacing, "zr", "coaxial_spacing_ratio"),
            ),
            self._pair(
                self._field(self.lower_collective_offset, "deg", "lower_collective_offset_deg"),
                self._field(self.lower_rotor_scale, "ratio", "lower_rotor_scale"),
            ),
            self._subhead("Derived"),
            self._pair(
                self._field(self.tip_speed_display, "ms", derived=True),
                self._field(self.tip_mach_display, "mach", derived=True),
            ),
            self._pair(self._field(self.disk_area_display, "m2", derived=True)),
        )

    def _blade_panel(self) -> pn.Column:
        return self._panel_column(
            self._pair(
                self._field(self.airfoil, key="airfoil"),
                self._field(self.geometry_mode, key="geometry_mode"),
            ),
            self._pair(
                self._field(self.chord, "m", "chord"),
                self._field(self.station_pitch_axis, "c", "station_pitch_axis_frac"),
            ),
            self._pair(
                self._field(self.root_twist, "deg", "root_twist_deg"),
                self._field(self.tip_twist, "deg", "tip_twist_deg"),
            ),
            self._pair(self._field(self.station_count, "count", "station_count")),
            self._subhead("Derived"),
            self._pair(
                self._field(self.solidity_display, "ratio", derived=True),
                self._field(self.blade_area_display, "m2", derived=True),
            ),
            self._pair(self._field(self.aspect_ratio_display, "ratio", derived=True)),
            self._subhead("Station table"),
            self.station_table,
            self.reset_stations_btn,
        )

    def _oper_panel(self) -> pn.Column:
        return self._panel_column(
            self._pair(
                self._field(self.gross, "kg", "gross"),
                self._field(self.flight_mode, key="flight_mode"),
            ),
            self._pair(
                self._field(self.density, "kgm3", "density"),
                self._field(self.kinematic_viscosity, "m2s", "kinematic_viscosity_m2_s"),
            ),
            self._pair(self._field(self.speed_of_sound, "ms", "speed_of_sound_m_s")),
            self._subhead("Atmosphere helper"),
            self._pair(
                self._field(self.altitude, "m", "altitude_m"),
                self._field(self.temperature, "celsius", "temperature_C"),
            ),
            self.apply_isa_btn,
            pn.pane.HTML(
                '<div class="pycopter-menu-note">Altitude and temperature are display'
                " convenience only. They reach the solver only through this button,"
                " which writes density, viscosity and speed of sound.</div>",
                sizing_mode="stretch_width",
                margin=0,
            ),
            self._subhead("Derived"),
            self._pair(
                self._field(self.target_thrust_display, "n", derived=True),
                self._field(self.disk_loading_display, "ratio", derived=True),
            ),
        )

    def _solver_panel(self) -> pn.Column:
        return self._panel_column(
            self._pair(
                self._field(self.trim_mode, key="trim_mode"),
                self._field(self.collective_pitch, "deg", "collective_pitch_deg"),
            ),
            self._pair(
                self._field(self.blade_element_count, "count", "blade_element_count"),
                self._field(self.element_spacing, key="element_spacing"),
            ),
            self._pair(
                self._field(self.min_collective, "deg", "min_collective_deg"),
                self._field(self.max_collective, "deg", "max_collective_deg"),
            ),
            self._pair(
                self._field(self.thrust_tolerance, "ratio", "thrust_tolerance"),
                self._field(self.max_trim_iterations, "count", "max_trim_iterations"),
            ),
            self._pair(
                self._field(self.tip_loss_model, key="tip_loss_model"),
                self._field(self.root_loss_model, key="root_loss_model"),
            ),
            self._pair(self._field(self.induced_power_factor, "ratio", "induced_power_factor")),
            self._subhead("Coaxial"),
            self._pair(self._field(self.coaxial_trim_mode, key="coaxial_trim_mode")),
        )

    def _xfoil_panel(self) -> pn.Column:
        return self._panel_column(
            self._subhead("MPI workers"),
            self.mpi_strip,
            self._pair(
                self._field(self.new_polars, key="new_polars"),
                self._field(self.xfoil_workers, "count", "xfoil_parallel_workers"),
            ),
            self._pair(
                self._field(self.polar_alpha_min, "deg", "polar_alpha_min_deg"),
                self._field(self.polar_alpha_max, "deg", "polar_alpha_max_deg"),
            ),
            self._pair(
                self._field(self.polar_alpha_step, "deg", "polar_alpha_step_deg"),
                self._field(self.polar_n_crit, "ratio", "polar_n_crit"),
            ),
            self._pair(
                self._field(self.polar_reynolds_bin, "re", "polar_reynolds_bin"),
                self._field(self.polar_mach_bin, "mach", "polar_mach_bin"),
            ),
            self._pair(
                self._field(self.xfoil_max_iterations, "count", "xfoil_max_iterations"),
                self._field(self.xfoil_timeout, "s", "xfoil_timeout_s"),
            ),
            self._subhead("Advanced"),
            self._pair(self._field(self.xfoil_backend, key="xfoil_parallel_backend")),
            self._pair(self._field(self.xfoil_cache_dir, key="xfoil_cache_directory")),
        )

    def _prop_panel(self) -> pn.Column:
        self.electric_fields = pn.Column(
            self._pair(
                self._field(self.battery_capacity, "wh", "battery_capacity_Wh"),
                self._field(self.battery_usable, "ratio", "battery_usable_fraction"),
            ),
            self._pair(
                self._field(self.motor_efficiency, "ratio", "motor_efficiency"),
                self._field(self.esc_efficiency, "ratio", "esc_efficiency"),
            ),
            sizing_mode="stretch_width",
            margin=0,
        )
        self.fossil_fields = pn.Column(
            self._pair(
                self._field(self.fuel_capacity, "kg", "fuel_capacity_kg"),
                self._field(self.sfc, "kgkwh", "specific_fuel_consumption_kg_per_kWh"),
            ),
            sizing_mode="stretch_width",
            margin=0,
            visible=False,
        )
        return self._panel_column(
            self._pair(
                self._field(self.propulsion_model, key="propulsion_model"),
                self._field(self.transmission_loss, "ratio", "transmission_loss"),
            ),
            self.electric_fields,
            self.fossil_fields,
            self._subhead("Active model outputs"),
            self.propulsion_display,
            self._subhead("Legacy forward flight"),
            self._pair(
                self._field(self.velocity, "kmh", "velocity_kmh"),
                self._field(self.fpa, "m2", "fpa"),
            ),
        )

    # ---- canvas ------------------------------------------------------

    def _build_canvas(self) -> None:
        self.metric_strip = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
        self.plot_select = pn.widgets.Select(options=[], sizing_mode="stretch_width", margin=(2, 4))
        self.generate_plot_btn = pn.widgets.Button(
            label="GENERATE",
            width=100,
            disabled=True,
            margin=(2, 4),
            css_classes=["pycopter-primary"],
            stylesheets=[PRIMARY_BUTTON_CSS],
        )
        self.overlay_runs = pn.widgets.Checkbox(
            label="overlay selected runs (0)",
            value=False,
            css_classes=["pycopter-checkbox"],
            stylesheets=[CHECKBOX_CSS],
            width=190,
            margin=(6, 4),
        )
        self.plot_pane = pn.pane.Matplotlib(
            self.plot_fig,
            sizing_mode="stretch_both",
            align="start",
            tight=False,
            dpi=PLOT_RENDER_DPI,
            css_classes=["pycopter-plot-frame"],
            stylesheets=[PLOT_FRAME_CSS],
        )
        self.summary_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["Metric", "Value"]),
            show_index=False,
            sizing_mode="stretch_both",
            layout="fit_data_stretch",
            selectable=True,
            editors={"Metric": None, "Value": None},
            configuration=RESULT_TABLE_CONFIGURATION,
            css_classes=["pycopter-result-table"],
            stylesheets=[RESULT_TABLE_CSS],
        )
        self.load_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            sizing_mode="stretch_both",
            layout="fit_data_table",
            selectable=True,
            frozen_columns=["r_over_R"],
            configuration=RESULT_TABLE_CONFIGURATION,
            css_classes=["pycopter-result-table", "pycopter-load-table"],
            stylesheets=[LOAD_TABLE_CSS],
        )
        self.loads_shading = pn.widgets.RadioButtonGroup(
            options=list(LOADS_SHADING_OPTIONS),
            value="ALPHA",
            width=190,
            margin=(2, 4),
            stylesheets=[SEGMENT_CSS],
        )
        self.loads_clamped_only = pn.widgets.Checkbox(
            label="clamped rows only",
            value=False,
            css_classes=["pycopter-checkbox"],
            stylesheets=[CHECKBOX_CSS],
            width=150,
            margin=(6, 4),
        )
        self.loads_footer = pn.pane.HTML(
            "", sizing_mode="stretch_width", css_classes=["pycopter-table-foot"], margin=0
        )
        self.load_table_container = pn.Column(
            pn.Row(
                self.loads_shading,
                self.loads_clamped_only,
                sizing_mode="stretch_width",
                css_classes=["pycopter-canvas-controls"],
                margin=0,
            ),
            pn.Column(
                self.load_table,
                sizing_mode="stretch_both",
                css_classes=["pycopter-table-scroll"],
                margin=0,
            ),
            self.loads_footer,
            sizing_mode="stretch_both",
            margin=0,
        )
        self.polar_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["airfoil", "reynolds", "mach", "status"]),
            show_index=False,
            sizing_mode="stretch_both",
            layout="fit_data_stretch",
            selectable=True,
            configuration=RESULT_TABLE_CONFIGURATION,
            css_classes=["pycopter-result-table"],
            stylesheets=[RESULT_TABLE_CSS],
        )
        self.polar_footer = pn.pane.HTML(
            "", sizing_mode="stretch_width", css_classes=["pycopter-table-foot"], margin=0
        )
        self.result_tabs = pn.Tabs(
            ("PLOT", self.plot_pane),
            ("SUMMARY", self.summary_table),
            ("LOADS", self.load_table_container),
            (
                "POLARS",
                pn.Column(self.polar_table, self.polar_footer, sizing_mode="stretch_both", margin=0),
            ),
            dynamic=False,
            sizing_mode="stretch_both",
            margin=0,
        )
        self.canvas = pn.Column(
            self.metric_strip,
            pn.Row(
                self.plot_select,
                self.generate_plot_btn,
                self.overlay_runs,
                sizing_mode="stretch_width",
                css_classes=["pycopter-canvas-controls"],
                margin=0,
            ),
            self.result_tabs,
            sizing_mode="stretch_both",
            css_classes=["pycopter-results"],
            margin=0,
        )

    # ---- run rail ----------------------------------------------------

    def _build_run_rail(self) -> None:
        self.run_rail_head = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
        self.run_rail = RunRail(rows=[], sizing_mode="stretch_both", margin=0)
        self.set_baseline_btn = pn.widgets.Button(
            label="SET BASELINE",
            sizing_mode="stretch_width",
            stylesheets=[CONDENSED_BUTTON_CSS],
        )
        self.export_runs_btn = pn.widgets.FileDownload(
            callback=self._save_runs_csv_callback,
            filename="pycopter_session_runs.csv",
            label="EXPORT CSV",
            sizing_mode="stretch_width",
            stylesheets=[CONDENSED_BUTTON_CSS],
        )
        self.delete_run_btn = pn.widgets.Button(
            label="DELETE",
            sizing_mode="stretch_width",
            css_classes=["pycopter-danger"],
            stylesheets=[DANGER_BUTTON_CSS],
        )
        self.run_rail_region = pn.Column(
            self.run_rail_head,
            pn.pane.HTML(
                '<div class="pycopter-runrail-cols"><span style="flex:1 1 auto">Run</span>'
                '<span style="width:56px;text-align:right">Power W</span>'
                '<span style="width:56px;text-align:right">FoM</span>'
                '<span style="width:62px;text-align:right">&Delta; Power</span></div>',
                sizing_mode="stretch_width",
                margin=0,
            ),
            pn.Column(
                self.run_rail,
                sizing_mode="stretch_both",
                scroll=True,
                css_classes=["pycopter-column"],
                margin=0,
            ),
            pn.Row(
                self.set_baseline_btn,
                self.export_runs_btn,
                self.delete_run_btn,
                sizing_mode="stretch_width",
                css_classes=["pycopter-runrail-foot"],
                margin=0,
            ),
            width=RUN_RAIL_WIDTH,
            sizing_mode="stretch_height",
            css_classes=["pycopter-runrail"],
            margin=0,
        )

    # ---- log ---------------------------------------------------------

    def _build_log(self) -> None:
        self.log_splitter = LogSplitter(
            height=LOG_GRIP_HEIGHT, sizing_mode="stretch_width", margin=0
        )
        self.log_line = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
        self.log_counts = pn.pane.HTML("", width=260, margin=0)
        self.log_toggle_btn = pn.widgets.Button(
            label="EXPAND ˄", width=96, margin=(3, 6), stylesheets=[CONDENSED_BUTTON_CSS]
        )
        self.log_filter = pn.widgets.RadioButtonGroup(
            options=list(LOG_SEVERITIES),
            value="ALL",
            orientation="vertical",
            width=90,
            margin=0,
            css_classes=["pycopter-log-filter"],
            stylesheets=[LOG_FILTER_CSS],
        )
        self.output_log = pn.widgets.Terminal(
            output="",
            # Starts at the collapsed inner height so the terminal is not
            # rendering more rows than the clipped strip can show.
            height=max(20, LOG_MIN_HEIGHT - LOG_STRIP_HEIGHT - LOG_TERMINAL_INSET),
            sizing_mode="stretch_width",
            options={
                "convertEol": True,
                "cursorBlink": False,
                "disableStdin": True,
                "fontFamily": MONO_FONT,
                "fontSize": 12,
                "scrollback": int(DEFAULT_CONFIG["log_scrollback_lines"]),
                "theme": {
                    "background": THEME["field"],
                    "foreground": THEME["text"],
                    "cursor": THEME["accent"],
                    "selectionBackground": "#2a4666",
                },
            },
            css_classes=["pycopter-log"],
            stylesheets=[TERMINAL_CSS],
        )
        self.log_filter_counts = pn.pane.HTML("", width=42, margin=0)
        self.log_body = pn.Row(
            pn.Row(self.log_filter, self.log_filter_counts, width=132, margin=0),
            self.output_log,
            height=max(20, LOG_MIN_HEIGHT - LOG_STRIP_HEIGHT - LOG_TERMINAL_INSET),
            sizing_mode="stretch_width",
            css_classes=["pycopter-log-body"],
            margin=0,
        )
        self.log_region = pn.Column(
            pn.Row(
                self.log_line,
                self.log_counts,
                self.log_toggle_btn,
                sizing_mode="stretch_width",
                css_classes=["pycopter-log-strip"],
                margin=0,
            ),
            self.log_body,
            height=LOG_MIN_HEIGHT,
            sizing_mode="stretch_width",
            css_classes=["pycopter-log-region"],
            margin=0,
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        self.brand_btn.on_click(lambda _: self._toggle_menu())
        self.new_btn.on_click(lambda _: self._new_config())
        self.load_btn.on_click(lambda _: self._load_config())
        self.preferences_btn.on_click(lambda _: self._toggle_preferences())
        self.key_bindings.param.watch(lambda *_: self._on_key_command(), "command")
        self.save_as_name.param.watch(lambda *_: self._on_save_as_name(), "value")

        self.tab_selector.param.watch(lambda *_: self._on_tab_changed(), "value")
        self.collapse_btn.on_click(lambda _: self._toggle_inspector())
        self.showing_toggle.param.watch(lambda *_: self._on_showing_changed(), "value")

        self.revert_btn.on_click(lambda _: self._revert_inputs())
        self.solve_btn.on_click(lambda _: self.solve_as_new_run())
        self.cancel_btn.on_click(lambda _: self._cancel_active_job())
        self.apply_isa_btn.on_click(lambda _: self._apply_isa())
        self.reset_stations_btn.on_click(lambda _: self._reset_station_rows_from_uniform())
        self.show_clamped_btn.on_click(lambda _: self._show_clamped_rows())

        self.propulsion_model.param.watch(lambda *_: self._on_propulsion_changed(), "value")
        self.flight_mode.param.watch(lambda *_: self._update_plot_options(), "value")

        self.generate_plot_btn.on_click(lambda _: self._generate_plot())
        self.plot_select.param.watch(lambda *_: self._sync_enabled_state(), "value")
        self.overlay_runs.param.watch(lambda *_: self._generate_plot(), "value")
        self.loads_shading.param.watch(lambda *_: self._refresh_load_table(), "value")
        self.loads_clamped_only.param.watch(lambda *_: self._refresh_load_table(), "value")

        self.run_rail.param.watch(lambda *_: self._on_rail_event(), "event")
        self.set_baseline_btn.on_click(lambda _: self._set_baseline())
        self.delete_run_btn.on_click(lambda _: self._delete_active_run())

        self.log_toggle_btn.on_click(lambda _: self._toggle_log())
        self.log_filter.param.watch(lambda *_: self._on_log_filter_changed(), "value")
        self.log_splitter.param.watch(lambda *_: self._on_log_resized(), "height_px")
        self.plot_area_probe.param.watch(
            lambda *_: self._on_plot_area_resized(), ["width_px", "height_px"]
        )

        for key, widget in self._bindings.items():
            widget.param.watch(lambda *_: self._on_input_changed(), "value")
        self.station_table.param.watch(lambda *_: self._on_input_changed(), "value")

    def _on_key_command(self) -> None:
        command = str(self.key_bindings.command).split("|")[0]
        if command == "new":
            self._new_config()
        elif command == "open":
            self.app_menu.visible = True

    def _on_save_as_name(self) -> None:
        name = str(self.save_as_name.value).strip() or "pycopter_config.json"
        if not name.endswith(".json"):
            name += ".json"
        self.save_as_download.filename = name

    def _toggle_menu(self) -> None:
        self.app_menu.visible = not self.app_menu.visible

    def _toggle_preferences(self) -> None:
        self.preferences_panel.visible = not self.preferences_panel.visible
        self.app_menu.height = (
            MENU_HEIGHT + MENU_PREFERENCES_HEIGHT
            if self.preferences_panel.visible
            else MENU_HEIGHT
        )

    def _on_tab_changed(self) -> None:
        self.inspector_tab = str(self.tab_selector.value)
        for name, panel in self.tab_panels.items():
            panel.visible = name == self.inspector_tab
        self._refresh_inspector_head()

    def _toggle_inspector(self) -> None:
        """Give the inspector's 320 px to the canvas without resizing anything else."""
        collapsed = self.inspector.visible
        self.inspector.visible = not collapsed
        self.collapse_btn.label = "›" if collapsed else "‹"

    def _on_showing_changed(self) -> None:
        self.showing = SHOWING_OPTIONS[str(self.showing_toggle.value)]
        self._refresh_result_views()
        if self._plot_save_available:
            self._generate_plot()

    def _on_propulsion_changed(self) -> None:
        """Swap which model's fields, outputs and plots are offered."""
        self._update_plot_options()
        self._refresh_propulsion_display()
        if self.current_case is not None:
            self._update_summary_table(
                self.current_case,
                normalize_config(
                    {**self._active_inputs(), "propulsion_model": self._propulsion_model()}
                ),
            )

    def _on_input_changed(self) -> None:
        if self._applying:
            return
        self._update_derived_displays()
        self._refresh_edited_marks()
        self._sync_enabled_state()
        self._refresh_breadcrumb()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _current_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"version": CONFIG_VERSION}
        for key, widget in self._bindings.items():
            config[key] = widget.value
        # Compatibility keys the Web UI no longer edits directly.
        config["headspeed_input_mode"] = "rpm"
        config["tip_speed_mach"] = 0.0
        return normalize_config(config)

    def _station_rows(self) -> list[dict[str, Any]]:
        frame = self.station_table.value.copy()
        return frame.to_dict(orient="records")

    def _apply_config(
        self,
        config: dict[str, Any],
        station_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        cfg = normalize_config(config)
        self._applying = True
        try:
            for key, widget in self._bindings.items():
                value = cfg.get(key)
                if isinstance(widget, pn.widgets.Select):
                    if value in widget.options.values() or value in list(widget.options):
                        widget.value = value
                elif isinstance(widget, pn.widgets.IntInput):
                    widget.value = int(value)
                elif isinstance(widget, pn.widgets.FloatInput):
                    widget.value = float(value)
                else:
                    widget.value = str(value)
            self.station_table.value = pd.DataFrame(
                station_rows or station_rows_from_uniform(cfg)
            )
        finally:
            self._applying = False

        self._baseline_config = cfg
        self.runs.max_runs = int(cfg["max_session_runs"])
        self.current_case = None
        self._apply_preferences(cfg)
        self._update_derived_displays()
        self._refresh_edited_marks()
        self._sync_enabled_state()
        self._update_plot_options()
        self._refresh_all()

    def _apply_preferences(self, cfg: dict[str, Any]) -> None:
        self.plot_pane.dpi = int(cfg["plot_render_dpi"])
        options = dict(self.output_log.options)
        options["scrollback"] = int(cfg["log_scrollback_lines"])
        self.output_log.options = options

    def _propulsion_model(self) -> str:
        return str(self.propulsion_model.value)

    def _revert_inputs(self) -> None:
        """Put every field back to the last loaded config or run snapshot."""
        rows = self._baseline_config.get("station_rows")
        self._apply_config(self._baseline_config, rows)
        self._log("Inputs reverted to the last loaded configuration.")

    def _apply_isa(self) -> None:
        try:
            air = isa_atmosphere(float(self.altitude.value), float(self.temperature.value))
        except ValueError as err:
            self._log(f"ERROR - {err}")
            return
        self.density.value = round(air["density_kg_m3"], 6)
        self.kinematic_viscosity.value = float(f"{air['kinematic_viscosity_m2_s']:.4g}")
        self.speed_of_sound.value = round(air["speed_of_sound_m_s"], 2)
        self._log(
            f"ISA at {float(self.altitude.value):.0f} m, {float(self.temperature.value):.1f} C: "
            f"density {air['density_kg_m3']:.4f} [kg/m3] | "
            f"kinematic viscosity {air['kinematic_viscosity_m2_s']:.3e} [m2/s] | "
            f"speed of sound {air['speed_of_sound_m_s']:.1f} [m/s] "
            f"(ISA temperature would be {air['isa_temperature_C']:.1f} C)."
        )

    def _reset_station_rows_from_uniform(self) -> None:
        cfg = self._current_config()
        self.station_table.value = pd.DataFrame(station_rows_from_uniform(cfg))
        self._log("Geometry control points reset from root/tip twist inputs.")

    def _refresh_edited_marks(self) -> None:
        """Outline every field that differs from the loaded config or preset."""
        edited = 0
        for key, wrapper in self._field_wrappers.items():
            widget = self._bindings[key]
            changed = not _values_match(self._baseline_config.get(key), widget.value)
            classes = list(getattr(wrapper, "_pycopter_classes", ["pycopter-field"]))
            if changed:
                classes.append("pycopter-edited")
                edited += 1
            if list(wrapper.css_classes or []) != classes:
                wrapper.css_classes = classes
            normal, marked = self._field_stylesheets[key]
            wanted = marked if changed else normal
            if list(widget.stylesheets or []) != wanted:
                widget.stylesheets = list(wanted)
        self._edited_count = edited
        self._refresh_inspector_head()

    def _update_derived_displays(self) -> None:
        try:
            config = self._current_config()
            derived = derived_geometry(config, self._station_rows())
        except Exception:
            return
        self.lower_headspeed_rpm.value = derived["lower_headspeed_rpm"]
        self.tip_speed_display.value = f"{derived['tip_speed_m_s']:.1f}"
        self.tip_mach_display.value = f"{derived['tip_speed_mach']:.3f}"
        self.disk_area_display.value = f"{derived['disk_area_m2']:.4f}"
        self.solidity_display.value = f"{derived['solidity']:.4f}"
        self.blade_area_display.value = f"{derived['blade_area_m2']:.4f}"
        self.aspect_ratio_display.value = f"{derived['aspect_ratio']:.2f}"
        self.target_thrust_display.value = f"{derived['target_thrust_N']:.3f}"
        self.disk_loading_display.value = f"{derived['disk_loading_N_m2']:.1f} N/m2"

    # ------------------------------------------------------------------
    # Config file handling
    # ------------------------------------------------------------------

    def _save_config_callback(self) -> io.BytesIO:
        payload = {
            "version": CONFIG_VERSION,
            "config": self._current_config(),
            "station_rows": self._station_rows(),
        }
        data = io.BytesIO(json.dumps(payload, indent=4).encode("utf-8"))
        data.seek(0)
        return data

    def _save_plot_callback(self) -> io.BytesIO:
        if not self._plot_save_available:
            self._log("WARNING - No plot has been generated yet. Generate a plot before saving it.")
            return self._empty_export()

        data = io.BytesIO()
        self.plot_fig.savefig(
            data,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor=self.plot_fig.get_facecolor(),
            edgecolor="none",
        )
        data.seek(0)
        return data

    def _save_summary_callback(self) -> io.BytesIO:
        frame = self.summary_table.value
        if self.current_case is None or frame.empty:
            self._log("WARNING - No summary table has been generated yet. Calculate hover before saving it.")
            return self._empty_export()
        return self._dataframe_csv(frame)

    def _save_loads_callback(self) -> io.BytesIO:
        frame = self.load_table.value
        if self.current_case is None or frame.empty:
            self._log("WARNING - No blade element loads table has been generated yet. Calculate hover before saving it.")
            return self._empty_export()
        return self._dataframe_csv(frame)

    def _save_runs_callback(self) -> io.BytesIO:
        if not len(self.runs):
            self._log("WARNING - No runs have been solved yet, so there is no session to export.")
            return self._empty_export()
        data = io.BytesIO(json.dumps(self.runs.to_payload(), indent=4).encode("utf-8"))
        data.seek(0)
        return data

    def _save_runs_csv_callback(self) -> io.BytesIO:
        rows = self.runs.csv_rows()
        if not rows:
            self._log("WARNING - No runs have been solved yet, so there is no session to export.")
            return self._empty_export()
        return self._dataframe_csv(pd.DataFrame(rows))

    def _dataframe_csv(self, frame: pd.DataFrame) -> io.BytesIO:
        data = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
        data.seek(0)
        return data

    def _empty_export(self) -> io.BytesIO:
        data = io.BytesIO(b"")
        data.seek(0)
        return data

    def _load_config(self) -> None:
        if not self.load_file.value:
            self._log("No configuration file selected.")
            return
        try:
            payload = json.loads(self.load_file.value.decode("utf-8"))
            config = payload.get("config", payload)
            rows = payload.get("station_rows")
            incoming = normalize_config(config)
            self._log_config_diff(incoming)
            self.config_filename = str(self.load_file.filename or "config.json")
            self._remember_recent_config(self.config_filename, config, rows)
            self._apply_config(config, rows)
            self.app_menu.visible = False
            self._log(f"Configuration '{self.config_filename}' loaded.")
        except Exception as err:
            self._log(f"ERROR - Could not load configuration: {err}")

    def _log_config_diff(self, incoming: dict[str, Any]) -> None:
        changes = diff_inputs(self._current_config(), incoming)
        if not changes:
            self._log("Incoming configuration matches the current session inputs.")
            return
        self._log(f"Incoming configuration changes {len(changes)} fields:")
        for change in changes[:12]:
            self._log(f"  {change['key']}: {change['from']} -> {change['to']}")
        if len(changes) > 12:
            self._log(f"  ... and {len(changes) - 12} more.")

    def _remember_recent_config(
        self,
        filename: str,
        config: dict[str, Any],
        rows: list[dict[str, Any]] | None,
    ) -> None:
        payload = {"config": config, "station_rows": rows}
        self.recent_configs = [
            item for item in self.recent_configs if item[0] != filename
        ]
        self.recent_configs.insert(0, (filename, payload))
        self.recent_configs = self.recent_configs[:5]
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        if not self.recent_configs:
            self.recent_menu.objects = [
                pn.pane.HTML(
                    '<div class="pycopter-menu-note">none this session</div>',
                    sizing_mode="stretch_width",
                    margin=0,
                )
            ]
            return
        buttons = []
        for filename, payload in self.recent_configs:
            button = pn.widgets.Button(
                label=f"  {filename}",
                sizing_mode="stretch_width",
                stylesheets=[MENU_ITEM_CSS],
                margin=0,
            )
            button.on_click(
                lambda _event, saved=payload, name=filename: self._apply_recent(name, saved)
            )
            buttons.append(button)
        self.recent_menu.objects = buttons

    def _apply_recent(self, filename: str, payload: dict[str, Any]) -> None:
        self.config_filename = filename
        self._apply_config(payload["config"], payload.get("station_rows"))
        self.app_menu.visible = False
        self._log(f"Recent configuration '{filename}' applied.")

    def _new_config(self) -> None:
        self.config_filename = "untitled.json"
        self._apply_config(DEFAULT_CONFIG, DEFAULT_STATION_ROWS)
        self.runs.clear()
        self.checked_run_ids = []
        self.active_run_id = None
        self.current_case = None
        self._clear_results()
        self.app_menu.visible = False
        self._log("New session started.")

    def _clear_results(self) -> None:
        superseded = self.plot_fig
        self.plot_fig = self._blank_figure("Solve a run to see results.")
        self._current_plot_name = ""
        self._plot_save_available = False
        self.plot_pane.object = self.plot_fig
        if superseded is not self.plot_fig:
            plt.close(superseded)
        self._set_read_only_table_value(self.summary_table, pd.DataFrame(columns=["Metric", "Value"]))
        self._set_read_only_table_value(self.load_table, pd.DataFrame())
        self._set_read_only_table_value(
            self.polar_table, pd.DataFrame(columns=["airfoil", "reynolds", "mach", "status"])
        )
        self._sync_enabled_state()
        self._update_plot_options()
        self._refresh_all()

    # ------------------------------------------------------------------
    # Runs and the calculation worker
    # ------------------------------------------------------------------

    def solve_as_new_run(self, background: bool | None = None) -> RunRecord | None:
        """Solve the current inputs as a new, persisted run."""
        if self._active_job is not None:
            self._log("A calculation is already running. Cancel it before starting another.")
            return None
        try:
            config = self._current_config()
            station_rows = self._station_rows()
        except Exception as err:
            self._log(f"ERROR - {err}")
            return None

        inputs = {**config, "station_rows": station_rows}
        record = self.runs.create(self._run_label(config), inputs, status="running")
        self.active_run_id = record.run_id
        self._log("")
        self._log(f"{record.run_id} solving {record.label}...")
        self._refresh_rail()
        self._refresh_status()

        provider = self._xfoil_provider_for_config(config)

        def work(cancel):
            return run_hover_case(config, station_rows, polar_provider=provider)

        return self._dispatch_job(record.run_id, "hover", work, background)

    def _run_label(self, config: dict[str, Any]) -> str:
        if config["rotor_system_type"] == "coaxial":
            return f"coax z/R {float(config['coaxial_spacing_ratio']):.2f}"
        return f"single {float(config['headspeed_rpm']):.0f} rpm"

    def _dispatch_job(self, run_id, kind, work, background: bool | None):
        use_background = self._background_enabled if background is None else background
        self._cancel_event = threading.Event()
        self._active_job = {"run_id": run_id, "kind": kind}
        self._sync_enabled_state()

        if not use_background:
            try:
                outcome = work(self._cancel_event.is_set)
                self._queue.put(("finish", (run_id, kind, "done", outcome)))
            except Exception as err:
                self._queue.put(
                    ("finish", (run_id, kind, "failed", (err, traceback.format_exc())))
                )
            self._drain_worker_queue()
            return self.runs.get(run_id)

        def target():
            try:
                outcome = work(self._cancel_event.is_set)
                self._queue.put(("finish", (run_id, kind, "done", outcome)))
            except Exception as err:
                self._queue.put(
                    ("finish", (run_id, kind, "failed", (err, traceback.format_exc())))
                )

        self._worker_thread = threading.Thread(target=target, daemon=True)
        self._worker_thread.start()
        return self.runs.get(run_id)

    def _cancel_active_job(self) -> None:
        if self._active_job is None:
            return
        self._cancel_event.set()
        self._log("Cancel requested. The run stops after the point in flight.")

    def _drain_worker_queue(self) -> None:
        """Apply calculation-thread progress on the Panel event loop."""
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                if kind == "log":
                    self._log(payload)
                elif kind == "xfoil":
                    self._on_xfoil_event(*payload)
                elif kind == "sweep_point":
                    self._on_sweep_point(*payload)
                elif kind == "finish":
                    self._on_job_finished(*payload)
            except Exception as err:  # keep the poller alive
                self._log(f"ERROR - {err}")

    def _on_xfoil_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "batch_start":
            self._xfoil_state = {
                "state": "running",
                "workers": int(payload.get("workers", 0)),
                "jobs": int(payload.get("jobs", 0)),
                "error": "",
            }
        elif event == "batch_done":
            self._xfoil_state = {
                "state": "ok",
                "workers": int(payload.get("workers", 0)),
                "jobs": 0,
                "error": "",
            }
        elif event == "batch_failed":
            self._xfoil_state = {
                "state": "failed",
                "workers": self._xfoil_state.get("workers", 0),
                "jobs": 0,
                "error": str(payload.get("error", "")),
            }
        self._refresh_mpi_strip()
        self._refresh_status()

    def _on_sweep_point(self, parent_id, index, total, z_over_R, point) -> None:
        parent = self.runs.get(parent_id)
        if parent is None:
            return
        self.runs.update(parent_id, progress=(index, total))
        child_inputs = {**parent.inputs, "coaxial_spacing_ratio": z_over_R}
        if point.result is not None:
            self.runs.create(
                f"coax z/R {z_over_R:.2f}",
                child_inputs,
                status="done",
                parent_id=parent_id,
            )
            self.runs.update(
                self.runs.records[-1].run_id,
                result=point.result,
                warnings=point.result.warnings,
            )
        else:
            self.runs.create(
                f"coax z/R {z_over_R:.2f}",
                child_inputs,
                status="failed",
                parent_id=parent_id,
            )
            self.runs.update(self.runs.records[-1].run_id, error=point.error)
        self._refresh_rail()

    def _on_job_finished(self, run_id, kind, outcome, payload) -> None:
        self._active_job = None
        self._worker_thread = None
        if outcome == "failed":
            error, trace = payload
            self.runs.update(run_id, status="failed", error=f"{type(error).__name__}: {error}")
            self._log(f"ERROR - {run_id} failed: {error}")
            for line in trace.strip().splitlines()[-4:]:
                self._log(f"  {line}")
            self._xfoil_state["state"] = "failed"
        elif kind == "hover":
            self._finish_hover_run(run_id, payload)
        elif kind == "sweep":
            self._finish_sweep_run(run_id, payload)
        self._sync_enabled_state()
        self._refresh_all()

    def _finish_hover_run(self, run_id: str, case: HoverCase) -> None:
        warnings = tuple(case.result.warnings)
        self.runs.update(run_id, status="done", result=case.result, warnings=warnings)
        self.current_case = case
        self.active_run_id = run_id
        record = self.runs.get(run_id)
        self._log_hover_result(case, record.inputs)
        if str(record.inputs.get("flight_mode")) == "forward_flight":
            self._calculate_forward_flight()
        self._update_plot_options()
        self._refresh_result_views()
        self._generate_plot()

    def _finish_sweep_run(self, run_id: str, payload) -> None:
        sweep, base_config = payload
        status = "cancelled" if sweep.cancelled else "done"
        warnings = []
        if sweep.failed_points:
            warnings.append(
                f"{len(sweep.failed_points)} of {sweep.requested_points} spacing points did not converge"
            )
        self.runs.update(
            run_id,
            status=status,
            progress=(len(sweep.points), sweep.requested_points),
            warnings=tuple(warnings),
        )
        if not sweep.solved_points:
            self._log("WARNING - The spacing sweep produced no converged points.")
            return
        self._log_sweep_summary(sweep)
        figure = self._figure_from_interference_sweep(sweep, base_config)
        self._install_figure(figure, "Interference Loss vs Spacing")

    def _select_run(self, run_id: str) -> None:
        record = self.runs.get(run_id)
        if record is None:
            return
        self.active_run_id = run_id
        if record.result is not None:
            rotor = build_rotor_spec(normalize_config(record.inputs), record.inputs.get("station_rows", []))
            self.current_case = HoverCase(
                rotor=rotor,
                result=record.result,
                system_type="coaxial" if record.is_coaxial else "single",
            )
        else:
            self.current_case = None
        self._update_plot_options()
        self._refresh_all()
        if self.current_case is not None:
            self._generate_plot()

    def _on_rail_event(self) -> None:
        parts = str(self.run_rail.event).split("|")
        if len(parts) < 2:
            return
        kind, run_id = parts[0], parts[1]
        if kind == "select":
            self._select_run(run_id)
        elif kind == "check":
            if run_id not in self.checked_run_ids:
                self.checked_run_ids.append(run_id)
            self._on_checked_runs_changed()
        elif kind == "uncheck":
            self.checked_run_ids = [item for item in self.checked_run_ids if item != run_id]
            self._on_checked_runs_changed()
        elif kind == "toggle":
            if run_id in self.collapsed_parents:
                self.collapsed_parents.discard(run_id)
            else:
                self.collapsed_parents.add(run_id)
            self._refresh_rail()

    def _on_checked_runs_changed(self) -> None:
        self.overlay_runs.label = f"overlay selected runs ({len(self.checked_run_ids)})"
        self._refresh_rail()
        if self.overlay_runs.value and self._plot_save_available:
            self._generate_plot()

    def _set_baseline(self) -> None:
        if self.active_run_id is None:
            self._log("Select a run before setting it as the baseline.")
            return
        self.runs.set_baseline(self.active_run_id)
        self._log(f"{self.active_run_id} is now the delta baseline.")
        self._refresh_all()

    def _delete_active_run(self) -> None:
        if self.active_run_id is None:
            self._log("Select a run before deleting it.")
            return
        removed = self.runs.delete(self.active_run_id)
        self.checked_run_ids = [
            item for item in self.checked_run_ids if self.runs.get(item) is not None
        ]
        self._log(f"Deleted {removed} run record(s).")
        remaining = self.runs.newest_first()
        self.active_run_id = remaining[0].run_id if remaining else None
        if self.active_run_id:
            self._select_run(self.active_run_id)
        else:
            self.current_case = None
            self._clear_results()

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        self._update_derived_displays()
        self._refresh_breadcrumb()
        self._refresh_status()
        self._refresh_metric_strip()
        self._refresh_warning_bar()
        self._refresh_rail()
        self._refresh_inspector_head()
        self._refresh_mpi_strip()
        self._refresh_log_strip()
        self._refresh_propulsion_display()

    def _refresh_result_views(self) -> None:
        if self.current_case is None:
            return
        config = normalize_config(self._active_inputs())
        self._update_summary_table(self.current_case, config)
        self._update_load_table(self.current_case)
        self._update_polar_table()
        self._refresh_metric_strip()
        self._refresh_warning_bar()

    def _active_inputs(self) -> dict[str, Any]:
        record = self.runs.get(self.active_run_id)
        if record is not None:
            return record.inputs
        return self._current_config()

    def _refresh_inspector_head(self) -> None:
        edited = getattr(self, "_edited_count", 0)
        meta = f"{edited} edited" if edited else "no edits"
        if self.inspector_tab == "XFOIL":
            provider = self._xfoil_provider
            bins = provider.cache_bin_count() if provider is not None else 0
            meta = f"{meta} &middot; cache {bins}"
        self.inspector_head.object = (
            f'<div class="pycopter-section-head"><span>Inspector &mdash; '
            f'{self.inspector_tab}</span><span class="meta">{meta}</span></div>'
        )

    def _refresh_breadcrumb(self) -> None:
        record = self.runs.get(self.active_run_id)
        run_name = f"{record.run_id} {record.label}" if record else "no run"
        config = self._current_config()
        tags = []
        if config["rotor_system_type"] == "coaxial":
            tags.append("COAXIAL")
            tags.append(str(config["coaxial_trim_mode"]).replace("_", " ").upper())
        else:
            tags.append(str(config["trim_mode"]).replace("_", " ").upper())
        tag_html = "".join(f'<span class="pycopter-tag">{tag}</span>' for tag in tags)
        self.breadcrumb.object = (
            f'<div class="pycopter-crumb">{self.config_filename} / '
            f'<span class="active">{run_name}</span>{tag_html}</div>'
        )
        is_coaxial = config["rotor_system_type"] == "coaxial"
        self.showing_toggle.visible = is_coaxial
        self.showing_label.visible = is_coaxial

    def _refresh_status(self) -> None:
        if self._active_job is not None:
            state, text = "running", "solver running"
        elif self._xfoil_state.get("state") == "failed":
            state, text = "failed", "solver failed"
        else:
            state, text = "idle", "solver idle"
        provider = self._xfoil_provider
        bins = provider.cache_bin_count() if provider is not None else 0
        files = provider.cache_file_count() if provider is not None else 0
        workers = int(self.xfoil_workers.value)
        backend = str(self.xfoil_backend.value).upper()
        self.status_pane.object = (
            f'<div class="pycopter-status"><span class="dot {state}"></span>{text}'
            f" &middot; XFOIL cache {bins} bins / {files} files"
            f" &middot; {backend} {workers} workers</div>"
        )

    def _refresh_mpi_strip(self) -> None:
        workers = int(self.xfoil_workers.value)
        state = self._xfoil_state.get("state", "idle")
        active = int(self._xfoil_state.get("workers", 0))
        cells = []
        for index in range(1, workers + 1):
            if state == "running" and index <= active:
                cell_state = "running"
            elif state in ("ok", "failed") and index <= max(active, 1):
                cell_state = state
            else:
                cell_state = "idle"
            cells.append(f'<div class="{cell_state}">{index}</div>')
        provider = self._xfoil_provider
        bins = provider.cache_bin_count() if provider is not None else 0
        airfoils = 0
        if provider is not None:
            airfoils = len({record["airfoil"] for record in provider.polar_bin_report()})
        queued = int(self._xfoil_state.get("jobs", 0))
        error = self._xfoil_state.get("error", "")
        note = f" &middot; last error: {error[:80]}" if error else ""
        self.mpi_strip.object = (
            f'<div class="pycopter-mpi">{"".join(cells)}</div>'
            f'<div class="pycopter-mpi-meta">{str(self.xfoil_backend.value)} &middot; '
            f"{workers} max &middot; polar cache {bins} bins &middot; {airfoils} airfoils "
            f"&middot; {queued} queued{note}</div>"
        )

    def _refresh_metric_strip(self) -> None:
        record = self.runs.get(self.active_run_id)
        baseline = self.runs.baseline
        values = self._metric_values(record)
        base_values = self._metric_values(baseline) if baseline is not record else {}
        specs = COAXIAL_METRICS if (record and record.is_coaxial) else SINGLE_ROTOR_METRICS

        cells = []
        for key, label, unit, goodness in specs:
            if key == "ct_cp":
                text = (
                    f"{values['ct']:.5f} / {values['cp']:.5f}" if "ct" in values else "-"
                )
                cells.append(self._metric_cell(label, text, "", ""))
                continue
            if key == "endurance":
                text = values.get("endurance_text", "-")
                unit = values.get("endurance_unit", "")
            value = values.get(key)
            text = "-" if value is None else self._format_metric(value)
            if key == "endurance":
                text = values.get("endurance_text", "-")
            delta_html = self._delta_html(key, values.get(key), base_values.get(key), goodness)
            cells.append(self._metric_cell(label, text, unit, delta_html))
        self.metric_strip.object = (
            f'<div class="pycopter-metrics">{"".join(cells)}</div>'
        )

    def _metric_cell(self, label: str, value: str, unit: str, delta_html: str) -> str:
        unit_html = f'<span class="unit">{unit}</span>' if unit else ""
        return (
            f'<div class="cell"><span class="label">{label}</span>'
            f'<span class="value">{value}</span>{unit_html}{delta_html}</div>'
        )

    def _delta_html(self, key, value, base, goodness) -> str:
        if value is None or base is None or goodness == "none":
            return ""
        if goodness == "zero":
            delta = abs(value) - abs(base)
            good = delta < 0
        else:
            delta = value - base
            good = delta > 0 if goodness == "up" else delta < 0
        if abs(base) > 1e-12:
            text = f"{delta / abs(base) * 100.0:+.1f}%"
        else:
            text = f"{delta:+.3g}"
        if abs(delta) <= 1e-9:
            return '<span class="delta flat">&plusmn;0</span>'
        css = "good" if good else "bad"
        return f'<span class="delta {css}">{text}</span>'

    def _format_metric(self, value: float) -> str:
        magnitude = abs(value)
        if magnitude >= 1000.0:
            return f"{value:,.0f}"
        if magnitude >= 100.0:
            return f"{value:.1f}"
        if magnitude >= 1.0:
            return f"{value:.3f}"
        return f"{value:.4f}"

    def _metric_values(self, record: RunRecord | None) -> dict[str, Any]:
        if record is None or record.result is None:
            return {}
        config = normalize_config(record.inputs)
        result = record.result
        if isinstance(result, CoaxialHoverResult):
            active = result.lower if self.showing == "lower" else result.upper
            values: dict[str, Any] = {
                "total_thrust_N": result.total_thrust_N,
                "total_power_W": result.total_power_W,
                "interference_loss_ratio": result.interference_loss_ratio * 100.0,
                "active_collective_deg": active.collective_pitch_deg,
                "net_aircraft_yaw_torque_Nm": result.net_aircraft_yaw_torque_Nm,
                "active_figure_of_merit": active.figure_of_merit,
                "wake_velocity_m_s": result.wake_velocity_m_s,
                "wake_radius_m": result.wake_radius_m,
            }
            if self.showing == "delta":
                # What the coaxial interaction cost against the isolated pair.
                isolated = result.isolated_upper, result.isolated_lower
                values["total_thrust_N"] = result.total_thrust_N - sum(
                    rotor.total_thrust_N for rotor in isolated
                )
                values["total_power_W"] = result.interference_power_delta_W
                values["active_collective_deg"] = (
                    result.upper.collective_pitch_deg
                    - result.isolated_upper.collective_pitch_deg
                )
                values["net_aircraft_yaw_torque_Nm"] = (
                    result.net_aircraft_yaw_torque_Nm
                    - sum(rotor.aircraft_yaw_torque_Nm for rotor in isolated)
                )
                values["active_figure_of_merit"] = (
                    result.upper.figure_of_merit - result.isolated_upper.figure_of_merit
                )
            power_W = result.total_power_W
        else:
            values = {
                "total_thrust_N": result.total_thrust_N,
                "power_W": result.power_W,
                "collective_pitch_deg": result.collective_pitch_deg,
                "figure_of_merit": result.figure_of_merit,
                "ct": result.ct,
                "cp": result.cp,
                "mean_loss_factor": result.mean_loss_factor,
                "aircraft_yaw_torque_Nm": result.aircraft_yaw_torque_Nm,
            }
            power_W = result.power_W

        try:
            if config["propulsion_model"] == "electric":
                endurance = electric_summary(power_W, config)["hover_endurance_min"]
                values["endurance"] = endurance
                values["endurance_text"] = f"{endurance:.1f}"
                values["endurance_unit"] = "min"
            else:
                summary = propulsion_summary(
                    HoverCase(rotor=None, result=result, system_type="single"), config
                )
                endurance = summary["hover_endurance_hr"]
                values["endurance"] = endurance
                values["endurance_text"] = f"{endurance:.2f}"
                values["endurance_unit"] = "hr"
        except Exception:
            values["endurance_text"] = "-"
            values["endurance_unit"] = ""
        return values

    def _refresh_propulsion_display(self) -> None:
        """Only the active propulsion model's outputs are ever shown."""
        electric = self._propulsion_model() == "electric"
        if hasattr(self, "electric_fields"):
            self.electric_fields.visible = electric
            self.fossil_fields.visible = not electric
        record = self.runs.get(self.active_run_id)
        if record is None or record.result is None:
            self.propulsion_display.object = (
                '<div class="pycopter-menu-note">Solve a run to see '
                f'{"electric" if electric else "fossil fuel"} estimates.</div>'
            )
            return
        config = normalize_config({**record.inputs, "propulsion_model": self._propulsion_model()})
        case = HoverCase(
            rotor=None,
            result=record.result,
            system_type="coaxial" if record.is_coaxial else "single",
        )
        try:
            summary = propulsion_summary(case, config)
        except Exception as err:
            self.propulsion_display.object = f'<div class="pycopter-menu-note">{err}</div>'
            return
        if electric:
            rows = [
                ("Electric input", f"{summary['electric_input_W'] / 1000.0:.3f} kW"),
                ("Usable battery", f"{summary['usable_energy_Wh']:.1f} Wh"),
                ("Hover endurance", f"{summary['hover_endurance_min']:.2f} min"),
            ]
        else:
            rows = [
                ("Engine input", f"{summary['engine_input_W'] / 1000.0:.3f} kW"),
                ("Fuel flow", f"{summary['fuel_flow_kg_hr']:.3f} kg/hr"),
                ("Hover endurance", f"{summary['hover_endurance_hr']:.3f} hr"),
            ]
        body = "".join(
            f'<div style="display:flex;justify-content:space-between;font-size:11px;'
            f'color:{THEME["muted"]};padding:1px 0"><span>{label}</span>'
            f'<span style="color:{THEME["text"]}">{value}</span></div>'
            for label, value in rows
        )
        self.propulsion_display.object = body

    def _refresh_warning_bar(self) -> None:
        record = self.runs.get(self.active_run_id)
        warnings = record.warnings if record else ()
        if not warnings:
            self.warning_bar.visible = False
            return
        self.warning_bar.visible = True
        self.warning_text.object = (
            f'<div class="pycopter-warnbar-text">{len(warnings)} warning(s): '
            f'{" &middot; ".join(warnings)}</div>'
        )
        clamped = 0
        if record is not None and record.result is not None:
            clamped = record.result.clamped_element_count
        self.show_clamped_btn.visible = clamped > 0

    def _show_clamped_rows(self) -> None:
        self.loads_clamped_only.value = True
        self.result_tabs.active = 2

    def _refresh_rail(self) -> None:
        baseline = self.runs.baseline
        base_power = baseline.total_power_W if baseline else None
        rows = []
        for record in self.runs.newest_first():
            if record.parent_id and record.parent_id in self.collapsed_parents:
                continue
            rows.append(self._rail_row(record, baseline, base_power))
        self.run_rail.rows = rows
        baseline_text = baseline.run_id if baseline else "none"
        self.run_rail_head.object = (
            '<div class="pycopter-runrail-head"><span>Runs &mdash; session</span>'
            f'<span class="meta">{len(self.runs)} &middot; baseline {baseline_text}</span></div>'
        )

    def _rail_row(self, record, baseline, base_power) -> dict[str, Any]:
        is_baseline = baseline is not None and record.run_id == baseline.run_id
        dot = "baseline" if is_baseline else record.status
        power = record.total_power_W
        fom = record.figure_of_merit
        delta, delta_class = "—", "flat"
        if is_baseline:
            delta = "baseline"
        elif power is not None and base_power:
            change = (power - base_power) / abs(base_power) * 100.0
            delta = f"{change:+.1f}%"
            delta_class = "bad" if change > 0.05 else ("good" if change < -0.05 else "flat")

        meta_parts = [record.system_type]
        if record.status == "running" and record.progress:
            done, total = record.progress
            eta = record.eta_seconds
            eta_text = f" · est {eta:.0f}s" if eta else ""
            meta_parts = [f"point {done} / {total}{eta_text}"]
        elif record.status == "queued":
            meta_parts.append("queued")
        elif record.status == "failed":
            meta_parts.append("failed")
        elif record.status == "cancelled":
            meta_parts.append("cancelled")
        if record.warnings:
            meta_parts.append(f"{len(record.warnings)} warnings")
        if record.parent_id:
            meta_parts.append("sweep child")

        progress_pct = None
        if record.status == "running" and record.progress and record.progress[1]:
            progress_pct = round(record.progress[0] / record.progress[1] * 100.0, 1)

        children = self.runs.children_of(record.run_id)
        return {
            "run_id": record.run_id,
            "label": record.label,
            "meta": " · ".join(part for part in meta_parts if part),
            "status": record.status,
            "dot": dot,
            "current": record.run_id == self.active_run_id,
            "checked": record.run_id in self.checked_run_ids,
            "child": bool(record.parent_id),
            "has_children": bool(children),
            "collapsed": record.run_id in self.collapsed_parents,
            "power": "—" if power is None else f"{power:.1f}",
            "fom": "—" if fom is None else f"{fom:.3f}",
            "delta": delta,
            "delta_class": delta_class,
            "progress_pct": progress_pct,
            "error": record.error or "",
        }

    # ------------------------------------------------------------------
    # Result tables
    # ------------------------------------------------------------------

    def _update_summary_table(self, case: HoverCase, config: dict[str, Any]) -> None:
        config = normalize_config(config)
        rows: list[dict[str, str]] = [
            {"Metric": "Rotor System", "Value": "Coaxial" if case.system_type == "coaxial" else "Single rotor"},
            {"Metric": "Showing", "Value": self._showing_label()},
            {"Metric": "Total Thrust [N]", "Value": f"{total_hover_thrust_N(case):.3f}"},
            {"Metric": "Shaft Power [kW]", "Value": f"{total_hover_power_W(case) / 1000.0:.3f}"},
        ]

        result = case.result
        if isinstance(result, CoaxialHoverResult):
            upper_rpm = float(config["headspeed_rpm"])
            lower_rpm = upper_rpm * float(config["lower_rotor_speed_ratio"])
            rows.extend(
                [
                    {"Metric": "Upper Headspeed [rpm]", "Value": f"{upper_rpm:.1f}"},
                    {"Metric": "Lower Headspeed [rpm]", "Value": f"{lower_rpm:.1f}"},
                    {"Metric": "Lower/Upper RPM Ratio", "Value": f"{float(config['lower_rotor_speed_ratio']):.3f}"},
                    {"Metric": "Upper Power [kW]", "Value": f"{result.upper.power_W / 1000.0:.3f}"},
                    {"Metric": "Lower Power [kW]", "Value": f"{result.lower.power_W / 1000.0:.3f}"},
                    {"Metric": "Upper Thrust [N]", "Value": f"{result.upper.total_thrust_N:.3f}"},
                    {"Metric": "Lower Thrust [N]", "Value": f"{result.lower.total_thrust_N:.3f}"},
                    {"Metric": "Upper Thrust Share", "Value": f"{result.upper.total_thrust_N / max(result.total_thrust_N, 1e-9):.4f}"},
                    {"Metric": "Lower Thrust Share", "Value": f"{result.lower.total_thrust_N / max(result.total_thrust_N, 1e-9):.4f}"},
                    {"Metric": "Net Aircraft Yaw Torque [Nm]", "Value": f"{result.net_aircraft_yaw_torque_Nm:+.4f}"},
                    {"Metric": "Net Aircraft Yaw Direction", "Value": self._yaw_direction(result.net_aircraft_yaw_torque_Nm)},
                    {"Metric": "Interference Delta [W]", "Value": f"{result.interference_power_delta_W:.3f}"},
                    {"Metric": "Interference Loss", "Value": f"{result.interference_loss_ratio:.4f}"},
                    {"Metric": "Wake Radius [m]", "Value": f"{result.wake_radius_m:.4f}"},
                    {"Metric": "Wake Velocity [m/s]", "Value": f"{result.wake_velocity_m_s:.4f}"},
                ]
            )

        for label, hover in self._iter_hover_results(case):
            rows.extend(
                [
                    {"Metric": f"{label} Required Collective [deg]", "Value": f"{hover.collective_pitch_deg:.3f}"},
                    {"Metric": f"{label} Torque [Nm]", "Value": f"{hover.total_torque_Nm:.4f}"},
                    {"Metric": f"{label} Aircraft Yaw Torque [Nm]", "Value": f"{hover.aircraft_yaw_torque_Nm:+.4f}"},
                    {"Metric": f"{label} Induced Power [kW]", "Value": f"{hover.induced_power_W / 1000.0:.4f}"},
                    {"Metric": f"{label} Profile Power [kW]", "Value": f"{hover.profile_power_W / 1000.0:.4f}"},
                    {"Metric": f"{label} Ideal Power [kW]", "Value": f"{hover.ideal_power_W / 1000.0:.4f}"},
                    {"Metric": f"{label} Mean Induced Velocity [m/s]", "Value": f"{hover.mean_induced_velocity_m_s:.4f}"},
                    {"Metric": f"{label} Figure of Merit", "Value": f"{hover.figure_of_merit:.4f}"},
                    {"Metric": f"{label} Ct", "Value": f"{hover.ct:.6f}"},
                    {"Metric": f"{label} Cp", "Value": f"{hover.cp:.6f}"},
                    {"Metric": f"{label} Solidity", "Value": f"{hover.solidity:.5f}"},
                    {"Metric": f"{label} Mean Loss Factor", "Value": f"{hover.mean_loss_factor:.5f}"},
                    {"Metric": f"{label} Per-Blade Thrust [N]", "Value": f"{hover.per_blade_thrust_N:.4f}"},
                    {"Metric": f"{label} Per-Blade Torque [Nm]", "Value": f"{hover.per_blade_torque_Nm:.4f}"},
                    {"Metric": f"{label} Root Flap Moment [Nm/blade]", "Value": f"{hover.root_flap_bending_moment_Nm_per_blade:.4f}"},
                    {"Metric": f"{label} Root Lag Moment [Nm/blade]", "Value": f"{hover.root_lag_moment_Nm_per_blade:.4f}"},
                    {"Metric": f"{label} Pitch Moment [Nm/blade]", "Value": f"{hover.aerodynamic_pitching_moment_Nm_per_blade:.4f}"},
                    {"Metric": f"{label} Clamped Elements", "Value": f"{hover.clamped_element_count}"},
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
        self._set_read_only_table_value(self.summary_table, pd.DataFrame(rows))

    def _showing_label(self) -> str:
        for label, value in SHOWING_OPTIONS.items():
            if value == self.showing:
                return label
        return "TOTAL"

    def _yaw_direction(self, yaw_torque_Nm: float) -> str:
        if abs(yaw_torque_Nm) < 1e-9:
            return "balanced"
        if yaw_torque_Nm > 0.0:
            return "CCW / left yaw (+)"
        return "CW / right yaw (-)"

    def _update_load_table(self, case: HoverCase) -> None:
        frames = []
        for label, hover in self._iter_hover_results(case):
            frame = pd.DataFrame(load_rows_for_result(hover))
            if frame.empty:
                continue
            frame.insert(0, "rotor", label.lower())
            frames.append(frame)
        frame = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        if not frame.empty:
            frame["CLAMPED"] = np.where(frame["alpha_clamped"], "CLAMPED", "")
            # A frozen r/R column reads as the row key while the table scrolls.
            columns = ["r_over_R"] + [name for name in frame.columns if name != "r_over_R"]
            frame = frame[columns]
        self._loads_frame = frame
        self._refresh_load_table()

    def _refresh_load_table(self) -> None:
        frame = getattr(self, "_loads_frame", pd.DataFrame())
        if frame.empty:
            self._set_read_only_table_value(self.load_table, pd.DataFrame())
            self.loads_footer.object = ""
            return
        shown = frame
        if self.loads_clamped_only.value:
            shown = frame[frame["alpha_clamped"]].reset_index(drop=True)
        self._set_read_only_table_value(self.load_table, shown)
        self.load_table.groups = self._load_column_groups(shown)
        self._apply_load_shading(shown)
        self._refresh_loads_footer(frame, shown)

    def _load_column_groups(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name, members in LOAD_COLUMN_GROUPS:
            present = [column for column in members if column in frame.columns]
            if present:
                groups[name] = present
        return groups

    def _apply_load_shading(self, frame: pd.DataFrame) -> None:
        """Shade the selected column by how close it is to the polar limit."""
        mode = LOADS_SHADING_OPTIONS[str(self.loads_shading.value)]
        try:
            # Mutating the styler in place does not tell an already-rendered
            # model that its cell styles changed, so every path re-pushes them.
            self.load_table.style.clear()
            self.load_table._update_style()
        except Exception:
            pass
        if mode == "off" or frame.empty:
            return
        alpha_max = float(self._active_inputs().get("polar_alpha_max_deg", 18.0))
        column = "alpha_deg" if mode == "alpha" else "cl"

        def shade(data: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            if column not in data.columns:
                return styles
            if mode == "alpha":
                values = data[column].astype(float)
                colors = np.where(
                    data.get("alpha_clamped", False),
                    f"background-color: rgba(180, 87, 77, .32)",
                    np.where(
                        values >= alpha_max - 2.0,
                        "background-color: rgba(201, 151, 63, .28)",
                        "background-color: rgba(95, 158, 106, .18)",
                    ),
                )
            else:
                ratio = data["cl"].astype(float) / data["cd"].astype(float).replace(0.0, np.nan)
                normalized = (ratio - ratio.min()) / max(float(ratio.max() - ratio.min()), 1e-9)
                colors = [
                    f"background-color: rgba(89, 128, 166, {0.08 + 0.34 * float(value):.3f})"
                    if np.isfinite(value)
                    else ""
                    for value in normalized.fillna(0.0)
                ]
            styles[column] = colors
            if "alpha_clamped" in data.columns:
                clamped = data["alpha_clamped"].astype(bool)
                for name in data.columns:
                    styles.loc[clamped, name] = (
                        styles.loc[clamped, name] + "; background-color: rgba(180, 87, 77, .14)"
                        if name != column
                        else styles.loc[clamped, name]
                    )
            return styles

        try:
            self.load_table.style.apply(shade, axis=None)
            self.load_table._update_style()
        except Exception as err:
            self._log(f"WARNING - Load table shading is unavailable: {err}")

    def _refresh_loads_footer(self, frame: pd.DataFrame, shown: pd.DataFrame) -> None:
        clamped = int(frame["alpha_clamped"].sum()) if "alpha_clamped" in frame else 0
        alpha_max = float(self._active_inputs().get("polar_alpha_max_deg", 18.0))
        near = 0
        if "alpha_deg" in frame:
            near = int(((frame["alpha_deg"] >= alpha_max - 2.0) & ~frame["alpha_clamped"]).sum())
        css = "bad" if clamped else "warn" if near else ""
        self.loads_footer.object = (
            f'<div>{len(shown)} of {len(frame)} rows shown &middot; '
            f'<span class="{css}">{clamped} clamped</span> &middot; '
            f"{near} within 2 deg of the {alpha_max:.1f} deg polar limit</div>"
        )

    def _update_polar_table(self) -> None:
        provider = self._xfoil_provider
        report = provider.polar_bin_report() if provider is not None else []
        if not report:
            self._set_read_only_table_value(
                self.polar_table, pd.DataFrame(columns=["airfoil", "reynolds", "mach", "status"])
            )
            self.polar_footer.object = (
                '<div>No XFOIL polar bins in this session. Analytic providers are not binned.</div>'
            )
            return
        frame = pd.DataFrame(report)[
            ["airfoil", "reynolds", "mach", "alpha_min_deg", "alpha_max_deg",
             "status", "lookups", "cache_file"]
        ]
        self._set_read_only_table_value(self.polar_table, frame)
        generated = int((frame["status"] == "generated").sum())
        cached = int((frame["status"] == "cache").sum())
        substituted = int((frame["status"] == "substituted").sum())
        css = "bad" if substituted else ""
        self.polar_footer.object = (
            f"<div>{len(frame)} bins &middot; {cached} cache hits &middot; "
            f'{generated} generated &middot; <span class="{css}">{substituted} substituted'
            "</span></div>"
        )

    def _set_read_only_table_value(self, table: pn.widgets.Tabulator, frame: pd.DataFrame) -> None:
        table.value = frame
        table.editors = {column: None for column in frame.columns}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_hover_result(self, case: HoverCase, config: dict[str, Any]) -> None:
        config = normalize_config(config)
        result = case.result
        if isinstance(result, CoaxialHoverResult):
            upper_rpm = float(config["headspeed_rpm"])
            lower_rpm = upper_rpm * float(config["lower_rotor_speed_ratio"])
            self._log(
                f"SOLVER Total Thrust: {result.total_thrust_N:.3f} [N] | "
                f"Total Shaft Power: {result.total_power_W / 1000.0:.3f} [kW] | "
                f"Interference Loss: {result.interference_loss_ratio:.3f}"
            )
            self._log(
                f"SOLVER Upper RPM: {upper_rpm:.1f} | Lower RPM: {lower_rpm:.1f} | "
                f"Upper Collective: {result.upper.collective_pitch_deg:.3f} [deg] | "
                f"Lower Collective: {result.lower.collective_pitch_deg:.3f} [deg]"
            )
            self._log(
                f"SOLVER Upper Thrust: {result.upper.total_thrust_N:.3f} [N] | "
                f"Lower Thrust: {result.lower.total_thrust_N:.3f} [N] | "
                f"Wake Velocity: {result.wake_velocity_m_s:.3f} [m/s] | "
                f"Wake Radius: {result.wake_radius_m:.4f} [m]"
            )
            self._log(
                "SOLVER Aircraft Yaw Torque: "
                f"{result.net_aircraft_yaw_torque_Nm:+.4f} [Nm] "
                f"({self._yaw_direction(result.net_aircraft_yaw_torque_Nm)})"
            )
        else:
            self._log(
                f"SOLVER Required Collective: {result.collective_pitch_deg:.3f} [deg] | "
                f"Induced Velocity: {result.mean_induced_velocity_m_s:.3f} [m/s] | "
                f"Thrust: {result.total_thrust_N:.3f} [N]"
            )
            self._log(
                f"SOLVER Shaft Power: {result.power_W / 1000.0:.3f} [kW] | "
                f"Induced: {result.induced_power_W / 1000.0:.3f} [kW] | "
                f"Profile: {result.profile_power_W / 1000.0:.3f} [kW]"
            )
            self._log(
                f"SOLVER Ct: {result.ct:.5f} | Cp: {result.cp:.5f} | "
                f"Figure of Merit: {result.figure_of_merit:.3f} | Mean Loss: {result.mean_loss_factor:.3f}"
            )
            self._log(
                "SOLVER Aircraft Yaw Torque: "
                f"{result.aircraft_yaw_torque_Nm:+.4f} [Nm] "
                f"({self._yaw_direction(result.aircraft_yaw_torque_Nm)})"
            )
        for warning in result.warnings:
            self._log(f"WARNING - {warning}.")
        if result.clamped_element_count:
            self._log(
                "WARNING - Expand XFOIL Alpha Min/Max and keep polar generation enabled "
                "if you want real data instead of clamped table edges."
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

    def _calculate_forward_flight(self) -> None:
        """Legacy single-rotor forward-flight estimate for the active run."""
        if self.current_case is None:
            self._log("Solve a run before estimating forward flight.")
            return
        if self.current_case.system_type != "single":
            self._log("Legacy forward-flight estimates are available for single-rotor cases only.")
            return
        try:
            config = normalize_config(self._active_inputs())
            hover = primary_hover_result(self.current_case)
            estimate = estimate_forward_flight(
                self.current_case.rotor,
                hover,
                velocity_m_s=float(config["velocity_kmh"]) / 3.6,
                density_kg_m3=float(config["density"]),
                flat_plate_area_m2=float(config["fpa"]),
            )
            self._log(
                f"Forward flight at {config['velocity_kmh']:.2f} [km/hr] | "
                f"Downwash Velocity Ratio: {estimate.downwash_velocity_ratio:.3f} | "
                f"Total Power: {estimate.total_power_W / 1000.0:.3f} [kW]"
            )
        except Exception as err:
            self._log(f"ERROR - {err}")

    def _log_sweep_summary(self, sweep) -> None:
        losses = [point.result.interference_loss_ratio for point in sweep.solved_points]
        powers = [point.result.total_power_W / 1000.0 for point in sweep.solved_points]
        yaws = [abs(point.result.net_aircraft_yaw_torque_Nm) for point in sweep.solved_points]
        self._log(
            "Spacing sweep complete: "
            f"{len(sweep.solved_points)}/{sweep.requested_points} points converged; "
            f"interference loss range {min(losses):.3f}..{max(losses):.3f}; "
            f"power range {min(powers):.3f}..{max(powers):.3f} [kW]; "
            f"max |yaw torque| {max(yaws):.4f} [Nm]"
            + (" (cancelled)" if sweep.cancelled else "")
            + "."
        )

    def _toggle_log(self) -> None:
        expanded = self.log_region.height > LOG_MIN_HEIGHT
        height = LOG_MIN_HEIGHT if expanded else LOG_EXPANDED_HEIGHT
        self.log_splitter.height_px = height
        self._on_log_resized()

    def _on_log_filter_changed(self) -> None:
        self._log_filter = str(self.log_filter.value)
        self._rewrite_log()

    def _on_log_resized(self) -> None:
        """Commit a dragged log height so the terminal re-flows its rows."""
        height = max(LOG_MIN_HEIGHT, int(self.log_splitter.height_px))
        inner = max(20, height - LOG_STRIP_HEIGHT - LOG_TERMINAL_INSET)
        self.log_region.height = height
        self.log_body.height = inner
        self.output_log.height = inner
        self.log_toggle_btn.label = (
            "COLLAPSE ˅" if height > LOG_MIN_HEIGHT else "EXPAND ˄"
        )

    def _refresh_log_strip(self) -> None:
        latest = self.output_lines[-1] if self.output_lines else "ready"
        counts = self._log_counts
        self.log_line.object = (
            f'<div class="pycopter-log-line"><span class="tag">LOG</span>{_escape(latest)}</div>'
        )
        self.log_counts.object = (
            '<div class="pycopter-log-counts">'
            f'<span class="warn">{counts["WARN"]} warnings</span> &middot; '
            f'<span class="bad">{counts["ERROR"]} errors</span> &middot; '
            '<span class="hint">drag rule to resize</span></div>'
        )
        # One count per severity button, aligned to the 20 px filter rows.
        totals = {"ALL": len(self.output_lines), **counts}
        rows = "".join(
            f'<div class="pycopter-log-count {severity.lower()}">{totals[severity]}</div>'
            for severity in LOG_SEVERITIES
        )
        self.log_filter_counts.object = f'<div class="pycopter-log-counts-col">{rows}</div>'

    def _log(self, text: str) -> None:
        prefix = datetime.now().strftime("%H:%M:%S")
        line = "" if text == "" else f"[{prefix}] {text}"
        self.output_lines.append(line)
        if "WARNING" in text:
            self._log_counts["WARN"] += 1
        if "ERROR" in text:
            self._log_counts["ERROR"] += 1
        if text.startswith("SOLVER"):
            self._log_counts["SOLVER"] += 1
        limit = int(self.log_scrollback.value) if hasattr(self, "log_scrollback") else 1000
        if len(self.output_lines) > limit:
            self.output_lines = self.output_lines[-limit:]
            self._rewrite_log()
        elif self._line_passes_filter(line):
            self.output_log.write(("\n" if text == "" else f"{line}\n"))
        if hasattr(self, "log_line"):
            self._refresh_log_strip()

    def _line_passes_filter(self, line: str) -> bool:
        if self._log_filter == "ALL":
            return True
        if self._log_filter == "WARN":
            return "WARNING" in line
        if self._log_filter == "ERROR":
            return "ERROR" in line
        return "SOLVER" in line

    def _rewrite_log(self) -> None:
        self.output_log.clear()
        kept = [line for line in self.output_lines if self._line_passes_filter(line)]
        if kept:
            self.output_log.write("\n".join(kept) + "\n")

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _generate_plot(self) -> None:
        if not self.plot_select.value:
            return
        plot_name = self.plot_select.value
        if (
            plot_name == "Interference Loss vs Spacing"
            and self._background_enabled
            and self._active_job is None
            and self.current_case is not None
        ):
            self._start_spacing_sweep()
            return
        try:
            if plot_name == "Blade Geometry":
                fig = self._plot_blade_geometry()
            elif self.current_case is None:
                self._log("Solve a run before generating this plot.")
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
            elif plot_name == "Stall Margin vs Radius":
                fig = self._plot_stall_margin(self.current_case)
            elif plot_name == "Geometry Load Contribution":
                fig = self._plot_geometry_load_contribution(self.current_case)
            elif plot_name == "Disk Loading Sensitivity":
                fig = self._plot_disk_loading_sensitivity(self.current_case)
            elif plot_name == "Rotor Diameter Sizing":
                fig = self._plot_rotor_diameter_sizing(self.current_case)
            elif plot_name in ("Reference RPM Sweep", "Headspeed RPM Sweep"):
                fig = self._plot_headspeed_sweep(self.current_case)
            elif plot_name == "Collective Authority Curve":
                fig = self._plot_collective_authority(self.current_case)
            elif plot_name == "Energy Capacity vs Endurance":
                fig = self._plot_energy_capacity_endurance(self.current_case)
            elif plot_name == "Efficiency and Loss Sensitivity":
                fig = self._plot_efficiency_loss_sensitivity(self.current_case)
            elif plot_name == "Payload Endurance Sweep":
                fig = self._plot_payload_endurance_sweep(self.current_case)
            elif plot_name == "Coaxial Interference":
                fig = self._plot_coaxial_interference(self.current_case)
            elif plot_name == "Interference Loss vs Spacing":
                fig = self._plot_interference_loss_vs_spacing(self.current_case)
            elif plot_name == "Coaxial Thrust Share and Yaw":
                fig = self._plot_coaxial_thrust_share_yaw(self.current_case)
            elif plot_name == "Electric Range vs Velocity":
                fig = self._plot_electric_range(self.current_case)
            elif plot_name == "Fuel Range, Endurance vs Velocity":
                fig = self._plot_fuel_range(self.current_case)
            elif plot_name == "Forward Flight Powers vs Velocity":
                fig = self._plot_forward_powers(self.current_case)
            else:
                fig = self._blank_figure("No plot selected.")
            self._install_figure(fig, plot_name)
        except Exception as err:
            self._log(f"ERROR - {err}")

    def _install_figure(self, fig, plot_name: str) -> None:
        superseded = self.plot_fig
        self.plot_fig = fig
        self.plot_pane.object = fig
        self._current_plot_name = plot_name
        self._plot_save_available = True
        if superseded is not fig:
            plt.close(superseded)
        self.save_plot_download.filename = self._plot_filename(plot_name)
        self.result_tabs.active = 0
        self._sync_enabled_state()

    def _start_spacing_sweep(self) -> None:
        """Run the coaxial spacing sweep off the event loop, as a sweep run."""
        config = normalize_config(self._active_inputs())
        if config["rotor_system_type"] != "coaxial":
            self._log("The spacing sweep needs a coaxial run.")
            return
        station_rows = config.get("station_rows") or self._station_rows()
        points = int(config["sweep_points"])
        spacings = np.linspace(COAXIAL_SPACING_SWEEP_MIN, COAXIAL_SPACING_SWEEP_MAX, points)
        record = self.runs.create(
            "interference sweep",
            {**config, "station_rows": station_rows},
            status="running",
            progress=(0, points),
        )
        self._log(
            f"{record.run_id} sweeping interference loss over {points} points, "
            f"z/R {COAXIAL_SPACING_SWEEP_MIN:.2f}..{COAXIAL_SPACING_SWEEP_MAX:.2f}."
        )
        self._refresh_rail()

        provider = self._xfoil_provider_for_config(config)
        parent_id = record.run_id
        result_queue = self._queue

        def work(cancel):
            sweep = self._run_interference_sweep(
                config,
                station_rows,
                provider,
                spacings,
                cancel=cancel,
                progress=lambda index, total, spacing, point: result_queue.put(
                    ("sweep_point", (parent_id, index, total, spacing, point))
                ),
            )
            return (sweep, config)

        self._dispatch_job(record.run_id, "sweep", work, None)

    def _run_interference_sweep(
        self,
        config: dict[str, Any],
        station_rows: list[dict[str, Any]],
        provider,
        spacings,
        *,
        cancel=None,
        progress=None,
    ):
        upper = build_rotor_spec(config, station_rows, name="upper")
        lower = build_rotor_spec(
            config,
            station_rows,
            name="lower",
            scale=float(config["lower_rotor_scale"]),
            headspeed_ratio=float(config["lower_rotor_speed_ratio"]),
            rotation_direction=-1,
        )
        return sweep_interference_loss(
            build_coaxial_spec(config, upper, lower),
            build_operating_point(config),
            spacings,
            polar_provider=provider,
            settings=build_solver_settings(config),
            progress=progress,
            cancel=cancel,
        )

    def _figure_size(self) -> tuple[float, float]:
        """Return the figure size in inches that matches the on-screen plot frame."""
        if self._plot_area_size_px is None:
            return PLOT_FIGSIZE

        width_px, height_px = self._plot_area_size_px
        width_in = min(max(width_px / PLOT_CSS_DPI, PLOT_MIN_SIZE_IN[0]), PLOT_MAX_SIZE_IN[0])
        height_in = min(max(height_px / PLOT_CSS_DPI, PLOT_MIN_SIZE_IN[1]), PLOT_MAX_SIZE_IN[1])
        return width_in, height_in

    def _on_plot_area_resized(self) -> None:
        """Track the browser-reported plot frame size and refresh cheap figures.

        Recomputing a sweep plot means re-running the hover solver many times,
        so only the placeholder figure is redrawn here. Real plots keep their
        current raster (CSS keeps them inside the frame) until the next
        Generate Plot or hover calculation redraws them at the new size.
        """
        width_px = int(self.plot_area_probe.width_px)
        height_px = int(self.plot_area_probe.height_px)
        if width_px <= 0 or height_px <= 0:
            return

        previous = self._plot_area_size_px
        self._plot_area_size_px = (width_px, height_px)
        if previous is not None and self._plot_save_available:
            return

        if not self._plot_save_available:
            superseded = self.plot_fig
            self.plot_fig = self._blank_figure("Solve a run to see results.")
            self.plot_pane.object = self.plot_fig
            plt.close(superseded)

    def _plot_filename(self, plot_name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "_" for char in plot_name).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return f"pycopter_{slug or 'plot'}.png"

    def _plot_blade_geometry(self):
        frame = self.station_table.value.copy()
        fig, ax1 = plt.subplots(figsize=self._figure_size())
        self._accent_axis(ax1, SERIES_COLORS[0])
        ax1.plot(frame["r_over_R"], frame["chord_m"], marker="o", color=SERIES_COLORS[0], label="Chord [m]")
        ax1.set_xlabel("r/R")
        ax1.set_ylabel("Chord [m]")
        ax1.grid(True)
        ax2 = self._twin_axis(ax1, SERIES_COLORS[1])
        ax2.plot(frame["r_over_R"], frame["twist_deg"], marker="s", color=SERIES_COLORS[1], label="Twist [deg]")
        ax2.set_ylabel("Twist [deg]")
        ax1.set_title("Blade Geometry")
        self._combined_legend(ax1, ax2)
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
        fig, ax = plt.subplots(figsize=self._figure_size())
        ax.bar(x, induced, label="Induced")
        ax.bar(x, profile, bottom=induced, label="Profile")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Shaft Power [kW]")
        ax.set_title("Hover Power Breakdown")
        ax.grid(True, axis="y")
        ax.legend()
        return self._finish_plot(fig)

    def _plot_radial_loads(self, case: HoverCase):
        # Element torque is roughly 3% of element thrust, so a shared y-axis
        # renders the torque curve as a flat line along zero.
        return self._plot_element_series(
            case,
            (
                ("dT_N", "Element Thrust [N/blade]", "dT [N/blade]"),
                ("dQ_Nm", "Element Torque [Nm/blade]", "dQ [Nm/blade]"),
            ),
            "Per-Blade Radial Loads",
        )

    def _plot_alpha_re_mach(self, case: HoverCase):
        # Mach is around 1e-6 of Reynolds, so these three quantities need three
        # separate axes rather than Reynolds and Mach sharing one.
        return self._plot_element_series(
            case,
            (
                ("alpha_deg", "Alpha [deg]", "alpha [deg]"),
                ("reynolds", "Reynolds [-]", "Re [-]"),
                ("mach", "Mach [-]", "Mach [-]"),
            ),
            "Section Flow Conditions",
        )

    def _plot_induced_loss(self, case: HoverCase):
        return self._plot_element_series(
            case,
            (
                ("induced_velocity_m_s", "Induced Velocity [m/s]", "vi [m/s]"),
                ("loss_factor", "Loss Factor [-]", "loss factor [-]"),
            ),
            "Induced Velocity and Root/Tip Loss",
        )

    def _plot_section_coefficients(self, case: HoverCase):
        # Cd and Cm are a few percent of Cl, so one shared coefficient axis
        # hides both of them along the zero line.
        return self._plot_element_series(
            case,
            (
                ("cl", "Cl [-]", "Cl [-]"),
                ("cd", "Cd [-]", "Cd [-]"),
                ("cm", "Cm [-]", "Cm [-]"),
            ),
            "Section Coefficients",
        )

    def _plot_pitch_moment(self, case: HoverCase):
        fig, ax = plt.subplots(figsize=self._figure_size())
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
        return self._plot_element_series(
            case,
            (
                (lambda rows: rows["dT_N"].cumsum(), "Cumulative Thrust [N/blade]", "thrust [N/blade]"),
                (lambda rows: rows["dP_W"].cumsum(), "Cumulative Power [W/blade]", "power [W/blade]"),
            ),
            "Cumulative Per-Blade Load Build-Up",
        )

    def _plot_stall_margin(self, case: HoverCase):
        cfg = normalize_config(self._active_inputs())
        alpha_min = float(cfg["polar_alpha_min_deg"])
        alpha_max = float(cfg["polar_alpha_max_deg"])
        fig, ax = plt.subplots(figsize=self._figure_size())
        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            lower_margin = rows["alpha_deg"] - alpha_min
            upper_margin = alpha_max - rows["alpha_deg"]
            nearest_margin = np.minimum(lower_margin, upper_margin)
            ax.plot(rows["r_over_R"], nearest_margin, marker="o", label=f"{label} nearest alpha limit")
            ax.fill_between(
                rows["r_over_R"],
                0.0,
                nearest_margin,
                where=nearest_margin < 0.0,
                alpha=0.18,
            )
        ax.axhline(0.0, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax.set_title("Alpha Envelope Margin")
        ax.set_xlabel("r/R")
        ax.set_ylabel("Alpha Margin to Polar Limit [deg]")
        ax.grid(True)
        ax.legend()
        return self._finish_plot(fig)

    def _plot_geometry_load_contribution(self, case: HoverCase):
        station_frame = self.station_table.value.copy()
        fig, (ax_geom, ax_loads) = plt.subplots(2, 1, figsize=self._figure_size(), sharex=False)
        self._accent_axis(ax_geom, SERIES_COLORS[0])
        ax_geom.plot(
            station_frame["r_over_R"],
            station_frame["chord_m"],
            marker="o",
            color=SERIES_COLORS[0],
            label="Chord [m]",
        )
        ax_geom.set_ylabel("Chord [m]")
        ax_geom.grid(True)
        ax_twist = self._twin_axis(ax_geom, SERIES_COLORS[1])
        ax_twist.plot(
            station_frame["r_over_R"],
            station_frame["twist_deg"],
            marker="s",
            color=SERIES_COLORS[1],
            label="Twist [deg]",
        )
        ax_twist.set_ylabel("Twist [deg]")
        ax_geom.set_title("Control-Point Geometry")
        self._combined_legend(ax_geom, ax_twist)

        for label, hover in self._iter_hover_results(case):
            rows = pd.DataFrame(load_rows_for_result(hover))
            thrust_sum = max(abs(float(rows["dT_N"].sum())), 1e-9)
            power_sum = max(abs(float(rows["dP_W"].sum())), 1e-9)
            ax_loads.plot(
                rows["r_over_R"],
                rows["dT_N"] / thrust_sum * 100.0,
                label=f"{label} thrust share",
            )
            ax_loads.plot(
                rows["r_over_R"],
                rows["dP_W"] / power_sum * 100.0,
                linestyle="--",
                label=f"{label} power share",
            )
        ax_loads.set_title("Element Contribution")
        ax_loads.set_xlabel("r/R")
        ax_loads.set_ylabel("Element Contribution [%/blade]")
        ax_loads.grid(True)
        ax_loads.legend()
        fig.suptitle("Geometry vs Load Contribution")
        fig.legend(loc="upper right")
        return self._finish_plot(fig)

    def _plot_disk_loading_sensitivity(self, case: HoverCase):
        base_config = self._current_config()
        station_rows = self._station_rows()
        provider = self._xfoil_provider_for_config(base_config)
        gross = float(base_config["gross"])
        gross_values = np.linspace(
            max(MIN_SWEEP_GROSS_MASS_KG, 0.5 * gross),
            max(MIN_SWEEP_GROSS_MASS_KG, 2.0 * gross),
            DESIGN_SWEEP_POINTS,
        )
        self._log(
            "Calculating disk loading sensitivity "
            f"from {gross_values[0]:.2f} to {gross_values[-1]:.2f} [kg] gross mass."
        )
        sweep = self._collect_hover_sweep(
            "Disk loading sensitivity",
            base_config,
            station_rows,
            provider,
            gross_values,
            lambda value: {"gross": float(value)},
        )

        disk_loading = []
        shaft_power = []
        input_power = []
        collectives = []
        for _, cfg, sweep_case in sweep:
            disk_loading.append(total_hover_thrust_N(sweep_case) / max(sweep_case.rotor.disk_area_m2, 1e-9))
            power = self._power_metrics(sweep_case, cfg)
            shaft_power.append(power["shaft_kW"])
            input_power.append(power["input_kW"])
            collectives.append(self._mean_collective_deg(sweep_case))

        current_disk_loading = total_hover_thrust_N(case) / max(case.rotor.disk_area_m2, 1e-9)
        fig, (ax_power, ax_collective) = plt.subplots(2, 1, figsize=self._figure_size(), sharex=True)
        ax_power.plot(disk_loading, shaft_power, marker="o", label="Shaft Power [kW]")
        ax_power.plot(disk_loading, input_power, marker="s", linestyle="--", label=self._input_power_label(base_config))
        ax_power.axvline(current_disk_loading, color=THEME["warning"], linestyle="--", linewidth=1.0, label="Current")
        ax_power.set_ylabel("Power [kW]")
        ax_power.set_title("Power vs Disk Loading")
        ax_power.grid(True)
        ax_power.legend()

        ax_collective.plot(disk_loading, collectives, marker="o", color="tab:red", label="Mean Collective [deg]")
        ax_collective.axvline(current_disk_loading, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_collective.set_xlabel("Disk Loading [N/m2]")
        ax_collective.set_ylabel("Mean Collective [deg]")
        ax_collective.grid(True)
        ax_collective.legend()
        fig.suptitle("Disk Loading Sensitivity")
        return self._finish_plot(fig)

    def _plot_rotor_diameter_sizing(self, case: HoverCase):
        base_config = self._current_config()
        station_rows = self._station_rows()
        provider = self._xfoil_provider_for_config(base_config)
        diameter = float(base_config["rotor_diam"])
        diameters = np.linspace(max(0.05, 0.6 * diameter), max(0.05, 1.6 * diameter), DESIGN_SWEEP_POINTS)
        self._log(
            "Calculating rotor diameter sizing sweep "
            f"from {diameters[0]:.2f} to {diameters[-1]:.2f} [m]."
        )
        sweep = self._collect_hover_sweep(
            "Rotor diameter sizing",
            base_config,
            station_rows,
            provider,
            diameters,
            lambda value: {"rotor_diam": float(value)},
        )

        diameters_valid = []
        shaft_power = []
        endurance = []
        tip_mach = []
        for value, cfg, sweep_case in sweep:
            diameters_valid.append(float(value))
            power = self._power_metrics(sweep_case, cfg)
            shaft_power.append(power["shaft_kW"])
            endurance.append(power["endurance"])
            tip_mach.append(self._max_load_value(sweep_case, "mach"))

        fig, (ax_power, ax_endurance) = plt.subplots(2, 1, figsize=self._figure_size(), sharex=True)
        ax_power.plot(diameters_valid, shaft_power, marker="o", label="Shaft Power [kW]")
        ax_power.axvline(diameter, color=THEME["warning"], linestyle="--", linewidth=1.0, label="Current Diameter")
        ax_power.set_ylabel("Shaft Power [kW]")
        ax_power.set_title("Hover Power vs Rotor Diameter")
        ax_power.grid(True)
        ax_power.legend()

        self._accent_axis(ax_endurance, SERIES_COLORS[0])
        ax_endurance.plot(
            diameters_valid,
            endurance,
            marker="s",
            color=SERIES_COLORS[0],
            label=self._endurance_label(base_config),
        )
        ax_mach = self._twin_axis(ax_endurance, SERIES_COLORS[1])
        ax_mach.plot(diameters_valid, tip_mach, linestyle="--", color=SERIES_COLORS[1], label="Max Mach")
        ax_endurance.axvline(diameter, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_endurance.set_xlabel("Rotor Diameter [m]")
        ax_endurance.set_ylabel(self._endurance_label(base_config))
        ax_mach.set_ylabel("Max Mach")
        ax_endurance.grid(True)
        self._combined_legend(ax_endurance, ax_mach)
        fig.suptitle("Rotor Diameter Sizing")
        return self._finish_plot(fig)

    def _plot_headspeed_sweep(self, case: HoverCase):
        base_config = self._current_config()
        station_rows = self._station_rows()
        provider = self._xfoil_provider_for_config(base_config)
        rpm = float(base_config["headspeed_rpm"])
        rpms = np.linspace(max(100.0, 0.6 * rpm), max(100.0, 1.4 * rpm), DESIGN_SWEEP_POINTS)
        self._log(f"Calculating reference RPM sweep from {rpms[0]:.0f} to {rpms[-1]:.0f} [rpm].")
        sweep = self._collect_hover_sweep(
            "Headspeed RPM sweep",
            base_config,
            station_rows,
            provider,
            rpms,
            lambda value: {"headspeed_rpm": float(value)},
        )

        rpms_valid = []
        shaft_power = []
        max_mach = []
        mean_re = []
        collective_by_label: dict[str, list[float]] = {}
        for value, _, sweep_case in sweep:
            rpms_valid.append(float(value))
            shaft_power.append(total_hover_power_W(sweep_case) / 1000.0)
            max_mach.append(self._max_load_value(sweep_case, "mach"))
            mean_re.append(self._mean_load_value(sweep_case, "reynolds"))
            for label, hover in self._iter_hover_results(sweep_case):
                collective_by_label.setdefault(label, []).append(hover.collective_pitch_deg)

        fig, axes = plt.subplots(2, 2, figsize=self._figure_size(), sharex=True)
        ax_power, ax_collective, ax_mach, ax_re = axes.flatten()
        ax_power.plot(rpms_valid, shaft_power, marker="o", label="Shaft Power [kW]")
        ax_power.axvline(rpm, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_power.set_ylabel("Shaft Power [kW]")
        ax_power.set_title("Power")
        ax_power.grid(True)
        ax_power.legend()

        for label, values in collective_by_label.items():
            ax_collective.plot(rpms_valid, values, marker="o", label=f"{label} Collective [deg]")
        ax_collective.axvline(rpm, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_collective.set_ylabel("Collective [deg]")
        ax_collective.set_title("Trim Collective")
        ax_collective.grid(True)
        ax_collective.legend()

        ax_mach.plot(rpms_valid, max_mach, marker="o", color="tab:red", label="Max Mach")
        ax_mach.axvline(rpm, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_mach.set_xlabel("Upper / Reference Headspeed [rpm]")
        ax_mach.set_ylabel("Max Mach")
        ax_mach.set_title("Compressibility Check")
        ax_mach.grid(True)
        ax_mach.legend()

        ax_re.plot(rpms_valid, mean_re, marker="o", color="tab:green", label="Mean Reynolds")
        ax_re.axvline(rpm, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_re.set_xlabel("Upper / Reference Headspeed [rpm]")
        ax_re.set_ylabel("Mean Reynolds")
        ax_re.set_title("Section Reynolds")
        ax_re.grid(True)
        ax_re.legend()
        fig.suptitle("Reference RPM Sweep")
        return self._finish_plot(fig)

    def _plot_collective_authority(self, case: HoverCase):
        base_config = self._current_config()
        station_rows = self._station_rows()
        provider = self._xfoil_provider_for_config(base_config)
        current_collectives = [hover.collective_pitch_deg for _, hover in self._iter_hover_results(case)]
        lower = min(float(base_config["min_collective_deg"]), min(current_collectives) - 4.0)
        upper = max(float(base_config["max_collective_deg"]), max(current_collectives) + 4.0)
        collectives = np.linspace(lower, upper, CONTROL_SWEEP_POINTS)
        lower_offset = float(base_config["lower_collective_offset_deg"]) if case.system_type == "coaxial" else 0.0
        self._log(f"Calculating collective authority curve from {lower:.1f} to {upper:.1f} [deg].")

        valid = []
        failures = 0
        for collective in collectives:
            try:
                fixed_case = self._solve_fixed_collective_case(
                    base_config,
                    station_rows,
                    provider,
                    float(collective),
                    float(collective + lower_offset),
                )
                valid.append((float(collective), fixed_case))
            except Exception:
                failures += 1
        if failures:
            self._log(f"WARNING - Collective authority curve skipped {failures} non-converged points.")
        if len(valid) < 2:
            raise ValueError("Collective authority curve could not compute enough valid points.")

        x = [item[0] for item in valid]
        thrust = [total_hover_thrust_N(item[1]) for item in valid]
        yaw_torque = [self._net_aircraft_yaw_torque(item[1]) for item in valid]
        shaft_power = [total_hover_power_W(item[1]) / 1000.0 for item in valid]

        fig, (ax_thrust, ax_torque) = plt.subplots(2, 1, figsize=self._figure_size(), sharex=True)
        self._accent_axis(ax_thrust, SERIES_COLORS[0])
        ax_thrust.plot(x, thrust, marker="o", color=SERIES_COLORS[0], label="Total Thrust [N]")
        ax_power = self._twin_axis(ax_thrust, SERIES_COLORS[1])
        ax_power.plot(x, shaft_power, marker="s", linestyle="--", color=SERIES_COLORS[1], label="Shaft Power [kW]")
        ax_thrust.axhline(total_hover_thrust_N(case), color=THEME["warning"], linestyle="--", linewidth=1.0, label="Current Thrust")
        ax_thrust.set_ylabel("Total Thrust [N]")
        ax_power.set_ylabel("Shaft Power [kW]")
        ax_thrust.set_title("Lift Authority")
        ax_thrust.grid(True)
        self._combined_legend(ax_thrust, ax_power, loc="upper left")

        ax_torque.plot(x, yaw_torque, marker="o", color="tab:red", label="Aircraft Yaw Torque [Nm]")
        ax_torque.axhline(0.0, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_torque.set_xlabel("Upper Collective [deg]")
        ax_torque.set_ylabel("Aircraft Yaw Torque [Nm]")
        ax_torque.set_title("Torque / Yaw Response")
        ax_torque.grid(True)
        ax_torque.legend()
        fig.suptitle("Collective Authority Curve")
        return self._finish_plot(fig)

    def _plot_energy_capacity_endurance(self, case: HoverCase):
        cfg = self._current_config()
        shaft_power_W = total_hover_power_W(case)
        fig, ax = plt.subplots(figsize=self._figure_size())
        if cfg["propulsion_model"] == "electric":
            current_capacity = float(cfg["battery_capacity_Wh"])
            capacities = np.linspace(max(1.0, 0.25 * current_capacity), max(1.0, 2.0 * current_capacity), DESIGN_SWEEP_POINTS)
            endurance = [
                electric_summary(shaft_power_W, {**cfg, "battery_capacity_Wh": float(capacity)})["hover_endurance_min"]
                for capacity in capacities
            ]
            ax.plot(capacities, endurance, marker="o", label="Hover Endurance [min]")
            ax.axvline(current_capacity, color=THEME["warning"], linestyle="--", linewidth=1.0, label="Current Capacity")
            ax.set_xlabel("Battery Capacity [Wh]")
            ax.set_ylabel("Hover Endurance [min]")
            ax.set_title("Battery Capacity vs Hover Endurance")
        else:
            current_capacity = float(cfg["fuel_capacity_kg"])
            capacities = np.linspace(max(0.01, 0.25 * current_capacity), max(0.01, 2.0 * current_capacity), DESIGN_SWEEP_POINTS)
            endurance = [
                propulsion_summary(
                    HoverCase(rotor=case.rotor, result=case.result, system_type=case.system_type),
                    {**cfg, "fuel_capacity_kg": float(capacity)},
                )["hover_endurance_hr"]
                for capacity in capacities
            ]
            ax.plot(capacities, endurance, marker="o", color="tab:red", label="Hover Endurance [hr]")
            ax.axvline(current_capacity, color=THEME["warning"], linestyle="--", linewidth=1.0, label="Current Capacity")
            ax.set_xlabel("Fuel Capacity [kg]")
            ax.set_ylabel("Hover Endurance [hr]")
            ax.set_title("Fuel Capacity vs Hover Endurance")
        ax.grid(True)
        ax.legend()
        return self._finish_plot(fig)

    def _plot_efficiency_loss_sensitivity(self, case: HoverCase):
        cfg = self._current_config()
        shaft_power_W = total_hover_power_W(case)
        fig, ax = plt.subplots(figsize=self._figure_size())
        if cfg["propulsion_model"] == "electric":
            motor_values = np.linspace(0.60, 0.98, 25)
            loss_values = np.linspace(0.0, 0.30, 25)
            grid = np.zeros((len(loss_values), len(motor_values)))
            for row, loss in enumerate(loss_values):
                for col, motor_eff in enumerate(motor_values):
                    grid[row, col] = electric_summary(
                        shaft_power_W,
                        {
                            **cfg,
                            "motor_efficiency": float(motor_eff),
                            "transmission_loss": float(loss),
                        },
                    )["electric_input_W"] / 1000.0
            image = ax.imshow(
                grid,
                extent=[motor_values[0], motor_values[-1], loss_values[0], loss_values[-1]],
                origin="lower",
                aspect="auto",
                cmap="viridis",
            )
            ax.scatter([float(cfg["motor_efficiency"])], [float(cfg["transmission_loss"])], color=THEME["warning"], marker="x", label="Current")
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label("Electric Input [kW]")
            ax.set_xlabel("Motor Efficiency")
            ax.set_ylabel("Transmission Loss")
            ax.set_title("Electric Efficiency and Loss Sensitivity")
        else:
            sfc_current = float(cfg["specific_fuel_consumption_kg_per_kWh"])
            sfc_values = np.linspace(max(0.05, 0.5 * sfc_current), max(0.05, 1.5 * sfc_current), 25)
            loss_values = np.linspace(0.0, 0.30, 25)
            grid = np.zeros((len(loss_values), len(sfc_values)))
            for row, loss in enumerate(loss_values):
                for col, sfc in enumerate(sfc_values):
                    grid[row, col] = propulsion_summary(
                        HoverCase(rotor=case.rotor, result=case.result, system_type=case.system_type),
                        {
                            **cfg,
                            "specific_fuel_consumption_kg_per_kWh": float(sfc),
                            "transmission_loss": float(loss),
                        },
                    )["fuel_flow_kg_hr"]
            image = ax.imshow(
                grid,
                extent=[sfc_values[0], sfc_values[-1], loss_values[0], loss_values[-1]],
                origin="lower",
                aspect="auto",
                cmap="magma",
            )
            ax.scatter([sfc_current], [float(cfg["transmission_loss"])], color=THEME["warning"], marker="x", label="Current")
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label("Fuel Flow [kg/hr]")
            ax.set_xlabel("Specific Fuel Consumption [kg/kWh]")
            ax.set_ylabel("Transmission Loss")
            ax.set_title("Engine Fuel Sensitivity")
        ax.legend()
        return self._finish_plot(fig)

    def _plot_payload_endurance_sweep(self, case: HoverCase):
        base_config = self._current_config()
        station_rows = self._station_rows()
        provider = self._xfoil_provider_for_config(base_config)
        gross = float(base_config["gross"])
        gross_values = np.linspace(
            max(MIN_SWEEP_GROSS_MASS_KG, 0.5 * gross),
            max(MIN_SWEEP_GROSS_MASS_KG, 2.0 * gross),
            DESIGN_SWEEP_POINTS,
        )
        self._log(
            "Calculating payload endurance sweep "
            f"from {gross_values[0] - gross:+.2f} to {gross_values[-1] - gross:+.2f} [kg] relative payload."
        )
        sweep = self._collect_hover_sweep(
            "Payload endurance sweep",
            base_config,
            station_rows,
            provider,
            gross_values,
            lambda value: {"gross": float(value)},
        )

        payload_delta = []
        endurance = []
        collectives_by_label: dict[str, list[float]] = {}
        for value, cfg, sweep_case in sweep:
            payload_delta.append(float(value) - gross)
            power = self._power_metrics(sweep_case, cfg)
            endurance.append(power["endurance"])
            for label, hover in self._iter_hover_results(sweep_case):
                collectives_by_label.setdefault(label, []).append(hover.collective_pitch_deg)

        fig, (ax_endurance, ax_collective) = plt.subplots(2, 1, figsize=self._figure_size(), sharex=True)
        ax_endurance.plot(payload_delta, endurance, marker="o", color="tab:green", label=self._endurance_label(base_config))
        ax_endurance.axvline(0.0, color=THEME["warning"], linestyle="--", linewidth=1.0, label="Current Gross Mass")
        ax_endurance.set_ylabel(self._endurance_label(base_config))
        ax_endurance.set_title("Endurance vs Payload Delta")
        ax_endurance.grid(True)
        ax_endurance.legend()

        for label, values in collectives_by_label.items():
            ax_collective.plot(payload_delta, values, marker="o", label=f"{label} Collective [deg]")
        ax_collective.axvline(0.0, color=THEME["warning"], linestyle="--", linewidth=1.0)
        ax_collective.set_xlabel("Payload Delta from Current Gross Mass [kg]")
        ax_collective.set_ylabel("Required Collective [deg]")
        ax_collective.set_title("Trim Margin vs Payload")
        ax_collective.grid(True)
        ax_collective.legend()
        fig.suptitle("Payload Endurance Sweep")
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
        fig, ax = plt.subplots(figsize=self._figure_size())
        ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:red"])
        ax.set_ylabel("Power [kW]")
        ax.set_title("Coaxial Interference Power")
        ax.grid(True, axis="y")
        return self._finish_plot(fig)

    def _plot_interference_loss_vs_spacing(self, case: HoverCase):
        """Solve the spacing sweep on the calling thread and plot it.

        This is the path used when there is no Panel event loop to run the
        sweep off, and by direct callers such as the tests. The interactive
        path routes the same sweep through ``_start_spacing_sweep`` so the run
        rail can show progress and offer a cancel.
        """
        if not isinstance(case.result, CoaxialHoverResult):
            raise ValueError("Interference loss spacing sweep requires a coaxial result.")

        base_config = self._current_config()
        station_rows = self._station_rows()
        points = int(base_config["sweep_points"])
        self._log(
            "Calculating coaxial spacing sweep for interference loss. "
            f"Range: z/R {COAXIAL_SPACING_SWEEP_MIN:.2f}..{COAXIAL_SPACING_SWEEP_MAX:.2f} "
            f"over {points} points."
        )
        sweep = self._run_interference_sweep(
            base_config,
            station_rows,
            self._xfoil_provider_for_config(base_config),
            np.linspace(COAXIAL_SPACING_SWEEP_MIN, COAXIAL_SPACING_SWEEP_MAX, points),
        )
        if not sweep.solved_points:
            raise ValueError("The spacing sweep produced no converged points.")
        self._log_sweep_summary(sweep)
        return self._figure_from_interference_sweep(sweep, base_config)

    def _figure_from_interference_sweep(self, sweep, base_config: dict[str, Any]):
        """Draw the interference-loss curve from solved sweep points."""
        solved = sweep.solved_points
        spacings = [point.z_over_R for point in solved]
        losses = [point.result.interference_loss_ratio for point in solved]
        current_spacing = float(base_config["coaxial_spacing_ratio"])

        fig, ax = plt.subplots(figsize=self._figure_size())
        ax.plot(spacings, losses, marker="o", label="Interference Loss")
        ax.axvline(current_spacing, color=THEME["warning"], linestyle="--", linewidth=1.1, label="Current z/R")
        ax.set_title("Coaxial Spacing Sweep")
        ax.set_xlabel("Coaxial Spacing z/R")
        ax.set_ylabel("Interference Loss")
        ax.grid(True)

        for point in sweep.failed_points:
            ax.axvline(point.z_over_R, color=THEME["danger"], linestyle=":", linewidth=0.9)
        if sweep.failed_points:
            ax.plot([], [], color=THEME["danger"], linestyle=":", label="Did not converge")
        if sweep.cancelled:
            ax.set_title("Coaxial Spacing Sweep (cancelled)")
        ax.legend()

        best_index = int(np.argmin(losses))
        ax.scatter([spacings[best_index]], [losses[best_index]], color=THEME["accent"], zorder=4)
        ax.annotate(
            f"min {losses[best_index]:.3f} at z/R {spacings[best_index]:.2f}",
            xy=(spacings[best_index], losses[best_index]),
            xytext=(8, 10),
            textcoords="offset points",
            color=THEME["text"],
            fontsize=9,
        )
        return self._finish_plot(fig)

    def _plot_coaxial_thrust_share_yaw(self, case: HoverCase):
        if not isinstance(case.result, CoaxialHoverResult):
            raise ValueError("Coaxial thrust-share yaw plot requires a coaxial result.")

        base_config = self._current_config()
        station_rows = self._station_rows()
        provider = self._xfoil_provider_for_config(base_config)
        result = case.result
        common_collective = 0.5 * (result.upper.collective_pitch_deg + result.lower.collective_pitch_deg)
        upper_bias = result.upper.collective_pitch_deg - common_collective
        lower_bias = result.lower.collective_pitch_deg - common_collective
        yaw_inputs = np.linspace(-5.0, 5.0, CONTROL_SWEEP_POINTS)
        self._log("Calculating coaxial thrust-share yaw sweep from -5.0 to +5.0 [deg] differential collective.")

        valid = []
        failures = 0
        for yaw_input in yaw_inputs:
            upper_collective = common_collective + upper_bias - float(yaw_input)
            lower_collective = common_collective + lower_bias + float(yaw_input)
            try:
                fixed_case = self._solve_fixed_collective_case(
                    base_config,
                    station_rows,
                    provider,
                    upper_collective,
                    lower_collective,
                )
                if not isinstance(fixed_case.result, CoaxialHoverResult):
                    raise ValueError("Fixed differential sweep generated a non-coaxial result.")
                valid.append((float(yaw_input), fixed_case.result))
            except Exception:
                failures += 1
        if failures:
            self._log(f"WARNING - Coaxial thrust-share yaw sweep skipped {failures} non-converged points.")
        if len(valid) < 2:
            raise ValueError("Coaxial thrust-share yaw sweep could not compute enough valid points.")

        thrust_share = [
            item[1].upper.total_thrust_N / max(item[1].total_thrust_N, 1e-9)
            for item in valid
        ]
        yaw_torque = [item[1].net_aircraft_yaw_torque_Nm for item in valid]
        total_thrust = [item[1].total_thrust_N for item in valid]
        yaw_input_values = [item[0] for item in valid]

        fig, (ax_yaw, ax_thrust) = plt.subplots(2, 1, figsize=self._figure_size(), sharex=True)
        scatter = ax_yaw.scatter(
            thrust_share,
            yaw_torque,
            c=yaw_input_values,
            cmap="coolwarm",
            label="Differential Collective",
        )
        ax_yaw.plot(thrust_share, yaw_torque, color=THEME["muted"], linewidth=0.9, alpha=0.8)
        ax_yaw.axhline(0.0, color=THEME["warning"], linestyle="--", linewidth=1.0)
        current_share = result.upper.total_thrust_N / max(result.total_thrust_N, 1e-9)
        ax_yaw.axvline(current_share, color=THEME["warning"], linestyle=":", linewidth=1.0, label="Current Share")
        colorbar = fig.colorbar(scatter, ax=ax_yaw)
        colorbar.set_label("Yaw Input [deg]")
        ax_yaw.set_ylabel("Aircraft Yaw Torque [Nm]")
        ax_yaw.set_title("Yaw Torque vs Upper-Rotor Thrust Share")
        ax_yaw.grid(True)
        ax_yaw.legend()

        ax_thrust.plot(thrust_share, total_thrust, marker="o", label="Total Thrust [N]")
        ax_thrust.axhline(result.total_thrust_N, color=THEME["warning"], linestyle="--", linewidth=1.0, label="Current Thrust")
        ax_thrust.axvline(current_share, color=THEME["warning"], linestyle=":", linewidth=1.0)
        ax_thrust.set_xlabel("Upper Rotor Thrust Share")
        ax_thrust.set_ylabel("Total Thrust [N]")
        ax_thrust.set_title("Lift Coupling During Differential Collective")
        ax_thrust.grid(True)
        ax_thrust.legend()
        fig.suptitle("Coaxial Thrust Share and Yaw")
        return self._finish_plot(fig)

    def _plot_forward_powers(self, case: HoverCase):
        self._require_single_rotor(case)
        hover = primary_hover_result(case)
        cfg = self._current_config()
        estimates = velocity_sweep(case.rotor, hover, density_kg_m3=float(cfg["density"]), flat_plate_area_m2=float(cfg["fpa"]))
        velocities = [estimate.velocity_m_s * 3.6 for estimate in estimates]
        fig, ax = plt.subplots(figsize=self._figure_size())
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
        fig, ax1 = plt.subplots(figsize=self._figure_size())
        self._accent_axis(ax1, SERIES_COLORS[0])
        ax1.plot(velocities, endurance, color=SERIES_COLORS[0], label="Endurance [hr]")
        ax1.set_xlabel("Velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")
        ax1.grid(True)
        ax2 = self._twin_axis(ax1, SERIES_COLORS[1])
        ax2.plot(velocities, flight_range, color=SERIES_COLORS[1], label="Range [km]")
        ax2.set_ylabel("Range [km]")
        ax1.set_title("Electric Range and Endurance")
        self._combined_legend(ax1, ax2)
        return self._finish_plot(fig)

    def _plot_fuel_range(self, case: HoverCase):
        self._require_single_rotor(case)
        cfg = self._current_config()
        hover = primary_hover_result(case)
        estimates = velocity_sweep(case.rotor, hover, density_kg_m3=float(cfg["density"]), flat_plate_area_m2=float(cfg["fpa"]))
        velocities, endurance, flight_range = fossil_range_sweep(estimates, hover, cfg)
        fig, ax1 = plt.subplots(figsize=self._figure_size())
        self._accent_axis(ax1, SERIES_COLORS[0])
        ax1.plot(velocities, endurance, color=SERIES_COLORS[0], label="Endurance [hr]")
        ax1.set_xlabel("Free Stream Velocity [km/hr]")
        ax1.set_ylabel("Endurance [hr]")
        ax1.grid(True)
        ax2 = self._twin_axis(ax1, SERIES_COLORS[1])
        ax2.plot(velocities, flight_range, color=SERIES_COLORS[1], label="Range [km]")
        ax2.set_ylabel("Range [km]")
        ax1.set_title("Fuel Range and Endurance")
        self._combined_legend(ax1, ax2)
        return self._finish_plot(fig)

    def _accent_axis(self, ax, color: str, side: str = "left"):
        """Tag an axis so _finish_plot colours its ticks, label, and spine."""
        ax._pycopter_accent = color
        ax._pycopter_accent_side = side
        return ax

    def _twin_axis(self, ax, color: str):
        """Add a right-hand y-axis that shares x with ``ax``.

        Used when a quantity is too large or too small to read on the left
        axis. A panel never gets more than these two axes.
        """
        twin = ax.twinx()
        twin._pycopter_twin = True
        self._accent_axis(twin, color, side="right")
        return twin

    def _series_magnitude(self, values) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0
        return float(np.max(np.abs(finite)))

    def _normalized_shape(self, values):
        """Scale a series onto 0..1, which is what an autoscaled axis shows."""
        finite = np.asarray(values, dtype=float)
        if finite.size == 0 or not np.isfinite(finite).any():
            return None
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        if high - low <= 0.0:
            return None
        return (finite - low) / (high - low)

    def _shapes_coincide(self, first, second) -> bool:
        """True when two curves would overlap once each axis autoscales."""
        left = self._normalized_shape(first)
        right = self._normalized_shape(second)
        if left is None or right is None or left.size != right.size:
            return False
        return float(np.nanmax(np.abs(left - right))) < COINCIDENT_SHAPE_GAP

    def _axis_layout(self, series_values: list) -> list[tuple[int, int]]:
        """Assign each series to a (panel, axis) slot from its measured data.

        Returns one ``(panel_index, axis_index)`` per series, where axis 0 is a
        panel's left y-axis and axis 1 its right. Implements the three rules
        documented on SHARED_AXIS_MAX_RATIO.
        """
        magnitudes = [self._series_magnitude(values) for values in series_values]
        # panels[panel][axis] = list of series indices already placed there
        panels: list[list[list[int]]] = [[[0]]]
        placement = [(0, 0)]

        for index in range(1, len(series_values)):
            panel_index = len(panels) - 1
            panel = panels[panel_index]
            slot = None

            for axis_index, members in enumerate(panel):
                # A flat or all-zero series carries no scale of its own, so it
                # can join any axis rather than forcing a new panel.
                if magnitudes[index] == 0.0:
                    slot = (panel_index, axis_index)
                    break
                ratios = [
                    max(magnitudes[index], magnitudes[member])
                    / min(magnitudes[index], magnitudes[member])
                    for member in members
                    if magnitudes[member] > 0.0
                ]
                if ratios and max(ratios) <= SHARED_AXIS_MAX_RATIO:
                    slot = (panel_index, axis_index)
                    break

            if slot is None and len(panel) == 1:
                opposite = [series_values[member] for member in panel[0]]
                if not any(self._shapes_coincide(series_values[index], other) for other in opposite):
                    panel.append([])
                    slot = (panel_index, 1)

            if slot is None and len(panels) < MAX_PLOT_PANELS:
                panels.append([[]])
                slot = (len(panels) - 1, 0)

            if slot is None:
                # Out of panels: fall back to the least crowded axis available.
                slot = (len(panels) - 1, len(panels[-1]) - 1)

            panels[slot[0]][slot[1]].append(index)
            placement.append(slot)

        return placement

    def _rotor_linestyle(self, index: int) -> str:
        return ("-", "--", ":")[index % 3]

    def _combined_legend(self, primary_ax, *other_axes, loc: str = "best") -> None:
        """Draw one legend covering series from every axis in the figure."""
        handles: list[Any] = []
        labels: list[str] = []
        for ax in (primary_ax, *other_axes):
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        if handles:
            primary_ax.legend(handles, labels, loc=loc, fontsize=9)

    def _plot_element_series(self, case: HoverCase, series, title: str):
        """Plot per-element quantities against r/R with a data-driven axis layout.

        ``series`` is a sequence of ``(column, axis_label, legend_label)``,
        where ``column`` is either a load-table column name or a callable that
        derives a series from the load-table frame. Quantities within
        SHARED_AXIS_MAX_RATIO of each other share an axis, the rest take the
        right-hand axis, and anything that still does not fit gets its own
        stacked panel. Colour identifies the quantity and line style the rotor,
        so a coaxial case keeps the same coding as a single rotor.
        """
        rotors = list(self._iter_hover_results(case))
        frames = [pd.DataFrame(load_rows_for_result(hover)) for _, hover in rotors]
        colors = [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(len(series))]
        overlays = self._overlay_series_frames()

        def values_for(frame, column):
            return np.asarray(column(frame) if callable(column) else frame[column], dtype=float)

        # Lay the axes out from every rotor's data at once, so a coaxial case
        # and a single-rotor case of the same design agree on the layout.
        # Overlaid runs join the same measurement so a compared run cannot fall
        # off the scale that was chosen without it.
        overlay_frames = [frame for _, run_frames in overlays for _, frame in run_frames]
        combined = [
            np.concatenate(
                [values_for(frame, column) for frame in frames + overlay_frames]
            )
            for column, _, _ in series
        ]
        placement = self._axis_layout(combined)
        panel_count = max(panel for panel, _ in placement) + 1

        # The first panel carries up to two quantities and their legend, so it
        # gets the larger share of the figure height.
        height_ratios = [1.5] + [1.0] * (panel_count - 1)
        fig, panel_axes = plt.subplots(
            panel_count,
            1,
            figsize=self._figure_size(),
            sharex=True,
            squeeze=False,
            gridspec_kw={"height_ratios": height_ratios},
        )
        panel_axes = [row[0] for row in panel_axes]

        # Build only the axes the layout actually asked for.
        axes: dict[tuple[int, int], Any] = {}
        for series_index, (panel, axis_index) in enumerate(placement):
            if (panel, axis_index) in axes:
                continue
            if axis_index == 0:
                axes[(panel, axis_index)] = panel_axes[panel]
            else:
                axes[(panel, axis_index)] = self._twin_axis(panel_axes[panel], colors[series_index])

        # An axis carrying exactly one quantity is tinted to match its curve;
        # a shared axis stays neutral and relies on the legend.
        owners: dict[tuple[int, int], list[int]] = {}
        for series_index, slot in enumerate(placement):
            owners.setdefault(slot, []).append(series_index)
        for slot, members in owners.items():
            if len(members) == 1:
                self._accent_axis(axes[slot], colors[members[0]], side="right" if slot[1] else "left")

        # Overlaid runs are drawn first, wider and translucent, so the active
        # run sits on top of them. A compared curve that coincides with the
        # active one then reads as a halo instead of silently hiding it.
        for run_index, (run_label, run_frames) in enumerate(overlays):
            dashes = OVERLAY_DASHES[run_index % len(OVERLAY_DASHES)]
            for rotor_index, (rotor_label, frame) in enumerate(run_frames):
                prefix = f"{rotor_label} " if len(run_frames) > 1 else ""
                for series_index, (column, _, legend_label) in enumerate(series):
                    ax = axes[placement[series_index]]
                    ax.plot(
                        frame["r_over_R"],
                        values_for(frame, column),
                        color=colors[series_index],
                        dashes=dashes,
                        linewidth=2.4,
                        alpha=OVERLAY_ALPHA,
                        zorder=1,
                        label=(
                            f"{run_label} {prefix}{legend_label}"
                            if series_index == 0 and rotor_index == 0
                            else "_nolegend_"
                        ),
                    )

        marker_slots = max(1, len(series) * len(rotors))
        for rotor_index, ((rotor_label, _), frame) in enumerate(zip(rotors, frames)):
            linestyle = self._rotor_linestyle(rotor_index)
            prefix = f"{rotor_label} " if len(rotors) > 1 else ""
            marker_step = max(1, len(frame) // SERIES_MARKER_COUNT)
            for series_index, (column, _, legend_label) in enumerate(series):
                ax = axes[placement[series_index]]
                # Stagger markers so curves that share an axis stay separable
                # where they run close together.
                slot = series_index * len(rotors) + rotor_index
                ax.plot(
                    frame["r_over_R"],
                    values_for(frame, column),
                    color=colors[series_index],
                    linestyle=linestyle,
                    marker=SERIES_MARKERS[series_index % len(SERIES_MARKERS)],
                    markersize=4.5,
                    markevery=((slot * marker_step) // marker_slots, marker_step),
                    zorder=3,
                    label=f"{prefix}{legend_label}",
                )

        for slot, members in owners.items():
            axes[slot].set_ylabel(", ".join(series[index][1] for index in members))
        for panel in range(panel_count):
            panel_axes[panel].grid(True)
            twin = axes.get((panel, 1))
            self._combined_legend(panel_axes[panel], *(t for t in (twin,) if t is not None))
        panel_axes[-1].set_xlabel("r/R")
        panel_axes[0].set_title(title)
        return self._finish_plot(fig)

    def _overlay_cases(self) -> list[tuple[str, HoverCase]]:
        """Checked runs, other than the active one, that carry a result."""
        if not self.overlay_runs.value:
            return []
        cases: list[tuple[str, HoverCase]] = []
        for run_id in self.checked_run_ids:
            record = self.runs.get(run_id)
            if record is None or record.result is None or run_id == self.active_run_id:
                continue
            cases.append(
                (
                    record.run_id,
                    HoverCase(
                        rotor=None,
                        result=record.result,
                        system_type="coaxial" if record.is_coaxial else "single",
                    ),
                )
            )
        return cases

    def _overlay_series_frames(self) -> list[tuple[str, list[tuple[str, pd.DataFrame]]]]:
        """Load tables of the overlaid runs, keyed the same way as the active run."""
        overlays = []
        for run_id, case in self._overlay_cases():
            frames = [
                (label, pd.DataFrame(load_rows_for_result(hover)))
                for label, hover in self._iter_hover_results(case)
            ]
            frames = [(label, frame) for label, frame in frames if not frame.empty]
            if frames:
                overlays.append((run_id, frames))
        return overlays

    def _collect_hover_sweep(
        self,
        label: str,
        base_config: dict[str, Any],
        station_rows: list[dict[str, Any]],
        provider,
        values,
        updates_for_value,
    ) -> list[tuple[float, dict[str, Any], HoverCase]]:
        valid: list[tuple[float, dict[str, Any], HoverCase]] = []
        failures = 0
        first_error = ""
        for value in values:
            cfg = normalize_config({**base_config, **updates_for_value(value)})
            try:
                valid.append((float(value), cfg, run_hover_case(cfg, station_rows, polar_provider=provider)))
            except Exception as err:
                failures += 1
                if not first_error:
                    first_error = str(err)
        if failures:
            detail = f"; first error: {first_error}" if first_error else ""
            self._log(f"WARNING - {label} skipped {failures}/{len(values)} points{detail}.")
        if len(valid) < 2:
            raise ValueError(f"{label} could not compute enough valid points.")
        return valid

    def _solve_fixed_collective_case(
        self,
        config: dict[str, Any],
        station_rows: list[dict[str, Any]],
        provider,
        upper_collective_deg: float,
        lower_collective_deg: float | None = None,
    ) -> HoverCase:
        cfg = normalize_config(config)
        settings = build_solver_settings(cfg)
        upper_rotor = build_rotor_spec(cfg, station_rows, name="upper")
        operating_point = OperatingPoint(
            density_kg_m3=float(cfg["density"]),
            kinematic_viscosity_m2_s=float(cfg["kinematic_viscosity_m2_s"]),
            gross_mass_kg=float(cfg["gross"]),
            collective_pitch_deg=float(upper_collective_deg),
            trim_mode="fixed_collective",
        )
        if cfg["rotor_system_type"] == "coaxial":
            lower_collective = float(
                upper_collective_deg if lower_collective_deg is None else lower_collective_deg
            )
            lower_rotor = build_rotor_spec(
                cfg,
                station_rows,
                name="lower",
                scale=float(cfg["lower_rotor_scale"]),
                headspeed_ratio=float(cfg["lower_rotor_speed_ratio"]),
                rotation_direction=-1,
            )
            result = solve_coaxial_hover(
                CoaxialSpec(
                    upper_rotor=upper_rotor,
                    lower_rotor=lower_rotor,
                    spacing_ratio=float(cfg["coaxial_spacing_ratio"]),
                    trim_mode="equal_collective",
                    lower_collective_offset_deg=lower_collective - float(upper_collective_deg),
                ),
                operating_point,
                polar_provider=provider,
                settings=settings,
            )
            return HoverCase(rotor=upper_rotor, result=result, system_type="coaxial")

        solver = HoverSolver(provider, settings)
        result = solver.solve_fixed_collective(upper_rotor, operating_point, float(upper_collective_deg))
        return HoverCase(rotor=upper_rotor, result=result, system_type="single")

    def _power_metrics(self, case: HoverCase, config: dict[str, Any]) -> dict[str, float]:
        prop = propulsion_summary(case, config)
        if config["propulsion_model"] == "electric":
            return {
                "shaft_kW": total_hover_power_W(case) / 1000.0,
                "input_kW": prop["electric_input_W"] / 1000.0,
                "endurance": prop["hover_endurance_min"],
            }
        return {
            "shaft_kW": total_hover_power_W(case) / 1000.0,
            "input_kW": prop["engine_input_W"] / 1000.0,
            "endurance": prop["hover_endurance_hr"],
        }

    def _input_power_label(self, config: dict[str, Any]) -> str:
        if config["propulsion_model"] == "electric":
            return "Electric Input [kW]"
        return "Engine Input [kW]"

    def _endurance_label(self, config: dict[str, Any]) -> str:
        if config["propulsion_model"] == "electric":
            return "Hover Endurance [min]"
        return "Hover Endurance [hr]"

    def _mean_collective_deg(self, case: HoverCase) -> float:
        collectives = [hover.collective_pitch_deg for _, hover in self._iter_hover_results(case)]
        return float(np.mean(collectives)) if collectives else 0.0

    def _max_load_value(self, case: HoverCase, attr_name: str) -> float:
        values = [
            float(getattr(load, attr_name))
            for _, hover in self._iter_hover_results(case)
            for load in hover.element_loads
        ]
        return max(values) if values else 0.0

    def _mean_load_value(self, case: HoverCase, attr_name: str) -> float:
        values = [
            float(getattr(load, attr_name))
            for _, hover in self._iter_hover_results(case)
            for load in hover.element_loads
        ]
        return float(np.mean(values)) if values else 0.0

    def _net_aircraft_yaw_torque(self, case: HoverCase) -> float:
        if isinstance(case.result, CoaxialHoverResult):
            return case.result.net_aircraft_yaw_torque_Nm
        return case.result.aircraft_yaw_torque_Nm

    def _iter_hover_results(self, case: HoverCase):
        """Rotor results the canvas reads, filtered by the SHOWING toggle.

        A single rotor has nothing to choose between. A coaxial pair can be
        read as the upper rotor, the lower rotor, both together, or each rotor
        minus its isolated reference, and summary, loads and plots all follow
        the same choice so they never disagree.
        """
        if not isinstance(case.result, CoaxialHoverResult):
            return [("Rotor", case.result)]

        result = case.result
        if self.showing == "upper":
            return [("Upper", result.upper)]
        if self.showing == "lower":
            return [("Lower", result.lower)]
        if self.showing == "delta":
            return [
                ("Upper Δ", _hover_result_delta(result.upper, result.isolated_upper)),
                ("Lower Δ", _hover_result_delta(result.lower, result.isolated_lower)),
            ]
        return [("Upper", result.upper), ("Lower", result.lower)]

    def _require_single_rotor(self, case: HoverCase) -> None:
        if case.system_type != "single":
            raise ValueError("Legacy forward-flight plots are available for single-rotor cases only.")

    def _blank_figure(self, text: str):
        fig, ax = plt.subplots(figsize=self._figure_size())
        ax.text(0.5, 0.5, text, ha="center", va="center", color=THEME["muted"])
        ax.set_axis_off()
        return self._finish_plot(fig)

    def _finish_plot(self, fig):
        fig.patch.set_facecolor(THEME["plot"])
        for ax in fig.axes:
            accent = getattr(ax, "_pycopter_accent", None)
            is_twin = getattr(ax, "_pycopter_twin", False)

            ax.set_facecolor(THEME["plot"])
            ax.tick_params(axis="x", colors=THEME["muted"])
            ax.tick_params(axis="y", colors=accent or THEME["muted"])
            ax.xaxis.label.set_color(THEME["text"])
            # An accented axis carries one quantity, so tying the label and
            # ticks to the series colour says which curve the scale belongs to.
            ax.yaxis.label.set_color(accent or THEME["text"])
            ax.title.set_color(THEME["text"])
            for side, spine in ax.spines.items():
                spine.set_color(THEME["border"])
            if accent:
                ax.spines[getattr(ax, "_pycopter_accent_side", "left")].set_color(accent)
            for text in ax.get_xticklabels():
                text.set_color(THEME["muted"])
            for text in ax.get_yticklabels():
                text.set_color(accent or THEME["muted"])
            # Twin axes must not draw a second grid over the primary one.
            if not is_twin:
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
        # Slightly translucent so a curve running under the legend stays visible.
        legend.get_frame().set_alpha(0.82)
        for text in legend.get_texts():
            text.set_color(THEME["text"])


    # ------------------------------------------------------------------
    # Options and enabled state
    # ------------------------------------------------------------------

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
            "Stall Margin vs Radius",
            "Geometry Load Contribution",
            "Disk Loading Sensitivity",
            "Rotor Diameter Sizing",
            "Reference RPM Sweep",
            "Collective Authority Curve",
            "Energy Capacity vs Endurance",
            "Efficiency and Loss Sensitivity",
            "Payload Endurance Sweep",
        ]
        if self.current_case and self.current_case.system_type == "coaxial":
            options.append("Coaxial Interference")
            options.append("Interference Loss vs Spacing")
            options.append("Coaxial Thrust Share and Yaw")
        if (
            self.current_case
            and self.current_case.system_type == "single"
            and str(self.flight_mode.value) == "forward_flight"
        ):
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
        for widget in (
            self.coaxial_spacing,
            self.coaxial_trim_mode,
            self.lower_rotor_speed_ratio,
            self.lower_collective_offset,
            self.lower_rotor_scale,
        ):
            widget.disabled = not is_coaxial
        self.lower_headspeed_rpm.disabled = True
        self.station_table.disabled = self.geometry_mode.value != "station_table"
        self.chord.disabled = self.geometry_mode.value == "station_table"
        self.station_count.disabled = self.geometry_mode.value == "station_table"
        self.collective_pitch.disabled = self.trim_mode.value != "fixed_collective"
        for widget in (
            self.polar_alpha_min,
            self.polar_alpha_max,
            self.polar_alpha_step,
            self.polar_n_crit,
            self.xfoil_max_iterations,
            self.xfoil_timeout,
        ):
            # Strict cache-only mode never launches XFOIL, so its generation
            # settings would be misleading if they still looked live.
            widget.disabled = not bool(self.new_polars.value)

        running = self._active_job is not None
        # CANCEL takes the primary button's place while a run is in flight, so
        # the footer never holds two competing actions at half width.
        self.solve_btn.disabled = running
        self.solve_btn.visible = not running
        self.cancel_btn.visible = running
        self.generate_plot_btn.disabled = running or (
            self.current_case is None and self.plot_select.value != "Blade Geometry"
        )
        self.set_baseline_btn.disabled = self.active_run_id is None
        self.delete_run_btn.disabled = self.active_run_id is None

    def _update_lower_rpm_display(self) -> None:
        self.lower_headspeed_rpm.value = float(self.headspeed_rpm.value) * float(
            self.lower_rotor_speed_ratio.value
        )

    # ------------------------------------------------------------------
    # XFOIL provider
    # ------------------------------------------------------------------

    def _xfoil_provider_for_config(self, config: dict[str, Any]):
        key = self._xfoil_provider_cache_key(config)
        if self._xfoil_provider is None or self._xfoil_provider_key != key:
            self._cleanup_xfoil_provider()
            self._xfoil_provider = build_xfoil_provider(config, self._xfoil_progress)
            self._xfoil_provider_key = key
        return self._xfoil_provider

    def _xfoil_progress(self, event: str, payload: dict[str, Any]) -> None:
        """Called from the calculation thread; hand the event to the UI loop."""
        self._queue.put(("xfoil", (event, payload)))

    def _xfoil_provider_cache_key(self, config: dict[str, Any]) -> tuple[Any, ...]:
        return (
            bool(config["new_polars"]),
            float(config["polar_alpha_min_deg"]),
            float(config["polar_alpha_max_deg"]),
            float(config.get("polar_alpha_step_deg", 1.0)),
            float(config.get("polar_reynolds_bin", 100000.0)),
            float(config.get("polar_mach_bin", 0.1)),
            float(config.get("polar_n_crit", 9.0)),
            int(config.get("xfoil_max_iterations", 400)),
            int(config.get("xfoil_timeout_s", 60)),
            int(config["xfoil_parallel_workers"]),
            str(config["xfoil_parallel_backend"]),
            str(config.get("xfoil_cache_directory") or "").strip(),
        )

    def _cleanup_xfoil_provider(self) -> None:
        if self._xfoil_provider is None:
            return
        cleanup = getattr(self._xfoil_provider, "cleanup", None)
        if cleanup is not None:
            cleanup()
        self._xfoil_provider = None
        self._xfoil_provider_key = None


def _hover_result_delta(actual: HoverResult, reference: HoverResult) -> HoverResult:
    """Element-wise difference between a rotor and its isolated reference.

    Geometry columns keep the actual values so the radial axis and the chord
    distribution still read correctly; everything the coaxial interaction can
    change is differenced. Both results come from the same rotor and the same
    element count, so the rows line up.
    """
    scalar_fields = (
        "collective_pitch_deg",
        "total_thrust_N",
        "per_blade_thrust_N",
        "total_torque_Nm",
        "per_blade_torque_Nm",
        "aircraft_yaw_torque_Nm",
        "power_W",
        "induced_power_W",
        "profile_power_W",
        "ideal_power_W",
        "figure_of_merit",
        "mean_induced_velocity_m_s",
        "ct",
        "cp",
        "mean_loss_factor",
        "root_flap_bending_moment_Nm_per_blade",
        "root_lag_moment_Nm_per_blade",
        "aerodynamic_pitching_moment_Nm_per_blade",
    )
    geometry_fields = ("r_m", "r_over_R", "dr_m", "chord_m", "twist_deg")
    element_fields = tuple(
        name
        for name in ElementLoad.__dataclass_fields__
        if name not in geometry_fields and name != "alpha_clamped"
    )

    element_loads = []
    for left, right in zip(actual.element_loads, reference.element_loads):
        values = {name: getattr(left, name) for name in geometry_fields}
        values["alpha_clamped"] = left.alpha_clamped or right.alpha_clamped
        for name in element_fields:
            values[name] = getattr(left, name) - getattr(right, name)
        element_loads.append(ElementLoad(**values))

    scalars = {
        name: getattr(actual, name) - getattr(reference, name) for name in scalar_fields
    }
    return HoverResult(
        solidity=actual.solidity,
        element_loads=element_loads,
        substituted_polar_bins=actual.substituted_polar_bins,
        **scalars,
    )


def _values_match(left: Any, right: Any) -> bool:
    """Compare a config value with a widget value, tolerating float rounding."""
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return abs(float(left) - float(right)) <= 1e-9 * scale
    return left == right


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def create_app():
    """Create a fresh Panel app session."""
    return PycopterWebApp().view()
