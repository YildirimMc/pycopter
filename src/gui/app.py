"""Panel Web UI for PyCopter."""

from __future__ import annotations

import io
import json
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
    HoverResult,
    HoverSolver,
    OperatingPoint,
    solve_coaxial_hover,
)

from .calculations import (
    DEFAULT_CONFIG,
    DEFAULT_STATION_ROWS,
    CONFIG_VERSION,
    HoverCase,
    build_solver_settings,
    build_rotor_spec,
    build_xfoil_provider,
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


# Wide enough for the 410 px control stack plus panel padding and the column
# scrollbar, so a scrolling input column never clips its own widgets.
INPUT_COLUMN_WIDTH = 448
# The output log starts at its minimum height and can be dragged upwards, which
# takes room from the result area rather than from the page.
LOG_HEIGHT = 132
LOG_MIN_HEIGHT = 132
LOG_GRIP_HEIGHT = 9
# Padding between the log region box and the terminal widget inside it.
LOG_TERMINAL_INSET = 32
# The result area never shrinks below this, so dragging cannot hide the plot.
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
COAXIAL_SPACING_SWEEP_POINTS = 25
COAXIAL_SPACING_SWEEP_MIN = 0.05
COAXIAL_SPACING_SWEEP_MAX = 1.50
DESIGN_SWEEP_POINTS = 17
CONTROL_SWEEP_POINTS = 31
MIN_SWEEP_GROSS_MASS_KG = 0.05
# Outward offset in points for a third y-axis, so its spine clears the second.
AXIS_OFFSET_POINTS = 58
# One colour per plotted quantity. Rotors are separated by line style instead,
# so a quantity keeps its colour whether the case is single or coaxial.
SERIES_COLORS = ("#4db6ac", "#e08a4f", "#9d8df1", "#77b6ea")
# Sparse per-quantity markers keep curves apart when their shapes coincide.
# Reynolds and Mach, for example, are both linear in radius, so on separate
# axes they plot as the same line and colour alone would not separate them.
SERIES_MARKERS = ("o", "s", "^", "D")
SERIES_MARKER_COUNT = 9
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
    /* The dashboard is a fixed-height app shell: every region scrolls
       internally so expanding a section never grows or clips the page. */
    height: 100%;
    overflow: hidden;
}
.pycopter-shell {
    font-family: Arial, Helvetica, sans-serif;
    color: #e7ecf2;
    height: 100%;
    min-height: 0;
}
/* Panel wraps each child in its own div; these keep the flex chain able to
   shrink so the inner scroll containers, not the page, absorb overflow. */
.pycopter-shell > div,
.pycopter-body > div,
.pycopter-results > div {
    min-height: 0;
}
.pycopter-body {
    min-height: 0;
    overflow-x: auto;
    overflow-y: hidden;
}
.pycopter-column {
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    flex: 0 0 auto;
}
.pycopter-results {
    min-width: 520px;
    min-height: 0;
    flex: 1 1 auto;
}
.pycopter-toolbar {
    flex: 0 0 auto;
    overflow-x: auto;
}
.pycopter-log-region {
    flex: 0 0 auto;
}
/* Drag handle that grows the output log upwards into the result area. */
.pycopter-log-grip {
    flex: 0 0 auto;
    height: 9px;
    cursor: ns-resize;
    background: #1b1d20;
    border-top: 1px solid #3a4048;
    border-bottom: 1px solid #3a4048;
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
    border-radius: 2px;
    background: #59616b;
}
.pycopter-log-grip:hover::after,
.pycopter-log-grip.pycopter-grip-active::after {
    background: #4db6ac;
}
.pycopter-log-grip.pycopter-grip-active {
    background: #23262b;
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
    height: 100%;
}
.pycopter-log .xterm {
    height: 100% !important;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    background: #121416;
    border: 1px solid #3a4048;
    padding: 4px 6px;
}
.pycopter-log .xterm-viewport,
.pycopter-log .xterm-screen {
    background: #121416 !important;
}
.pycopter-log .xterm-viewport {
    overflow-y: auto !important;
}
.pycopter-log .xterm-rows {
    user-select: text;
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
    min-height: 0;
    overflow: hidden;
}
/* Keep the rendered figure inside its frame at any window size. The figure is
   also re-rendered at the measured frame size, so this only smooths the gap
   between a resize and the next Generate Plot. */
.pycopter-plot-frame img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}
.pycopter-table-scroll {
    overflow-x: auto;
    overflow-y: hidden;
    min-height: 0;
}
.pycopter-result-table .tabulator-tableholder {
    overflow-x: auto !important;
}
.pycopter-load-table .tabulator-table {
    min-width: max-content;
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
.pycopter-xfoil-checkbox,
.pycopter-xfoil-checkbox *,
.pycopter-xfoil-checkbox label,
.pycopter-xfoil-checkbox .bk,
.pycopter-xfoil-checkbox .bk-checkbox-label,
.pycopter-xfoil-checkbox .bk-input-group-label {
    color: #e7ecf2 !important;
    opacity: 1 !important;
}
.pycopter-xfoil-checkbox input[type="checkbox"] {
    accent-color: #4db6ac;
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

    height_px = param.Integer(default=LOG_HEIGHT)

    _template = (
        '<div id="grip" class="pycopter-log-grip"'
        ' title="Drag to resize the output log. Double-click to reset."></div>'
    )
    _scripts = {
        "render": """
            const MIN_HEIGHT = %(min_height)d;
            const RESULT_MIN = %(result_min)d;
            const GRIP_HEIGHT = %(grip_height)d;
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
                const inner = height - TERMINAL_INSET;
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
            "terminal_inset": LOG_TERMINAL_INSET,
        }
    }


class PycopterWebApp:
    """Stateful Panel application."""

    def __init__(self) -> None:
        pn.extension("tabulator", "terminal", raw_css=[RAW_CSS])
        self.current_case: HoverCase | None = None
        self.initialized_rotor = None
        self._xfoil_provider = None
        self._xfoil_provider_key: tuple[Any, ...] | None = None
        self.output_lines: list[str] = []
        self._plot_area_size_px: tuple[int, int] | None = None
        self._current_plot_name = ""
        self.plot_fig = self._blank_figure("Initialize rotor, then calculate hover.")
        self._plot_save_available = False

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
            self.save_plot_download,
            self.save_summary_download,
            self.save_loads_download,
            self.plot_area_probe,
            sizing_mode="stretch_width",
            css_classes=["pycopter-toolbar"],
        )

        rotor_column = pn.Column(
            self._panel("Rotor Parameters", self._rotor_controls()),
            width=INPUT_COLUMN_WIDTH,
            sizing_mode="stretch_height",
            scroll=True,
            css_classes=["pycopter-column"],
        )
        setup_column = pn.Column(
            self._panel("Calculation Parameters", self._calculation_controls()),
            self._panel("Propulsion Estimation", self._propulsion_controls()),
            width=INPUT_COLUMN_WIDTH,
            sizing_mode="stretch_height",
            scroll=True,
            css_classes=["pycopter-column"],
        )
        results_column = pn.Column(
            self._plot_controls(),
            self.result_tabs,
            sizing_mode="stretch_both",
            css_classes=["pycopter-results"],
        )
        return pn.Column(
            toolbar,
            pn.Row(
                rotor_column,
                setup_column,
                results_column,
                sizing_mode="stretch_both",
                css_classes=["pycopter-body"],
            ),
            self.log_splitter,
            self.log_region,
            css_classes=["pycopter-shell"],
            sizing_mode="stretch_both",
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
        self.save_plot_download = pn.widgets.FileDownload(
            callback=self._save_plot_callback,
            filename="pycopter_plot.png",
            label="Save Plot",
            width=110,
        )
        self.save_summary_download = pn.widgets.FileDownload(
            callback=self._save_summary_callback,
            filename="pycopter_summary.csv",
            label="Save Summary",
            width=125,
        )
        self.save_loads_download = pn.widgets.FileDownload(
            callback=self._save_loads_callback,
            filename="pycopter_blade_loads.csv",
            label="Save Loads",
            width=115,
        )

        self.airfoil = pn.widgets.TextInput(label="Airfoil", value=cfg["airfoil"], width=390)
        self.new_polars = pn.widgets.Checkbox(
            label="Run XFOIL For Missing Polars",
            value=cfg["new_polars"],
            css_classes=["pycopter-xfoil-checkbox"],
        )
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
        self.headspeed_rpm = pn.widgets.FloatInput(label="Upper / Ref RPM [rpm]", value=cfg["headspeed_rpm"], start=100.0, end=100000.0, step=50.0, width=185)
        self.tip_speed_mach = pn.widgets.FloatInput(label="Tip Speed Mach", value=cfg["tip_speed_mach"], start=0.02, end=0.9, step=0.01, width=135)
        self.init_rotor_btn = pn.widgets.Button(label="Initialize Rotor", width=390)

        self.geometry_mode = pn.widgets.Select(
            label="Geometry Source",
            options={"Uniform root/tip controls": "uniform", "Blade control point table": "station_table"},
            value=cfg["geometry_mode"],
            width=390,
        )
        self.station_table = pn.widgets.Tabulator(
            pd.DataFrame(DEFAULT_STATION_ROWS),
            show_index=False,
            height=220,
            width=400,
            layout="fit_columns",
            titles={
                "r_over_R": "r/R",
                "chord_m": "Chord [m]",
                "twist_deg": "Twist [deg]",
                "airfoil": "Airfoil Override",
                "pitch_axis_frac": "Moment Axis [c]",
            },
            editors={
                "r_over_R": {"type": "number", "min": 0.02, "max": 1.0, "step": 0.01},
                "chord_m": {"type": "number", "min": 0.001, "max": 2.5, "step": 0.001},
                "twist_deg": {"type": "number", "min": -30.0, "max": 30.0, "step": 0.25},
                "airfoil": "input",
                "pitch_axis_frac": {"type": "number", "min": 0.0, "max": 1.0, "step": 0.01},
            },
        )
        self.reset_stations_btn = pn.widgets.Button(label="Reset Geometry Points From Root/Tip Twist", width=390)

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

        self.coaxial_spacing = pn.widgets.FloatInput(label="Rotor Spacing z/R", value=cfg["coaxial_spacing_ratio"], start=0.05, end=1.5, step=0.05, width=185)
        self.coaxial_trim_mode = pn.widgets.Select(
            label="Coaxial Trim Objective",
            options={
                "Torque-balanced hover": "torque_balance",
                "Equal rotor thrust": "equal_thrust",
                "Equal collective pitch": "equal_collective",
            },
            value=cfg["coaxial_trim_mode"],
            width=390,
        )
        self.lower_rotor_speed_ratio = pn.widgets.FloatInput(
            label="Lower/Upper RPM Ratio",
            value=cfg["lower_rotor_speed_ratio"],
            start=0.25,
            end=3.0,
            step=0.01,
            width=185,
        )
        self.lower_headspeed_rpm = pn.widgets.FloatInput(
            label="Lower RPM [rpm]",
            value=float(cfg["headspeed_rpm"]) * float(cfg["lower_rotor_speed_ratio"]),
            start=0.0,
            end=300000.0,
            step=1.0,
            width=185,
            disabled=True,
        )
        self.lower_collective_offset = pn.widgets.FloatInput(label="Lower Collective Bias [deg]", value=cfg["lower_collective_offset_deg"], start=-10.0, end=10.0, step=0.25, width=185)
        self.lower_rotor_scale = pn.widgets.FloatInput(label="Lower Geometry Scale", value=cfg["lower_rotor_scale"], start=0.5, end=1.5, step=0.01, width=185)

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

        self.plot_select = pn.widgets.Select(label="Selected Plot", options=[], sizing_mode="stretch_width")
        self.generate_plot_btn = pn.widgets.Button(
            label="Generate Plot",
            width=150,
            align="end",
            disabled=True,
        )

        self.plot_area_probe = ResultAreaProbe(width=0, height=0, margin=0)
        self.log_splitter = LogSplitter(
            height=LOG_GRIP_HEIGHT,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.output_log = pn.widgets.Terminal(
            output="",
            height=LOG_HEIGHT - LOG_TERMINAL_INSET,
            sizing_mode="stretch_width",
            options={
                "convertEol": True,
                "cursorBlink": False,
                "disableStdin": True,
                "fontFamily": 'Consolas, "Courier New", monospace',
                "fontSize": 12,
                "scrollback": 1000,
                "theme": {
                    "background": "#121416",
                    "foreground": "#d9efe9",
                    "cursor": "#4db6ac",
                    "selectionBackground": "#264f78",
                },
            },
            css_classes=["pycopter-log"],
        )
        self.log_region = pn.Column(
            self.output_log,
            height=LOG_HEIGHT,
            sizing_mode="stretch_width",
            css_classes=["pycopter-panel", "pycopter-log-region"],
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
        )
        self.load_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            sizing_mode="stretch_both",
            layout="fit_data_table",
            selectable=True,
            configuration=RESULT_TABLE_CONFIGURATION,
            css_classes=["pycopter-result-table", "pycopter-load-table"],
        )
        self.load_table_container = pn.Column(
            self.load_table,
            sizing_mode="stretch_both",
            css_classes=["pycopter-table-scroll"],
        )
        self.plot_pane = pn.pane.Matplotlib(
            self.plot_fig,
            sizing_mode="stretch_both",
            align="start",
            tight=False,
            dpi=PLOT_RENDER_DPI,
            css_classes=["pycopter-plot-frame"],
        )
        self.result_tabs = pn.Tabs(
            ("Plot", self.plot_pane),
            ("Summary", self.summary_table),
            ("Blade Element Loads", self.load_table_container),
            dynamic=False,
            sizing_mode="stretch_both",
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
        self.headspeed_rpm.param.watch(lambda _: self._update_lower_rpm_display(), "value")
        self.lower_rotor_speed_ratio.param.watch(lambda _: self._update_lower_rpm_display(), "value")
        self.geometry_mode.param.watch(lambda _: self._sync_enabled_state(), "value")
        self.plot_area_probe.param.watch(
            lambda *_: self._on_plot_area_resized(),
            ["width_px", "height_px"],
        )
        self.log_splitter.param.watch(lambda *_: self._on_log_resized(), "height_px")

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
            self._geometry_controls(),
            pn.layout.Divider(),
            self.init_rotor_btn,
            width=410,
        )

    def _coaxial_controls(self) -> pn.Column:
        return pn.Column(
            self.coaxial_trim_mode,
            self._two_col(self.coaxial_spacing, self.lower_rotor_speed_ratio),
            self._two_col(self.lower_headspeed_rpm, self.lower_collective_offset),
            self._two_col(self.lower_rotor_scale),
            width=410,
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
                ("Coaxial Settings", self._coaxial_controls()),
                active=[],
                toggle=True,
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

    def _plot_controls(self) -> pn.Row:
        """Plot picker and trigger, shown directly above the result tabs."""
        return pn.Row(
            self.plot_select,
            self.generate_plot_btn,
            sizing_mode="stretch_width",
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
                "lower_rotor_speed_ratio": self.lower_rotor_speed_ratio.value,
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
        self.lower_rotor_speed_ratio.value = float(cfg["lower_rotor_speed_ratio"])
        self._update_lower_rpm_display()
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
        self.output_log.clear()

    def _clear_results(self) -> None:
        self.current_case = None
        self.initialized_rotor = None
        self.plot_fig = self._blank_figure("Initialize rotor, then calculate hover.")
        self._current_plot_name = ""
        self._plot_save_available = False
        self.plot_pane.object = self.plot_fig
        self._set_read_only_table_value(self.summary_table, pd.DataFrame(columns=["Metric", "Value"]))
        self._set_read_only_table_value(self.load_table, pd.DataFrame())
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
                "Upper Tip Speed: "
                f"{self.initialized_rotor.tip_speed_m_s:.3f} [m/s] | "
                f"Upper Tip Mach: {self.initialized_rotor.tip_speed_mach:.3f} | "
                f"Rotor Disk Area: {self.initialized_rotor.disk_area_m2:.4f} [m2] | "
                f"Solidity: {self.initialized_rotor.solidity:.4f}"
            )
            total_blades = self.initialized_rotor.num_blades * (2 if config["rotor_system_type"] == "coaxial" else 1)
            self._log(f"Blades per rotor: {self.initialized_rotor.num_blades} | Total blades: {total_blades}")
            if config["rotor_system_type"] == "coaxial":
                lower_rotor = build_rotor_spec(
                    config,
                    rows,
                    name="lower",
                    scale=float(config["lower_rotor_scale"]),
                    headspeed_ratio=float(config["lower_rotor_speed_ratio"]),
                    rotation_direction=-1,
                )
                self._log(
                    f"Upper RPM: {self.initialized_rotor.headspeed_rpm:.1f} [rpm] | "
                    f"Lower RPM: {lower_rotor.headspeed_rpm:.1f} [rpm] | "
                    f"Lower/Upper RPM Ratio: {float(config['lower_rotor_speed_ratio']):.3f} | "
                    f"Lower Tip Mach: {lower_rotor.tip_speed_mach:.3f}"
                )
            geometry_source = (
                "uniform root/tip controls"
                if config["geometry_mode"] == "uniform"
                else "blade control point table"
            )
            active_airfoils = sorted(
                {self.initialized_rotor.airfoil_at(station.r_over_R) for station in self.initialized_rotor.stations}
            )
            self._log(
                f"Geometry source: {geometry_source} | "
                f"Control points: {len(self.initialized_rotor.stations)} | "
                f"Solver blade elements per rotor: {config['blade_element_count']} | "
                f"Active airfoils: {', '.join(active_airfoils)}"
            )
        except Exception as err:
            self.initialized_rotor = None
            self.current_case = None
            self._log(f"ERROR - {err}")
        self._sync_enabled_state()
        self._update_plot_options()

    def _calculate_hover(self) -> None:
        try:
            config = self._current_config()
            case = run_hover_case(
                config,
                self._station_rows(),
                polar_provider=self._xfoil_provider_for_config(config),
            )
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
            upper_rpm = float(config["headspeed_rpm"])
            lower_rpm = upper_rpm * float(config["lower_rotor_speed_ratio"])
            self._log(
                f"Total Thrust: {result.total_thrust_N:.3f} [N] | "
                f"Total Shaft Power: {result.total_power_W / 1000.0:.3f} [kW] | "
                f"Interference Loss: {result.interference_loss_ratio:.3f}"
            )
            self._log(
                f"Upper RPM: {upper_rpm:.1f} [rpm] | "
                f"Lower RPM: {lower_rpm:.1f} [rpm] | "
                f"Lower/Upper RPM Ratio: {float(config['lower_rotor_speed_ratio']):.3f}"
            )
            self._log(
                f"Upper Required Collective: {result.upper.collective_pitch_deg:.3f} [deg] | "
                f"Lower Required Collective: {result.lower.collective_pitch_deg:.3f} [deg] | "
                f"Wake Velocity: {result.wake_velocity_m_s:.3f} [m/s]"
            )
            self._log(
                f"Upper Thrust: {result.upper.total_thrust_N:.3f} [N] | "
                f"Lower Thrust: {result.lower.total_thrust_N:.3f} [N] | "
                f"Upper/Lower Torque: {result.upper.total_torque_Nm:.4f}/{result.lower.total_torque_Nm:.4f} [Nm]"
            )
            self._log(
                "Aircraft Yaw Torque: "
                f"{result.net_aircraft_yaw_torque_Nm:+.4f} [Nm] "
                f"({self._yaw_direction(result.net_aircraft_yaw_torque_Nm)})"
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
            self._log(
                "Aircraft Yaw Torque: "
                f"{result.aircraft_yaw_torque_Nm:+.4f} [Nm] "
                f"({self._yaw_direction(result.aircraft_yaw_torque_Nm)})"
            )
        self._log_alpha_clamp_warning(case, config)
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

    def _log_alpha_clamp_warning(self, case: HoverCase, config: dict[str, Any]) -> None:
        clamped = []
        for label, hover in self._iter_hover_results(case):
            clamped.extend((label, load.alpha_deg) for load in hover.element_loads if load.alpha_clamped)
        if not clamped:
            return

        labels = sorted({label for label, _ in clamped})
        alphas = [alpha for _, alpha in clamped]
        self._log(
            "WARNING - "
            f"{len(clamped)} blade elements requested alpha outside the loaded polar range "
            f"[{float(config['polar_alpha_min_deg']):.1f}, {float(config['polar_alpha_max_deg']):.1f}] deg; "
            f"requested range was [{min(alphas):.1f}, {max(alphas):.1f}] deg on {', '.join(labels)}. "
            "Coefficients were clamped. Expand XFOIL Alpha Min/Max and enable Run XFOIL For Missing Polars if you want new data."
        )

    def _update_summary_table(self, case: HoverCase, config: dict[str, Any]) -> None:
        rows: list[dict[str, str]] = [
            {"Metric": "Rotor System", "Value": "Coaxial" if case.system_type == "coaxial" else "Single rotor"},
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
            hover_results = [("Upper", result.upper), ("Lower", result.lower)]
        else:
            hover_results = [("Rotor", result)]

        for label, hover in hover_results:
            rows.extend(
                [
                    {"Metric": f"{label} Required Collective [deg]", "Value": f"{hover.collective_pitch_deg:.3f}"},
                    {"Metric": f"{label} Torque [Nm]", "Value": f"{hover.total_torque_Nm:.4f}"},
                    {"Metric": f"{label} Aircraft Yaw Torque [Nm]", "Value": f"{hover.aircraft_yaw_torque_Nm:+.4f}"},
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
        self._set_read_only_table_value(self.summary_table, pd.DataFrame(rows))

    def _yaw_direction(self, yaw_torque_Nm: float) -> str:
        if abs(yaw_torque_Nm) < 1e-9:
            return "balanced"
        if yaw_torque_Nm > 0.0:
            return "CCW / left yaw (+)"
        return "CW / right yaw (-)"

    def _update_load_table(self, case: HoverCase) -> None:
        result = case.result
        if isinstance(result, CoaxialHoverResult):
            upper = pd.DataFrame(load_rows_for_result(result.upper))
            upper.insert(0, "rotor", "upper")
            lower = pd.DataFrame(load_rows_for_result(result.lower))
            lower.insert(0, "rotor", "lower")
            frame = pd.concat([upper, lower], ignore_index=True)
        else:
            frame = pd.DataFrame(load_rows_for_result(result))
        self._set_read_only_table_value(self.load_table, frame)

    def _set_read_only_table_value(self, table: pn.widgets.Tabulator, frame: pd.DataFrame) -> None:
        table.value = frame
        table.editors = {column: None for column in frame.columns}

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
        except Exception as err:
            self._log(f"ERROR - {err}")

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
            self.plot_fig = self._blank_figure("Initialize rotor, then calculate hover.")
            self.plot_pane.object = self.plot_fig
            plt.close(superseded)

    def _on_log_resized(self) -> None:
        """Commit a dragged log height so the terminal re-flows its rows.

        The drag itself only changes inline CSS, which reflows the layout but
        leaves the terminal rendering at its old row count.
        """
        height = max(LOG_MIN_HEIGHT, int(self.log_splitter.height_px))
        self.log_region.height = height
        self.output_log.height = max(40, height - LOG_TERMINAL_INSET)

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
        cfg = self._current_config()
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
        if not isinstance(case.result, CoaxialHoverResult):
            raise ValueError("Interference loss spacing sweep requires a coaxial result.")

        self._log(
            "Calculating coaxial spacing sweep for interference loss. "
            f"Range: z/R {COAXIAL_SPACING_SWEEP_MIN:.2f}..{COAXIAL_SPACING_SWEEP_MAX:.2f}."
        )
        base_config = self._current_config()
        station_rows = self._station_rows()
        provider = self._xfoil_provider_for_config(base_config)
        spacings = np.linspace(
            COAXIAL_SPACING_SWEEP_MIN,
            COAXIAL_SPACING_SWEEP_MAX,
            COAXIAL_SPACING_SWEEP_POINTS,
        )
        losses = []
        powers_kW = []
        yaw_torques = []
        for spacing in spacings:
            sweep_case = run_hover_case(
                {**base_config, "rotor_system_type": "coaxial", "coaxial_spacing_ratio": float(spacing)},
                station_rows,
                polar_provider=provider,
            )
            result = sweep_case.result
            if not isinstance(result, CoaxialHoverResult):
                raise ValueError("Spacing sweep generated a non-coaxial result.")
            losses.append(result.interference_loss_ratio)
            powers_kW.append(result.total_power_W / 1000.0)
            yaw_torques.append(result.net_aircraft_yaw_torque_Nm)

        current_spacing = float(base_config["coaxial_spacing_ratio"])
        fig, ax = plt.subplots(figsize=self._figure_size())
        ax.plot(spacings, losses, marker="o", label="Interference Loss")
        ax.axvline(current_spacing, color=THEME["warning"], linestyle="--", linewidth=1.1, label="Current z/R")
        ax.set_title("Coaxial Spacing Sweep")
        ax.set_xlabel("Coaxial Spacing z/R")
        ax.set_ylabel("Interference Loss")
        ax.grid(True)
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
        self._log(
            "Spacing sweep complete: "
            f"{COAXIAL_SPACING_SWEEP_POINTS} points from z/R {COAXIAL_SPACING_SWEEP_MIN:.2f} "
            f"to {COAXIAL_SPACING_SWEEP_MAX:.2f}; "
            f"interference loss range {min(losses):.3f}..{max(losses):.3f}; "
            f"power range {min(powers_kW):.3f}..{max(powers_kW):.3f} [kW]; "
            f"max |yaw torque| {max(abs(value) for value in yaw_torques):.4f} [Nm]."
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

    def _twin_axis(self, ax, color: str, outward_points: float = 0.0):
        """Add a right-hand y-axis that shares x with ``ax``.

        Quantities of different units or magnitudes each get their own axis, so
        a small one is not flattened onto the scale of a large one.
        """
        twin = ax.twinx()
        twin._pycopter_twin = True
        self._accent_axis(twin, color, side="right")
        if outward_points:
            twin.spines["right"].set_position(("outward", outward_points))
            twin.spines["right"].set_visible(True)
        return twin

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
        """Plot per-element quantities against r/R, one y-axis per quantity.

        ``series`` is a sequence of ``(column, axis_label, legend_label)``,
        where ``column`` is either a load-table column name or a callable that
        derives a series from the load-table frame. Colour identifies the
        quantity and line style identifies the rotor, so a coaxial case keeps
        the same colour coding as a single rotor.
        """
        fig, base_ax = plt.subplots(figsize=self._figure_size())
        colors = [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(len(series))]
        axes = [self._accent_axis(base_ax, colors[0])]
        for index in range(1, len(series)):
            axes.append(
                self._twin_axis(base_ax, colors[index], outward_points=AXIS_OFFSET_POINTS * (index - 1))
            )

        rotors = list(self._iter_hover_results(case))
        marker_slots = max(1, len(series) * len(rotors))
        for rotor_index, (rotor_label, hover) in enumerate(rotors):
            rows = pd.DataFrame(load_rows_for_result(hover))
            linestyle = self._rotor_linestyle(rotor_index)
            prefix = f"{rotor_label} " if len(rotors) > 1 else ""
            marker_step = max(1, len(rows) // SERIES_MARKER_COUNT)
            for index, (ax, color, (column, _, legend_label)) in enumerate(zip(axes, colors, series)):
                values = column(rows) if callable(column) else rows[column]
                # Stagger each curve's markers along the span so quantities that
                # trace the same shape, such as Reynolds and Mach, stay legible.
                slot = index * len(rotors) + rotor_index
                ax.plot(
                    rows["r_over_R"],
                    values,
                    color=color,
                    linestyle=linestyle,
                    marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
                    markersize=4.5,
                    markevery=((slot * marker_step) // marker_slots, marker_step),
                    label=f"{prefix}{legend_label}",
                )

        for ax, (_, axis_label, _) in zip(axes, series):
            ax.set_ylabel(axis_label)
        base_ax.set_xlabel("r/R")
        base_ax.set_title(title)
        base_ax.grid(True)
        self._combined_legend(base_ax, *axes[1:])
        return self._finish_plot(fig)

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
        if isinstance(case.result, CoaxialHoverResult):
            return [("Upper", case.result.upper), ("Lower", case.result.lower)]
        return [("Rotor", case.result)]

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
        for widget in (
            self.coaxial_spacing,
            self.coaxial_trim_mode,
            self.lower_rotor_speed_ratio,
            self.lower_collective_offset,
            self.lower_rotor_scale,
        ):
            widget.disabled = not is_coaxial
        self.lower_headspeed_rpm.disabled = True
        self.headspeed_rpm.disabled = self.headspeed_input_mode.value != "rpm"
        self.tip_speed_mach.disabled = self.headspeed_input_mode.value != "tip_mach"
        self.station_table.disabled = self.geometry_mode.value != "station_table"
        self.calc_hover_btn.disabled = self.initialized_rotor is None
        self.generate_plot_btn.disabled = self.initialized_rotor is None and self.current_case is None
        self.calc_forward_btn.disabled = self.current_case is None or self.current_case.system_type != "single"

    def _update_lower_rpm_display(self) -> None:
        self.lower_headspeed_rpm.value = float(self.headspeed_rpm.value) * float(self.lower_rotor_speed_ratio.value)

    def _xfoil_provider_for_config(self, config: dict[str, Any]):
        key = self._xfoil_provider_cache_key(config)
        if self._xfoil_provider is None or self._xfoil_provider_key != key:
            self._cleanup_xfoil_provider()
            self._xfoil_provider = build_xfoil_provider(config)
            self._xfoil_provider_key = key
        return self._xfoil_provider

    def _xfoil_provider_cache_key(self, config: dict[str, Any]) -> tuple[Any, ...]:
        return (
            bool(config["new_polars"]),
            float(config["polar_alpha_min_deg"]),
            float(config["polar_alpha_max_deg"]),
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

    def _reset_station_rows_from_uniform(self) -> None:
        cfg = self._current_config()
        self.station_table.value = pd.DataFrame(station_rows_from_uniform(cfg))
        self._log("Geometry control points reset from root/tip twist inputs.")

    def _log(self, text: str) -> None:
        prefix = datetime.now().strftime("%H:%M:%S")
        self.output_lines.append("" if text == "" else f"[{prefix}] {text}")
        if len(self.output_lines) > 300:
            self.output_lines = self.output_lines[-300:]
            self.output_log.clear()
            self.output_log.write("\n".join(self.output_lines) + "\n")
            return
        self.output_log.write(("\n" if text == "" else f"[{prefix}] {text}\n"))


def create_app():
    """Create a fresh Panel app session."""
    return PycopterWebApp().view()
