# Project Instructions

## Tech Stack
- Python application using a `src/` layout. Dependencies are pinned by range in `requirements.txt` (runtime) and `requirements-dev.txt` (Playwright only). There is still no packaging scaffolding, and none is wanted.
- Runtime dependencies: panel, matplotlib, numpy, pandas, Pillow, and mpi4py. Keep the dependency list this short on purpose: scipy was dropped because its only use was one `interp1d` call, and PyQt5 is not a declared dev dependency because it is a ~140 MB Qt runtime needed only by the legacy desktop GUI. Install PyQt5 on demand to run `src/main.py`.
- Core app domains are rotorcraft performance modeling, XFOIL polar generation, GUI wiring, and plots.
- The checked-in XFOIL executable is Windows-specific and lives under `data/XFOIL6.99/`.
- The project is meant to be clone-and-run: `start_webui.bat` provisions `.venv` from `requirements.txt` on first run. Keep that path working when changing dependencies or entry points.

## Code Style
- Follow the existing Python style: 4-space indentation, snake_case functions/variables, PascalCase classes.
- Treat `src/pycopter/pycopter.py`, `src/pycopter/bemt.py`, `src/pycopter/models.py`, `src/pycopter/polars.py`, and `src/pycopter/utils.py` as the main hand-written calculation code.
- `src/gui/` holds the supported Panel Web UI and gets all new GUI work. `src/gui_old/` is the legacy PyQt desktop GUI, kept for reference only.
- Treat `src/gui_old/pycopterui.py` as generated legacy PyQt5 code. Its wiring is intact and verified; do not invest in it further.
- Prefer explicit imports in new code. Existing wildcard exports/imports are historical.
- Keep numerical units clear in names, docstrings, or output text because the model mixes kg, N, m/s, km/hr, SHP, and kW-style quantities.
- Keep numerical/model logic independent from GUI code so the same calculation layer can power PyQt, a browser UI, notebooks, or tests.
- Any new calculation input or output that the GUI must expose must be documented in `GUI.md` with key name, units, value range/default, short description, and Python API endpoint before or alongside implementation.

## Rotor Solver Design
- The primary rotor model is now a station-based BEMT hover solver in `src/pycopter/bemt.py`, with typed contracts in `src/pycopter/models.py` and polar providers in `src/pycopter/polars.py`.
- `RotorSpec`, `OperatingPoint`, `HoverSolverSettings`, `HoverSolver`, and `solve_coaxial_hover` are the preferred API for new code. The legacy `Rotor` class remains as a facade for existing GUI-style calls.
- Hover calculations integrate local blade-element lift/drag with annular momentum balance at each radial station. Local angle of attack, Reynolds number, Mach number, Prandtl root/tip loss, thrust, torque, power, and pitch moment are preserved in `HoverResult.element_loads`.
- `HoverResult` also carries structured warnings: `clamped_element_count`, `clamped_alpha_range_deg`, and `substituted_polar_bins`. The GUI drives its warning state from these, never by parsing log text. `HoverSolverSettings.element_spacing` defaults to `uniform`; `cosine` is a resolution choice only.
- Coaxial hover is modeled as upper and lower rotor solves with a first-order upper-wake velocity/contraction estimate applied to the lower rotor. Treat it as a better initial estimator than total-blade-count shortcuts, not as a full free-wake model.
- For scientific traceability, do not reintroduce hardcoded mean-drag curves as the primary model. Use local polar lookup or an explicit documented calibration path.
- XFOIL polar generation is treated as scratch data. `XfoilPolarProvider` defaults to a provider-owned temporary cache under ignored `data/XFOIL6.99/tmp/`; generated polar files must not be committed.
- `HoverSolver` prefetches radial airfoil/Re/Mach bins through compatible polar providers before element iterations. `XfoilPolarProvider` distributes independent missing XFOIL jobs with `mpi4py.futures.MPIPoolExecutor(max_workers=8)` by default; missing MPI runtime support is an installation error, not a reason to silently run serial.
- User feedback has been explicit that broad refactors, hidden fallbacks, and test-only evidence are unacceptable for XFOIL/hover bugs. For this path, prioritize the smallest direct code change, install required runtime dependencies when possible, and verify by running the actual hover calculation while confirming concurrent XFOIL execution.

## Testing
- Tests live under `tests/` and use `unittest`. They cover the BEMT solver, element spacing, the cancellable interference sweep, XFOIL caching, parallel dispatch and polar bin provenance, real-world hover cases, the GUI calculation layer, the session run store, and headless browser render/layout checks.
- Verification command:
  `$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m unittest discover -v`
- A syntax-only check that does not require third-party packages is:
  `python -m compileall -q src tests`
- Browser tests need `playwright` plus `python -m playwright install --only-shell`. They skip themselves when it is missing, so a missing browser must never be read as a passing layout.
- Headless launches go through `channel="chromium-headless-shell"` (`HEADLESS_CHANNEL` in `tests/test_web_gui_browser.py`, and the same literal in `scripts/capture_screenshots.py`). Playwright has no fallback from the `chromium` executable to the shell, so dropping the channel silently reintroduces the full ~410 MB chromium download as a requirement.
- There is no `pyproject.toml`, `tox.ini`, or `Makefile`, and none is wanted.

## Build & Run
- One-click path, which also provisions `.venv` on first run:
  `start_webui.bat`
- Manual setup:
  `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Start the Web UI from the repo root (`webui.py` puts `src/` on `sys.path` itself):
  `.venv\Scripts\python.exe webui.py`
- Do not install packages unless the current task explicitly calls for dependency setup.
- The legacy PyQt GUI still runs with `PYTHONPATH=src` set, once PyQt5 is installed by hand (`pip install "PyQt5>=5.15,<6"`; it is not in either requirements file):
  `python src/main.py`
- Regenerate README screenshots after UI changes:
  `PYTHONPATH=src .venv\Scripts\python.exe scripts/capture_screenshots.py`

## Project Structure
- `webui.py` launches the Panel Web UI; `start_webui.bat` is the one-click Windows wrapper.
- `src/gui/app.py` is the Web UI: shell, inspector, canvas, run rail, log, and all plot functions. `src/gui/calculations.py` maps GUI config dicts onto the typed solver API. `src/gui/runs.py` holds the session `RunRecord`/`RunStore` behind the run rail.
- `scripts/capture_screenshots.py` regenerates `docs/images/` for the README.
- `README.md` is the GitHub landing page and `docs/TUTORIAL.md` the Web UI walkthrough. Keep both current when GUI behavior changes.
- `src/main.py` launches the PyQt application if GUI modules are available.
- `src/gui_old/interface.py` wires legacy UI buttons, file actions, plotting, and user inputs to the calculation model.
- `src/gui_old/input_checker.py` contains legacy GUI validation helpers.
- `src/gui_old/pycopterui.py` is generated UI layout code.
- `src/pycopter/pycopter.py` contains the legacy-compatible `Rotor` facade and placeholder aircraft-level classes.
- `src/pycopter/models.py` contains typed rotor geometry, operating-point, solver-setting, and result dataclasses.
- `src/pycopter/bemt.py` contains the hover BEMT solver, the first-order coaxial hover solver, and `sweep_interference_loss` (cancellable, progress-reporting spacing sweep).
- `src/pycopter/polars.py` contains analytic and XFOIL-backed airfoil polar providers, including temporary polar caching, optional MPI prefetch, and per-bin provenance through `polar_bin_report`.
- `src/pycopter/utils.py` contains legacy polar interpolation, XFOIL-facing `Polar`, Reynolds/Wald helpers, and reference-area utilities.
- `src/pycopter/xfoil.py` starts XFOIL and reads generated polar data. Its low-level default is `data/XFOIL6.99/polar.txt`, but new solver code should route through `XfoilPolarProvider` so per-condition temp cache paths are used.
- `data/` contains XFOIL binaries and reference data.
- `tutorials/presets/` contains sample helicopter configurations and generated figures.

## Web UI Layout Rules
- The dashboard is a fixed-height app shell sized to the viewport: pinned command bar, optional warning bar, then a middle band of icon rail (60 px), inspector (320 px), canvas (flex, min 656 px), and run rail (384 px), then a drag rule and the pinned log. Never reintroduce a fixed pixel width/height on the shell, result tabs, tables, or plot pane: that is what clipped the layout on 1920x1080.
- The inspector is tab-gated. All six panels stay built and only the active one is `visible`, so every field keeps its value and round-trips through save/load whichever tab is showing. Switching tabs, expanding the log, and collapsing the inspector must not resize any other region.
- Panel renders each component into its own shadow root. A CSS rule written as `.container .bk-input` never reaches the widget. Rules that touch a widget's internals go in that widget's own `stylesheets`; rules that style a host element, or stay inside one component, go in `RAW_CSS`. Browser-side helpers must walk shadow roots; a plain `document.querySelector` will not find dashboard elements.
- Matplotlib figures are drawn at the size the plot frame actually has on screen. `ResultAreaProbe` measures it in the browser and syncs it back, because Panel does not report pane size to the server. Use `self._figure_size()` for every new figure, never a hardcoded `figsize`.
- Resizing the window does not re-run sweep plots, since those re-run the solver many times. CSS keeps the existing raster inside the frame until the next GENERATE.
- The output log is resizable upwards against the result area via `LogSplitter`. Its start height is its minimum (30 px, the status strip alone). Bokeh does not run a layout pass for a plain height change, so the terminal must be refitted explicitly through the Panel terminal view or xterm keeps its old row count inside a taller box.
- Plot axis layout is decided from the data by `_axis_layout`: quantities within `SHARED_AXIS_MAX_RATIO` share a y-axis, a larger gap takes the right-hand axis, and a third scale gets its own stacked panel. Never reintroduce a third y-axis on an offset spine. Route per-element plots through `_plot_element_series`.
- Two identically shaped curves on opposing autoscaled axes draw on top of each other and hide one another; `_shapes_coincide` detects this and forces a panel split. Reynolds and Mach are the live case.
- Overlaid runs are drawn under the active run, wider and translucent, so a coincident overlay reads as a halo rather than hiding it. Overlay data joins the same `_axis_layout` measurement.
- The coaxial `SHOWING` toggle drives `_iter_hover_results`, so metric strip, summary, loads, and plots always agree on which `HoverResult` they are reading.
- `tests/test_web_gui_browser.py` locks this in at 1920x937, 2560x1329, and 1600x728, across inspector tabs, plus inspector collapse and a real drag of the log splitter. Run it after any layout change.

## Conventions
- Most commit messages are short imperative or descriptive summaries.
- GUI actions are centralized in `Interface`; add new buttons/actions there and keep generated UI code separate where possible.
- `Rotor.hover()` establishes state needed by `Rotor.forward_flight()`. Call hover first when adding direct model usage.
- XFOIL output is mutable scratch data. Legacy code may still touch `data/XFOIL6.99/polar.txt`; new solver code should use provider-managed ignored temp paths and tests should account for these side effects.
- Many printed values are duplicated between model methods and GUI output. When changing formulas, check both the stored attributes and the displayed text.
- This is a solo-maintained project; prefer useful local scripts and simple docs over public-package boilerplate.
