# Project Instructions

## Tech Stack
- Python application using a `src/` layout. Dependencies are pinned by range in `requirements.txt` (runtime) and `requirements-dev.txt` (Playwright, plus PyQt5 for the legacy GUI). There is still no packaging scaffolding, and none is wanted.
- Runtime dependencies: panel, matplotlib, numpy, pandas, scipy, Pillow, and mpi4py. PyQt5 is needed only by the legacy desktop GUI.
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
- Coaxial hover is modeled as upper and lower rotor solves with a first-order upper-wake velocity/contraction estimate applied to the lower rotor. Treat it as a better initial estimator than total-blade-count shortcuts, not as a full free-wake model.
- For scientific traceability, do not reintroduce hardcoded mean-drag curves as the primary model. Use local polar lookup or an explicit documented calibration path.
- XFOIL polar generation is treated as scratch data. `XfoilPolarProvider` defaults to a provider-owned temporary cache under ignored `data/XFOIL6.99/tmp/`; generated polar files must not be committed.
- `HoverSolver` prefetches radial airfoil/Re/Mach bins through compatible polar providers before element iterations. `XfoilPolarProvider` distributes independent missing XFOIL jobs with `mpi4py.futures.MPIPoolExecutor(max_workers=8)` by default; missing MPI runtime support is an installation error, not a reason to silently run serial.
- User feedback has been explicit that broad refactors, hidden fallbacks, and test-only evidence are unacceptable for XFOIL/hover bugs. For this path, prioritize the smallest direct code change, install required runtime dependencies when possible, and verify by running the actual hover calculation while confirming concurrent XFOIL execution.

## Testing
- Tests live under `tests/` and use `unittest`. They cover the BEMT solver, XFOIL caching and parallel dispatch, real-world hover cases, the GUI calculation layer, and headless browser render/layout checks.
- Verification command:
  `$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m unittest discover -v`
- A syntax-only check that does not require third-party packages is:
  `python -m compileall -q src tests`
- Browser tests need `playwright` plus `python -m playwright install chromium`. They skip themselves when it is missing, so a missing browser must never be read as a passing layout.
- There is no `pyproject.toml`, `tox.ini`, or `Makefile`, and none is wanted.

## Build & Run
- One-click path, which also provisions `.venv` on first run:
  `start_webui.bat`
- Manual setup:
  `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Start the Web UI from the repo root (`webui.py` puts `src/` on `sys.path` itself):
  `.venv\Scripts\python.exe webui.py`
- Do not install packages unless the current task explicitly calls for dependency setup.
- The legacy PyQt GUI still runs with `PYTHONPATH=src` set once PyQt5 is installed:
  `python src/main.py`
- Regenerate README screenshots after UI changes:
  `PYTHONPATH=src .venv\Scripts\python.exe scripts/capture_screenshots.py`

## Project Structure
- `webui.py` launches the Panel Web UI; `start_webui.bat` is the one-click Windows wrapper.
- `src/gui/app.py` is the Web UI: widgets, layout, and all plot functions. `src/gui/calculations.py` maps GUI config dicts onto the typed solver API.
- `scripts/capture_screenshots.py` regenerates `docs/images/` for the README.
- `README.md` is the GitHub landing page and `docs/TUTORIAL.md` the Web UI walkthrough. Keep both current when GUI behavior changes.
- `src/main.py` launches the PyQt application if GUI modules are available.
- `src/gui_old/interface.py` wires legacy UI buttons, file actions, plotting, and user inputs to the calculation model.
- `src/gui_old/input_checker.py` contains legacy GUI validation helpers.
- `src/gui_old/pycopterui.py` is generated UI layout code.
- `src/pycopter/pycopter.py` contains the legacy-compatible `Rotor` facade and placeholder aircraft-level classes.
- `src/pycopter/models.py` contains typed rotor geometry, operating-point, solver-setting, and result dataclasses.
- `src/pycopter/bemt.py` contains the hover BEMT solver and first-order coaxial hover solver.
- `src/pycopter/polars.py` contains analytic and XFOIL-backed airfoil polar providers, including temporary polar caching and optional MPI prefetch.
- `src/pycopter/utils.py` contains legacy polar interpolation, XFOIL-facing `Polar`, Reynolds/Wald helpers, and reference-area utilities.
- `src/pycopter/xfoil.py` starts XFOIL and reads generated polar data. Its low-level default is `data/XFOIL6.99/polar.txt`, but new solver code should route through `XfoilPolarProvider` so per-condition temp cache paths are used.
- `data/` contains XFOIL binaries and reference data.
- `tutorials/presets/` contains sample helicopter configurations and generated figures.

## Web UI Layout Rules
- The dashboard is a fixed-height app shell sized to the viewport. The toolbar and run log stay pinned, and each input column scrolls internally. Never reintroduce a fixed pixel width/height on the shell, result tabs, tables, or plot pane: that is what clipped the layout on 1920x1080 and made expanding a collapsible section resize the page.
- Matplotlib figures are drawn at the size the plot frame actually has on screen. `ResultAreaProbe` measures it in the browser and syncs it back, because Panel does not report pane size to the server. Use `self._figure_size()` for every new figure, never a hardcoded `figsize`.
- Resizing the window does not re-run sweep plots, since those re-run the solver many times. CSS keeps the existing raster inside the frame until the next Generate Plot.
- Panel renders each component into its own shadow root, so browser-side helpers must walk shadow roots; a plain `document.querySelector` will not find dashboard elements.
- The output log is resizable upwards against the result area via `LogSplitter`. Its start height is its minimum. Bokeh does not run a layout pass for a plain height change, so the terminal must be refitted explicitly through the Panel terminal view or xterm keeps its old row count inside a taller box.
- Plot axis layout is decided from the data by `_axis_layout`: quantities within `SHARED_AXIS_MAX_RATIO` share a y-axis, a larger gap takes the right-hand axis, and a third scale gets its own stacked panel. Never reintroduce a third y-axis on an offset spine. Route per-element plots through `_plot_element_series`.
- Two identically shaped curves on opposing autoscaled axes draw on top of each other and hide one another; `_shapes_coincide` detects this and forces a panel split. Reynolds and Mach are the live case.
- `tests/test_web_gui_browser.py` locks this in at 1920x937, 2560x1329, and 1366x728, collapsed and expanded, plus a real drag of the log splitter. Run it after any layout change.

## Conventions
- Most commit messages are short imperative or descriptive summaries.
- GUI actions are centralized in `Interface`; add new buttons/actions there and keep generated UI code separate where possible.
- `Rotor.hover()` establishes state needed by `Rotor.forward_flight()`. Call hover first when adding direct model usage.
- XFOIL output is mutable scratch data. Legacy code may still touch `data/XFOIL6.99/polar.txt`; new solver code should use provider-managed ignored temp paths and tests should account for these side effects.
- Many printed values are duplicated between model methods and GUI output. When changing formulas, check both the stored attributes and the displayed text.
- This is a solo-maintained project; prefer useful local scripts and simple docs over public-package boilerplate.
