# Project Instructions

## Tech Stack
- Python application using a `src/` layout with a lightweight `requirements.txt`.
- Runtime dependencies: Panel, matplotlib, numpy, scipy, and Pillow.
- Core app domains are rotorcraft performance modeling, XFOIL polar generation, browser dashboard wiring, and plots.
- The checked-in XFOIL executable is Windows-specific and lives under `data/XFOIL6.99/`.

## Code Style
- Follow the existing Python style: 4-space indentation, snake_case functions/variables, PascalCase classes.
- Treat `src/pycopter/pycopter.py`, `src/pycopter/utils.py`, `src/gui/state.py`, and `src/gui/dashboard.py` as the main hand-written code.
- Treat `src/gui_old/` as the legacy PyQt5 implementation kept for reference.
- Prefer explicit imports in new code. Existing wildcard exports/imports are historical.
- Keep numerical units clear in names, docstrings, or output text because the model mixes kg, N, m/s, km/hr, SHP, and kW-style quantities.
- Keep numerical/model logic independent from GUI code so the same calculation layer can power the Panel dashboard, notebooks, or tests.

## Testing
- Current tests are under `tests/` and use stdlib `unittest`.
- Run tests with:
  `$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m unittest discover -v`
- Run a syntax-only check with:
  `$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src tests`
- There is currently no `pyproject.toml`, `tox.ini`, or `Makefile`.

## Build & Run
- Create or refresh the local virtual environment with:
  `python -m venv .venv`
- Activate it in PowerShell with:
  `.\.venv\Scripts\Activate.ps1`
- Install dependencies with:
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Run the Panel dashboard from the repo root with `PYTHONPATH=src` set:
  `python src/main.py`
- Or serve it explicitly with:
  `panel serve src/gui/dashboard.py --show`

## Project Structure
- `src/main.py` launches the Panel browser dashboard.
- `src/gui/state.py` contains dashboard settings, preset mapping, validation, and metric formatting.
- `src/gui/dashboard.py` builds the Panel workbench, callbacks, text output, and plots.
- `src/gui_old/` contains the old PyQt5 GUI implementation.
- `src/pycopter/pycopter.py` contains `Rotor` and placeholder domain classes.
- `src/pycopter/utils.py` contains polar interpolation, XFOIL-facing `Polar`, Reynolds/Wald helpers, and reference-area utilities.
- `src/pycopter/xfoil.py` starts XFOIL and reads generated polar data from `data/XFOIL6.99/polar.txt`.
- `data/` contains XFOIL binaries and reference data.
- `tutorials/presets/` contains sample helicopter configurations and generated figures.

## Conventions
- Most commit messages are short imperative or descriptive summaries.
- Dashboard actions are centralized in `PyCopterDashboard`; pure mapping/formatting belongs in `gui.state`.
- `Rotor.hover()` establishes state needed by `Rotor.forward_flight()`. Call hover first when adding direct model usage.
- XFOIL output is a mutable file in `data/XFOIL6.99/polar.txt`; tests and examples should account for that side effect.
- Many printed values are duplicated between model methods and GUI output. When changing formulas, check both the stored attributes and the displayed text.
- This is a solo-maintained project; prefer useful local scripts and simple docs over public-package boilerplate.
