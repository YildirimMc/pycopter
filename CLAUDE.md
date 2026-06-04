# Project Instructions

## Tech Stack
- Python application using a `src/` layout, without packaging scaffolding or dependency manifests for now.
- Runtime dependencies are inferred from imports: PyQt5, matplotlib, numpy, scipy, and Pillow.
- Core app domains are rotorcraft performance modeling, XFOIL polar generation, GUI wiring, and plots.
- The checked-in XFOIL executable is Windows-specific and lives under `data/XFOIL6.99/`.

## Code Style
- Follow the existing Python style: 4-space indentation, snake_case functions/variables, PascalCase classes.
- Treat `src/pycopter/pycopter.py`, `src/pycopter/utils.py`, and `src/gui/interface.py` as the main hand-written code.
- Treat `src/gui/pycopterui.py` as generated legacy PyQt5 code; avoid investing in it until the future GUI direction is chosen.
- Prefer explicit imports in new code. Existing wildcard exports/imports are historical.
- Keep numerical units clear in names, docstrings, or output text because the model mixes kg, N, m/s, km/hr, SHP, and kW-style quantities.
- Keep numerical/model logic independent from GUI code so the same calculation layer can power PyQt, a browser UI, notebooks, or tests.

## Testing
- Current tests are under `tests/` and use `unittest`, but they appear stale relative to the current `Rotor` API.
- A syntax-only check that does not require third-party packages is:
  `python -m compileall -q src tests`
- After choosing dependencies and installing them into `.venv`, use:
  `python -m pytest -q`
- There is currently no `pyproject.toml`, `requirements.txt`, `tox.ini`, or `Makefile`.

## Build & Run
- Create or refresh the local virtual environment with:
  `python -m venv .venv`
- Activate it in PowerShell with:
  `.\.venv\Scripts\Activate.ps1`
- Do not install packages unless the current task explicitly calls for dependency setup.
- With dependencies installed later, run the existing GUI from the repo root with `PYTHONPATH=src` set:
  `python src/main.py`

## Project Structure
- `src/main.py` launches the PyQt application.
- `src/gui/interface.py` wires UI buttons, file actions, plotting, and user inputs to the calculation model.
- `src/gui/input_checker.py` contains GUI validation helpers.
- `src/gui/pycopterui.py` is generated UI layout code.
- `src/pycopter/pycopter.py` contains `Rotor` and placeholder domain classes.
- `src/pycopter/utils.py` contains polar interpolation, XFOIL-facing `Polar`, Reynolds/Wald helpers, and reference-area utilities.
- `src/pycopter/xfoil.py` starts XFOIL and reads generated polar data from `data/XFOIL6.99/polar.txt`.
- `data/` contains XFOIL binaries and reference data.
- `tutorials/presets/` contains sample helicopter configurations and generated figures.

## Conventions
- Most commit messages are short imperative or descriptive summaries.
- GUI actions are centralized in `Interface`; add new buttons/actions there and keep generated UI code separate where possible.
- `Rotor.hover()` establishes state needed by `Rotor.forward_flight()`. Call hover first when adding direct model usage.
- XFOIL output is a mutable file in `data/XFOIL6.99/polar.txt`; tests and examples should account for that side effect.
- Many printed values are duplicated between model methods and GUI output. When changing formulas, check both the stored attributes and the displayed text.
- This is a solo-maintained project; prefer useful local scripts and simple docs over public-package boilerplate.
