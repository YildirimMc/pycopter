@echo off
setlocal
cd /d "%~dp0"

rem One-click launcher for the PyCopter Web UI. Creates the virtual
rem environment and installs dependencies on first run, then starts the server.

set "VENV_PYTHON=.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment in .venv ...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo Install Python 3.11 or newer from https://www.python.org/downloads/
        echo and make sure it is on PATH, then run this script again.
        goto :fail
    )

    echo Installing dependencies from requirements.txt ...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed.
        echo If mpi4py was the failure, install the Microsoft MPI runtime first:
        echo   https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
        goto :fail
    )
)

echo Starting PyCopter Web UI ...
set "PYTHONPATH=src"
"%VENV_PYTHON%" webui.py
if errorlevel 1 goto :fail

endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1
