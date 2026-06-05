"""Top-level package for PyCopter."""

__author__ = """Mehar Can Yildirim"""
__email__ = 'yildirimmehar@gmail.com'
__version__ = '0.0.1'

from .bemt import HoverSolver, solve_coaxial_hover
from .models import (
    BladeStation,
    CoaxialHoverResult,
    CoaxialSpec,
    ElementLoad,
    HoverResult,
    HoverSolverSettings,
    OperatingPoint,
    RotorSpec,
)
from .polars import AirfoilCoefficients, AirfoilPolar, LinearPolarProvider, XfoilPolarProvider
from .pycopter import Body, Copter, Engine, Optimizer, Rotor, RotorImperial

__all__ = [
    "AirfoilCoefficients",
    "AirfoilPolar",
    "BladeStation",
    "Body",
    "CoaxialHoverResult",
    "CoaxialSpec",
    "Copter",
    "ElementLoad",
    "Engine",
    "HoverResult",
    "HoverSolver",
    "HoverSolverSettings",
    "LinearPolarProvider",
    "OperatingPoint",
    "Optimizer",
    "Rotor",
    "RotorImperial",
    "RotorSpec",
    "XfoilPolarProvider",
    "solve_coaxial_hover",
]
