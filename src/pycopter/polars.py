"""Airfoil polar providers used by the BEMT solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .xfoil import MAX_XFOIL_ALPHA_DEG, Xfoil, get_repo_root, normalize_airfoil_name


@dataclass(frozen=True)
class AirfoilCoefficients:
    """Section coefficients at one local angle of attack and flow condition."""

    cl: float
    cd: float
    cm: float = 0.0
    alpha_deg: float = 0.0
    reynolds: float = 0.0
    mach: float = 0.0
    alpha_clamped: bool = False


class PolarProvider(Protocol):
    """Protocol for any source of Cl/Cd/Cm data."""

    def get_coefficients(
        self,
        airfoil: str,
        alpha_deg: float,
        reynolds: float,
        mach: float,
    ) -> AirfoilCoefficients:
        """Return section coefficients for the requested local flow state."""


@dataclass(frozen=True)
class AirfoilPolar:
    """Interpolated polar table for one airfoil/Re/Mach condition."""

    airfoil: str
    reynolds: float
    mach: float
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray

    @classmethod
    def from_xfoil_table(
        cls,
        airfoil: str,
        reynolds: float,
        mach: float,
        table: np.ndarray,
    ) -> "AirfoilPolar":
        table = np.atleast_2d(table)
        if table.shape[1] < 3:
            raise ValueError("XFOIL polar table must include alpha, Cl, and Cd columns.")

        alpha = table[:, 0].astype(float)
        cl = table[:, 1].astype(float)
        cd = table[:, 2].astype(float)
        cm = table[:, 4].astype(float) if table.shape[1] > 4 else np.zeros_like(alpha)
        order = np.argsort(alpha)
        return cls(
            airfoil=normalize_airfoil_name(airfoil),
            reynolds=float(reynolds),
            mach=float(mach),
            alpha_deg=alpha[order],
            cl=cl[order],
            cd=cd[order],
            cm=cm[order],
        )

    def coefficients_at(self, alpha_deg: float) -> AirfoilCoefficients:
        clamped_alpha = float(np.clip(alpha_deg, self.alpha_deg[0], self.alpha_deg[-1]))
        return AirfoilCoefficients(
            cl=float(np.interp(clamped_alpha, self.alpha_deg, self.cl)),
            cd=float(np.interp(clamped_alpha, self.alpha_deg, self.cd)),
            cm=float(np.interp(clamped_alpha, self.alpha_deg, self.cm)),
            alpha_deg=clamped_alpha,
            reynolds=self.reynolds,
            mach=self.mach,
            alpha_clamped=clamped_alpha != float(alpha_deg),
        )


@dataclass
class LinearPolarProvider:
    """Deterministic analytic polar for tests and early design studies."""

    lift_slope_per_rad: float = 2.0 * np.pi
    zero_lift_alpha_deg: float = 0.0
    cd0: float = 0.01
    induced_drag_factor: float = 0.01
    cm0: float = 0.0
    cl_max: float | None = 1.4

    def get_coefficients(
        self,
        airfoil: str,
        alpha_deg: float,
        reynolds: float,
        mach: float,
    ) -> AirfoilCoefficients:
        alpha_rad = np.deg2rad(alpha_deg - self.zero_lift_alpha_deg)
        cl = self.lift_slope_per_rad * alpha_rad
        if self.cl_max is not None:
            cl = float(np.clip(cl, -self.cl_max, self.cl_max))
        cd = self.cd0 + self.induced_drag_factor * cl**2
        return AirfoilCoefficients(
            cl=float(cl),
            cd=float(max(cd, 0.0)),
            cm=float(self.cm0),
            alpha_deg=float(alpha_deg),
            reynolds=float(reynolds),
            mach=float(mach),
        )


class XfoilPolarProvider:
    """
    XFOIL-backed polar source with coarse Re/Mach binning.

    The binning avoids regenerating a polar for tiny local-flow changes during
    the nonlinear BEMT solve. Generated polar files are stored per airfoil/Re/Mach
    condition, then cached in memory for repeated lookup during the current solve.
    """

    def __init__(
        self,
        new_polar: bool = True,
        alpha_min_deg: float = -10.0,
        alpha_max_deg: float = MAX_XFOIL_ALPHA_DEG,
        reynolds_bin: float = 25000.0,
        mach_bin: float = 0.02,
        timeout: int = 60,
        cache_directory: str | Path | None = None,
    ):
        self.new_polar = new_polar
        self.alpha_min_deg = alpha_min_deg
        self.alpha_max_deg = min(alpha_max_deg, MAX_XFOIL_ALPHA_DEG)
        if self.alpha_min_deg > self.alpha_max_deg:
            raise ValueError(
                f"alpha_min_deg must be <= {MAX_XFOIL_ALPHA_DEG} deg for XFOIL runs."
            )
        self.reynolds_bin = reynolds_bin
        self.mach_bin = mach_bin
        self.timeout = timeout
        self.cache_directory = (
            Path(cache_directory)
            if cache_directory is not None
            else get_repo_root() / "data" / "XFOIL6.99" / "polars"
        )
        self._cache: dict[tuple[str, float, float], AirfoilPolar] = {}

    def get_coefficients(
        self,
        airfoil: str,
        alpha_deg: float,
        reynolds: float,
        mach: float,
    ) -> AirfoilCoefficients:
        normalized = normalize_airfoil_name(airfoil)
        reynolds_key = self._round_to_positive_bin(
            max(reynolds, 1000.0),
            self.reynolds_bin,
        )
        mach_key = self._round_to_bin(max(mach, 0.0), self.mach_bin)
        key = (normalized, reynolds_key, mach_key)
        if key not in self._cache:
            self._cache[key] = self._generate_polar(normalized, reynolds_key, mach_key)
        return self._cache[key].coefficients_at(alpha_deg)

    def _round_to_bin(self, value: float, bin_size: float) -> float:
        if bin_size <= 0:
            return float(value)
        return float(round(value / bin_size) * bin_size)

    def _round_to_positive_bin(self, value: float, bin_size: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive for positive binning.")
        if bin_size <= 0:
            return float(value)
        return float(max(round(value / bin_size) * bin_size, bin_size))

    def _generate_polar(self, airfoil: str, reynolds: float, mach: float) -> AirfoilPolar:
        cache_path = self._cache_file_path(airfoil, reynolds, mach)
        xfoil = Xfoil(new_polar=self.new_polar, timeout=self.timeout)
        self._configure_xfoil_output(xfoil, cache_path)

        if not self.new_polar and cache_path.exists():
            return AirfoilPolar.from_xfoil_table(
                airfoil,
                reynolds,
                mach,
                xfoil.read_polar(),
            )

        if not xfoil.simulate(
            airfoil,
            mach,
            reynolds,
            alpha_min_deg=self.alpha_min_deg,
            alpha_max_deg=self.alpha_max_deg,
        ):
            raise RuntimeError(xfoil.error_message)
        return AirfoilPolar.from_xfoil_table(airfoil, reynolds, mach, xfoil.read_polar())

    def _cache_file_path(self, airfoil: str, reynolds: float, mach: float) -> Path:
        reynolds_part = f"re{int(round(reynolds))}"
        mach_part = f"m{self._safe_float_token(mach)}"
        alpha_part = (
            f"a{self._safe_float_token(self.alpha_min_deg)}_"
            f"{self._safe_float_token(self.alpha_max_deg)}"
        )
        filename = f"{normalize_airfoil_name(airfoil)}_{reynolds_part}_{mach_part}_{alpha_part}.txt"
        return self.cache_directory / filename

    def _safe_float_token(self, value: float) -> str:
        text = f"{value:.4g}".replace("-", "neg").replace(".", "p")
        return text.replace("+", "")

    def _configure_xfoil_output(self, xfoil: Xfoil, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xfoil.output_path = output_path
        repo_root = getattr(xfoil, "repo_root", None)
        if repo_root is None:
            xfoil.output_path_for_xfoil = output_path.as_posix()
            return
        try:
            relative_path = output_path.relative_to(repo_root)
            xfoil.output_path_for_xfoil = relative_path.as_posix()
        except ValueError:
            xfoil.output_path_for_xfoil = output_path.as_posix()
