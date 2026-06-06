"""Airfoil polar providers used by the BEMT solver."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Protocol

import numpy as np

from .xfoil import MAX_XFOIL_ALPHA_DEG, Xfoil, get_repo_root, normalize_airfoil_name

ParallelBackend = Literal["auto", "mpi", "serial"]


class _ShortTemporaryDirectory:
    """Provider-owned temp directory with XFOIL-compatible 8-char names."""

    def __init__(self, root: Path):
        self.path = self._create(root)
        self.name = str(self.path)

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def _create(self, root: Path) -> Path:
        for _ in range(100):
            path = root / f"px{uuid.uuid4().hex[:6]}"
            try:
                path.mkdir()
            except FileExistsError:
                continue
            return path
        raise RuntimeError("Could not create a unique XFOIL temporary directory.")


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

    def prepare_conditions(
        self,
        conditions: Iterable[tuple[str, float, float]],
    ) -> None:
        """Warm any polar cache needed for airfoil/Re/Mach lookup."""

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


@dataclass(frozen=True)
class XfoilPolarJobResult:
    """Serializable XFOIL worker result for one airfoil/Re/Mach polar."""

    airfoil: str
    reynolds: float
    mach: float
    table: list[list[float]]

    @property
    def key(self) -> tuple[str, float, float]:
        return (self.airfoil, self.reynolds, self.mach)

    def to_polar(self) -> AirfoilPolar:
        return AirfoilPolar.from_xfoil_table(
            self.airfoil,
            self.reynolds,
            self.mach,
            np.asarray(self.table, dtype=float),
        )


@dataclass(frozen=True)
class XfoilPolarJob:
    """Serializable XFOIL worker input for one independent polar generation."""

    airfoil: str
    reynolds: float
    mach: float
    alpha_min_deg: float
    alpha_max_deg: float
    new_polar: bool
    timeout: int
    cache_path: Path

    @property
    def key(self) -> tuple[str, float, float]:
        return (self.airfoil, self.reynolds, self.mach)

    def result_from_table(self, table) -> XfoilPolarJobResult:
        return XfoilPolarJobResult(
            airfoil=self.airfoil,
            reynolds=self.reynolds,
            mach=self.mach,
            table=np.asarray(table, dtype=float).tolist(),
        )


def _get_mpi_pool_executor():
    """Return mpi4py's executor class, or the import/runtime error."""
    try:
        from mpi4py.futures import MPIPoolExecutor
    except (ImportError, OSError, RuntimeError) as err:
        return None, err
    return MPIPoolExecutor, None


def _configure_xfoil_output_for_path(xfoil: Xfoil, output_path: Path) -> None:
    """Point one XFOIL process at its own output file."""
    output_path = Path(output_path)
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


def _run_xfoil_polar_job(job: XfoilPolarJob) -> XfoilPolarJobResult:
    """Generate or read one polar table. Safe to execute in an MPI worker."""
    xfoil = Xfoil(new_polar=job.new_polar, timeout=job.timeout)
    _configure_xfoil_output_for_path(xfoil, job.cache_path)

    if not job.new_polar and job.cache_path.exists():
        return job.result_from_table(xfoil.read_polar())

    if not xfoil.simulate(
        job.airfoil,
        job.mach,
        job.reynolds,
        alpha_min_deg=job.alpha_min_deg,
        alpha_max_deg=job.alpha_max_deg,
    ):
        raise RuntimeError(xfoil.error_message)

    return job.result_from_table(xfoil.read_polar())


@dataclass
class LinearPolarProvider:
    """Deterministic analytic polar for tests and early design studies."""

    lift_slope_per_rad: float = 2.0 * np.pi
    zero_lift_alpha_deg: float = 0.0
    cd0: float = 0.01
    induced_drag_factor: float = 0.01
    cm0: float = 0.0
    cl_max: float | None = 1.4

    def prepare_conditions(
        self,
        conditions: Iterable[tuple[str, float, float]],
    ) -> None:
        """Analytic coefficients need no cache warmup."""
        return None

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
    the nonlinear BEMT solve. Generated files live in a provider-owned temporary
    directory by default, while in-memory cache entries are reused for repeated
    lookup during the current solve.
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
        parallel_workers: int = 8,
        parallel_backend: ParallelBackend = "auto",
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
        if parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1.")
        if parallel_backend not in ("auto", "mpi", "serial"):
            raise ValueError("parallel_backend must be 'auto', 'mpi', or 'serial'.")
        self.parallel_workers = int(parallel_workers)
        self.parallel_backend = parallel_backend
        self._temporary_cache = None
        if cache_directory is None:
            temp_root = get_repo_root() / "data" / "XFOIL6.99" / "tmp"
            temp_root.mkdir(parents=True, exist_ok=True)
            self._temporary_cache = _ShortTemporaryDirectory(temp_root)
            self.cache_directory = Path(self._temporary_cache.name)
        else:
            self.cache_directory = Path(cache_directory)
        self._cache: dict[tuple[str, float, float], AirfoilPolar] = {}

    def cleanup(self) -> None:
        """Remove the provider-owned temporary polar cache."""
        if self._temporary_cache is not None:
            self._temporary_cache.cleanup()
            self._temporary_cache = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def get_coefficients(
        self,
        airfoil: str,
        alpha_deg: float,
        reynolds: float,
        mach: float,
    ) -> AirfoilCoefficients:
        key = self._condition_key(airfoil, reynolds, mach)
        if key not in self._cache:
            self.prepare_conditions([(airfoil, reynolds, mach)])
        return self._cache[key].coefficients_at(alpha_deg)

    def prepare_conditions(
        self,
        conditions: Iterable[tuple[str, float, float]],
    ) -> None:
        """
        Generate all missing condition bins before element iterations need them.

        Each XFOIL run is independent once airfoil/Re/Mach and alpha sweep are
        known, so missing jobs can be distributed through mpi4py when available.
        """
        jobs: list[XfoilPolarJob] = []
        queued_keys = set()

        for airfoil, reynolds, mach in conditions:
            key = self._condition_key(airfoil, reynolds, mach)
            if key in self._cache or key in queued_keys:
                continue
            airfoil_key, reynolds_key, mach_key = key
            if not self.new_polar:
                cache_path = self._cache_file_path(airfoil_key, reynolds_key, mach_key)
                if cache_path.exists():
                    self._cache[key] = self._read_cached_polar(
                        airfoil_key,
                        reynolds_key,
                        mach_key,
                    )
                    continue
            jobs.append(self._create_job(airfoil_key, reynolds_key, mach_key))
            queued_keys.add(key)

        if not jobs:
            return

        for result in self._run_jobs(jobs):
            self._cache[result.key] = result.to_polar()

    def _condition_key(
        self,
        airfoil: str,
        reynolds: float,
        mach: float,
    ) -> tuple[str, float, float]:
        normalized = normalize_airfoil_name(airfoil)
        reynolds_key = self._round_to_positive_bin(
            max(reynolds, 1000.0),
            self.reynolds_bin,
        )
        mach_key = self._round_to_bin(max(mach, 0.0), self.mach_bin)
        return (normalized, reynolds_key, mach_key)

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
        return _run_xfoil_polar_job(
            self._create_job(airfoil, reynolds, mach)
        ).to_polar()

    def _read_cached_polar(self, airfoil: str, reynolds: float, mach: float) -> AirfoilPolar:
        xfoil = Xfoil(new_polar=False, timeout=self.timeout)
        _configure_xfoil_output_for_path(
            xfoil,
            self._cache_file_path(airfoil, reynolds, mach),
        )
        return AirfoilPolar.from_xfoil_table(airfoil, reynolds, mach, xfoil.read_polar())

    def _create_job(self, airfoil: str, reynolds: float, mach: float) -> XfoilPolarJob:
        return XfoilPolarJob(
            airfoil=airfoil,
            reynolds=reynolds,
            mach=mach,
            alpha_min_deg=self.alpha_min_deg,
            alpha_max_deg=self.alpha_max_deg,
            new_polar=self.new_polar,
            timeout=self.timeout,
            cache_path=self._cache_file_path(airfoil, reynolds, mach),
        )

    def _run_jobs(self, jobs: list[XfoilPolarJob]) -> Iterable[XfoilPolarJobResult]:
        if (
            len(jobs) > 1
            and self.parallel_workers > 1
            and self.parallel_backend != "serial"
        ):
            executor_class, error = _get_mpi_pool_executor()
            if executor_class is not None:
                with executor_class(max_workers=self.parallel_workers) as executor:
                    yield from executor.map(_run_xfoil_polar_job, jobs)
                return
            if self.parallel_backend == "mpi":
                raise RuntimeError(
                    "MPI XFOIL polar generation was requested, but mpi4py could "
                    "not load an MPI runtime. Install MS-MPI or Intel MPI, or use "
                    "parallel_backend='auto'/'serial'."
                ) from error

        for job in jobs:
            yield _run_xfoil_polar_job(job)

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
        _configure_xfoil_output_for_path(xfoil, output_path)
