"""Airfoil polar providers used by the BEMT solver."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Protocol

import numpy as np

from .xfoil import (
    DEFAULT_ALPHA_STEP_DEG,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_N_CRIT,
    MAX_XFOIL_ALPHA_DEG,
    Xfoil,
    get_repo_root,
    normalize_airfoil_name,
)

ParallelBackend = Literal["mpi", "serial"]
PolarBinSource = Literal["cache", "generated"]


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
class PolarBinRecord:
    """Provenance of one airfoil/Re/Mach polar bin used by a calculation."""

    airfoil: str
    reynolds: float
    mach: float
    alpha_min_deg: float
    alpha_max_deg: float
    source: PolarBinSource
    # True when the requested Reynolds or Mach fell outside the provider's
    # supported range and a neighbouring flow condition was used instead.
    # Ordinary Re/Mach binning is not a substitution.
    substituted: bool
    lookups: int
    cache_file: str

    @property
    def label(self) -> str:
        return f"{self.airfoil} Re={self.reynolds:.4g} M={self.mach:.3g}"

    @property
    def status(self) -> str:
        return "substituted" if self.substituted else self.source

    def as_dict(self) -> dict[str, object]:
        return {
            "airfoil": self.airfoil,
            "reynolds": self.reynolds,
            "mach": self.mach,
            "alpha_min_deg": self.alpha_min_deg,
            "alpha_max_deg": self.alpha_max_deg,
            "source": self.source,
            "substituted": self.substituted,
            "status": self.status,
            "lookups": self.lookups,
            "cache_file": self.cache_file,
            "label": self.label,
        }


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
    alpha_step_deg: float = DEFAULT_ALPHA_STEP_DEG
    n_crit: float = DEFAULT_N_CRIT
    max_iterations: int = DEFAULT_MAX_ITERATIONS

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


def _job_to_payload(job: XfoilPolarJob) -> dict[str, object]:
    return {
        "airfoil": job.airfoil,
        "reynolds": job.reynolds,
        "mach": job.mach,
        "alpha_min_deg": job.alpha_min_deg,
        "alpha_max_deg": job.alpha_max_deg,
        "new_polar": job.new_polar,
        "timeout": job.timeout,
        "cache_path": str(job.cache_path),
        "alpha_step_deg": job.alpha_step_deg,
        "n_crit": job.n_crit,
        "max_iterations": job.max_iterations,
    }


def _job_from_payload(payload: dict[str, object]) -> XfoilPolarJob:
    return XfoilPolarJob(
        airfoil=str(payload["airfoil"]),
        reynolds=float(payload["reynolds"]),
        mach=float(payload["mach"]),
        alpha_min_deg=float(payload["alpha_min_deg"]),
        alpha_max_deg=float(payload["alpha_max_deg"]),
        new_polar=bool(payload["new_polar"]),
        timeout=int(payload["timeout"]),
        cache_path=Path(str(payload["cache_path"])),
        alpha_step_deg=float(payload.get("alpha_step_deg", DEFAULT_ALPHA_STEP_DEG)),
        n_crit=float(payload.get("n_crit", DEFAULT_N_CRIT)),
        max_iterations=int(payload.get("max_iterations", DEFAULT_MAX_ITERATIONS)),
    )


def _result_to_payload(result: XfoilPolarJobResult) -> dict[str, object]:
    return {
        "airfoil": result.airfoil,
        "reynolds": result.reynolds,
        "mach": result.mach,
        "table": result.table,
    }


def _result_from_payload(payload: dict[str, object]) -> XfoilPolarJobResult:
    return XfoilPolarJobResult(
        airfoil=str(payload["airfoil"]),
        reynolds=float(payload["reynolds"]),
        mach=float(payload["mach"]),
        table=payload["table"],
    )


def _get_mpi_pool_executor():
    """Return mpi4py's executor class, or the import/runtime error."""
    _ensure_mpi_runtime_path()
    try:
        from mpi4py.futures import MPIPoolExecutor
    except (ImportError, OSError, RuntimeError) as err:
        return None, err
    return MPIPoolExecutor, None


def _ensure_mpi_runtime_path() -> None:
    """Expose MS-MPI executables in processes started before PATH was refreshed."""
    msmpi_bin = Path("C:/Program Files/Microsoft MPI/Bin")
    if not msmpi_bin.exists():
        return
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(msmpi_bin) not in path_parts:
        os.environ["PATH"] = str(msmpi_bin) + os.pathsep + os.environ.get("PATH", "")


def _is_real_mpi4py_executor(executor_class) -> bool:
    return getattr(executor_class, "__module__", "").startswith("mpi4py.")


def _can_use_dynamic_mpi_spawn() -> bool:
    # On Windows/MS-MPI, COMM_SELF.Spawn fails outside an MPI process manager.
    # MPIPoolExecutor uses that spawn path, so use mpiexec workers instead.
    return os.name != "nt"


def _mpi_worker_environment() -> dict[str, str]:
    _ensure_mpi_runtime_path()
    env = os.environ.copy()
    repo_root = get_repo_root()
    path_entries = [str(repo_root / "src"), str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        path_entries.extend(existing_pythonpath.split(os.pathsep))

    seen = set()
    deduped = []
    for entry in path_entries:
        if entry and entry not in seen:
            deduped.append(entry)
            seen.add(entry)
    env["PYTHONPATH"] = os.pathsep.join(deduped)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _tail_process_text(text: str, max_chars: int = 4000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _mpi_batch_timeout_seconds(jobs: list[XfoilPolarJob], worker_count: int) -> int:
    job_timeout = max((job.timeout for job in jobs), default=60)
    waves = max(1, (len(jobs) + worker_count - 1) // worker_count)
    return int(max(60, job_timeout * waves + 60))


def _run_xfoil_jobs_with_mpiexec(
    jobs: list[XfoilPolarJob],
    max_workers: int,
) -> list[XfoilPolarJobResult]:
    _ensure_mpi_runtime_path()
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        raise RuntimeError(
            "MPI XFOIL polar generation was requested, but mpiexec was not found. "
            "Install MS-MPI or Intel MPI, or set parallel_backend='serial' explicitly "
            "for a non-parallel debug run."
        )

    worker_count = max(1, min(int(max_workers), len(jobs)))
    with tempfile.TemporaryDirectory(prefix="pycopter-mpi-") as tempdir:
        input_path = Path(tempdir) / "jobs.json"
        output_path = Path(tempdir) / "results.json"
        input_path.write_text(
            json.dumps([_job_to_payload(job) for job in jobs]),
            encoding="utf-8",
        )
        command = [
            mpiexec,
            "-n",
            str(worker_count),
            sys.executable,
            "-m",
            "pycopter.polars",
            "--xfoil-mpi-worker",
            str(input_path),
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=get_repo_root(),
                env=_mpi_worker_environment(),
                capture_output=True,
                text=True,
                timeout=_mpi_batch_timeout_seconds(jobs, worker_count),
            )
        except subprocess.TimeoutExpired as err:
            raise RuntimeError(
                f"MPI XFOIL polar generation timed out after {err.timeout} seconds."
            ) from err

        if completed.returncode != 0:
            details = "\n".join(
                part
                for part in (
                    _tail_process_text(completed.stdout),
                    _tail_process_text(completed.stderr),
                )
                if part
            )
            raise RuntimeError(
                "MPI XFOIL polar generation failed under mpiexec."
                + (f"\n{details}" if details else "")
            )
        if not output_path.exists():
            raise RuntimeError("MPI XFOIL polar generation did not produce results.")

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        errors = payload.get("errors", [])
        if errors:
            first = errors[0]
            raise RuntimeError(
                "MPI XFOIL worker failed while generating a polar: "
                f"{first.get('error', 'unknown error')}\n"
                f"{first.get('traceback', '')}"
            )
        return [_result_from_payload(result) for result in payload["results"]]


def _run_xfoil_mpi_worker_file(input_path: str, output_path: str) -> int:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        jobs_payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    else:
        jobs_payload = None
    jobs_payload = comm.bcast(jobs_payload, root=0)

    local_results = []
    local_errors = []
    for index, job_payload in enumerate(jobs_payload):
        if index % size != rank:
            continue
        try:
            result = _run_xfoil_polar_job(_job_from_payload(job_payload))
            local_results.append((index, _result_to_payload(result)))
        except Exception as err:  # pragma: no cover - reported to parent process.
            local_errors.append(
                {
                    "index": index,
                    "rank": rank,
                    "error": f"{type(err).__name__}: {err}",
                    "traceback": traceback.format_exc(),
                }
            )

    gathered = comm.gather(
        {"results": local_results, "errors": local_errors},
        root=0,
    )
    if rank == 0:
        combined_results = []
        combined_errors = []
        for worker_payload in gathered:
            combined_results.extend(worker_payload["results"])
            combined_errors.extend(worker_payload["errors"])
        ordered_results = [
            result for _, result in sorted(combined_results, key=lambda item: item[0])
        ]
        Path(output_path).write_text(
            json.dumps({"results": ordered_results, "errors": combined_errors}),
            encoding="utf-8",
        )
    return 0


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
    xfoil = Xfoil(
        new_polar=job.new_polar,
        timeout=job.timeout,
        alpha_step_deg=job.alpha_step_deg,
        n_crit=job.n_crit,
        max_iterations=job.max_iterations,
    )
    _configure_xfoil_output_for_path(xfoil, job.cache_path)

    if job.cache_path.exists():
        return job.result_from_table(xfoil.read_polar())
    if not job.new_polar:
        raise RuntimeError(_missing_cached_polar_message(job))

    if not xfoil.simulate(
        job.airfoil,
        job.mach,
        job.reynolds,
        alpha_min_deg=job.alpha_min_deg,
        alpha_max_deg=job.alpha_max_deg,
    ):
        raise RuntimeError(xfoil.error_message)

    return job.result_from_table(xfoil.read_polar())


def _missing_cached_polar_message(job: XfoilPolarJob) -> str:
    return (
        "Cached XFOIL polar is missing and XFOIL launch is disabled. "
        "Enable 'Run XFOIL For Missing Polars' or choose a cache directory "
        f"containing this bin: airfoil={job.airfoil}, Re={job.reynolds:.0f}, "
        f"Mach={job.mach:.3f}, alpha={job.alpha_min_deg:g}..{job.alpha_max_deg:g} deg."
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

    def prepare_conditions(
        self,
        conditions: Iterable[tuple[str, float, float]],
    ) -> None:
        """Analytic coefficients need no cache warmup."""
        return None

    def polar_bin_report(
        self,
        conditions: Iterable[tuple[str, float, float]] | None = None,
    ) -> list[dict[str, object]]:
        """Analytic coefficients are not binned, so there is nothing to report."""
        return []

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
        alpha_min_deg: float = -3.0,
        alpha_max_deg: float = MAX_XFOIL_ALPHA_DEG,
        reynolds_bin: float = 100000.0,
        mach_bin: float = 0.1,
        max_reynolds: float = 1_000_000.0,
        max_mach: float = 0.6,
        timeout: int = 60,
        cache_directory: str | Path | None = None,
        parallel_workers: int = 8,
        parallel_backend: ParallelBackend = "mpi",
        alpha_step_deg: float = DEFAULT_ALPHA_STEP_DEG,
        n_crit: float = DEFAULT_N_CRIT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        progress_callback=None,
    ):
        self.new_polar = new_polar
        self.alpha_min_deg = alpha_min_deg
        self.alpha_max_deg = min(alpha_max_deg, MAX_XFOIL_ALPHA_DEG)
        if self.alpha_min_deg > self.alpha_max_deg:
            raise ValueError(
                f"alpha_min_deg must be <= {MAX_XFOIL_ALPHA_DEG} deg for XFOIL runs."
            )
        if reynolds_bin <= 0:
            raise ValueError("reynolds_bin must be positive.")
        if mach_bin <= 0:
            raise ValueError("mach_bin must be positive.")
        if max_reynolds <= 0:
            raise ValueError("max_reynolds must be positive.")
        if max_mach <= 0:
            raise ValueError("max_mach must be positive.")
        self.reynolds_bin = reynolds_bin
        self.mach_bin = mach_bin
        self.max_reynolds = max_reynolds
        self.max_mach = max_mach
        self.timeout = timeout
        if parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1.")
        if parallel_backend not in ("mpi", "serial"):
            raise ValueError("parallel_backend must be 'mpi' or 'serial'.")
        if alpha_step_deg <= 0:
            raise ValueError("alpha_step_deg must be positive.")
        if n_crit <= 0:
            raise ValueError("n_crit must be positive.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self.parallel_workers = int(parallel_workers)
        self.parallel_backend = parallel_backend
        self.alpha_step_deg = float(alpha_step_deg)
        self.n_crit = float(n_crit)
        self.max_iterations = int(max_iterations)
        # Optional callable(event_name, payload) used by the Web UI to show
        # XFOIL batch state while a calculation runs off the event loop.
        self.progress_callback = progress_callback
        self._bin_records: dict[tuple[str, float, float], PolarBinRecord] = {}
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
        key, substituted = self._condition_key_with_flags(airfoil, reynolds, mach)
        if key not in self._cache:
            self.prepare_conditions([(airfoil, reynolds, mach)])
        self._touch_bin_record(key, substituted=substituted)
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
            key, substituted = self._condition_key_with_flags(airfoil, reynolds, mach)
            if key in self._cache or key in queued_keys:
                if substituted:
                    self._touch_bin_record(key, substituted=True, count=0)
                continue
            airfoil_key, reynolds_key, mach_key = key
            cache_path = self._cache_file_path(airfoil_key, reynolds_key, mach_key)
            if cache_path.exists():
                self._cache[key] = self._read_cached_polar(
                    airfoil_key,
                    reynolds_key,
                    mach_key,
                )
                self._record_bin(key, source="cache", substituted=substituted)
                continue
            if not self.new_polar:
                raise RuntimeError(
                    _missing_cached_polar_message(
                        self._create_job(airfoil_key, reynolds_key, mach_key)
                    )
                )
            jobs.append(self._create_job(airfoil_key, reynolds_key, mach_key))
            queued_keys.add(key)
            self._record_bin(key, source="generated", substituted=substituted)

        if not jobs:
            return

        worker_count = self._effective_worker_count(len(jobs))
        self._emit("batch_start", {"jobs": len(jobs), "workers": worker_count})
        try:
            for result in self._run_jobs(jobs):
                self._cache[result.key] = result.to_polar()
        except Exception as err:
            self._emit("batch_failed", {"jobs": len(jobs), "error": str(err)})
            raise
        self._emit("batch_done", {"jobs": len(jobs), "workers": worker_count})

    def polar_bin_report(
        self,
        conditions: Iterable[tuple[str, float, float]] | None = None,
    ) -> list[dict[str, object]]:
        """Return the provenance of every polar bin this provider has served.

        Passing ``conditions`` narrows the report to the bins those flow
        conditions resolve to, which is how one calculation reports only its
        own bins from a session-wide provider.
        """
        if conditions is None:
            records = list(self._bin_records.values())
        else:
            wanted = {
                self._condition_key(airfoil, reynolds, mach)
                for airfoil, reynolds, mach in conditions
            }
            records = [
                record
                for key, record in self._bin_records.items()
                if key in wanted
            ]
        records.sort(key=lambda record: (record.airfoil, record.reynolds, record.mach))
        return [record.as_dict() for record in records]

    def cache_bin_count(self) -> int:
        """Number of distinct polar bins loaded into this provider."""
        return len(self._bin_records)

    def cache_file_count(self) -> int:
        """Number of generated polar files sitting in the cache directory."""
        try:
            return sum(1 for path in self.cache_directory.glob("p*.txt") if path.is_file())
        except OSError:
            return 0

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event, payload)
        except Exception:
            # Progress reporting must never break a calculation.
            pass

    def _effective_worker_count(self, job_count: int) -> int:
        if job_count <= 1 or self.parallel_backend == "serial":
            return 1
        return max(1, min(self.parallel_workers, job_count))

    def _record_bin(
        self,
        key: tuple[str, float, float],
        *,
        source: PolarBinSource,
        substituted: bool,
    ) -> None:
        airfoil, reynolds, mach = key
        existing = self._bin_records.get(key)
        self._bin_records[key] = PolarBinRecord(
            airfoil=airfoil,
            reynolds=reynolds,
            mach=mach,
            alpha_min_deg=self.alpha_min_deg,
            alpha_max_deg=self.alpha_max_deg,
            source=source,
            substituted=substituted or bool(existing and existing.substituted),
            lookups=existing.lookups if existing else 0,
            cache_file=self._cache_file_path(airfoil, reynolds, mach).name,
        )

    def _touch_bin_record(
        self,
        key: tuple[str, float, float],
        *,
        substituted: bool,
        count: int = 1,
    ) -> None:
        existing = self._bin_records.get(key)
        if existing is None:
            self._record_bin(key, source="cache", substituted=substituted)
            existing = self._bin_records[key]
        self._bin_records[key] = replace(
            existing,
            lookups=existing.lookups + count,
            substituted=existing.substituted or substituted,
        )

    def _condition_key(
        self,
        airfoil: str,
        reynolds: float,
        mach: float,
    ) -> tuple[str, float, float]:
        return self._condition_key_with_flags(airfoil, reynolds, mach)[0]

    def _condition_key_with_flags(
        self,
        airfoil: str,
        reynolds: float,
        mach: float,
    ) -> tuple[tuple[str, float, float], bool]:
        """Return the bin key plus whether the request was outside provider range.

        Rounding a request into a Re/Mach bin is the documented binning scheme,
        not a substitution. Clamping a request that sits outside the provider's
        supported range is, because the returned polar then describes a
        different flow condition than the blade element asked for.
        """
        normalized = normalize_airfoil_name(airfoil)
        clamped_reynolds = min(max(reynolds, 1000.0), self.max_reynolds)
        clamped_mach = min(max(mach, 0.0), self.max_mach)
        substituted = (
            clamped_reynolds != float(reynolds) or clamped_mach != float(mach)
        )
        reynolds_key = self._round_to_positive_bin(clamped_reynolds, self.reynolds_bin)
        mach_key = self._round_to_bin(clamped_mach, self.mach_bin)
        return (normalized, reynolds_key, mach_key), substituted

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
        xfoil = Xfoil(
            new_polar=False,
            timeout=self.timeout,
            alpha_step_deg=self.alpha_step_deg,
            n_crit=self.n_crit,
            max_iterations=self.max_iterations,
        )
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
            alpha_step_deg=self.alpha_step_deg,
            n_crit=self.n_crit,
            max_iterations=self.max_iterations,
        )

    def _run_jobs(self, jobs: list[XfoilPolarJob]) -> Iterable[XfoilPolarJobResult]:
        if (
            len(jobs) > 1
            and self.parallel_workers > 1
            and self.parallel_backend != "serial"
        ):
            executor_class, error = _get_mpi_pool_executor()
            if executor_class is not None and (
                not _is_real_mpi4py_executor(executor_class)
                or _can_use_dynamic_mpi_spawn()
            ):
                executor_kwargs = {}
                if _is_real_mpi4py_executor(executor_class):
                    repo_root = get_repo_root()
                    executor_kwargs = {
                        "main": False,
                        "path": [str(repo_root / "src"), str(repo_root)],
                        "wdir": str(repo_root),
                        "env": _mpi_worker_environment(),
                    }
                with executor_class(
                    max_workers=self.parallel_workers,
                    **executor_kwargs,
                ) as executor:
                    yield from executor.map(_run_xfoil_polar_job, jobs)
                return
            if executor_class is not None:
                try:
                    yield from _run_xfoil_jobs_with_mpiexec(jobs, self.parallel_workers)
                    return
                except RuntimeError as mpiexec_error:
                    if self.parallel_backend == "mpi":
                        raise mpiexec_error from error

            if self.parallel_backend == "mpi":
                raise RuntimeError(
                    "MPI XFOIL polar generation was requested, but mpi4py could "
                    "not load an MPI runtime. Install MS-MPI or Intel MPI, or set "
                    "parallel_backend='serial' explicitly for a non-parallel debug run."
                ) from error

        for job in jobs:
            yield _run_xfoil_polar_job(job)

    def _cache_file_path(self, airfoil: str, reynolds: float, mach: float) -> Path:
        key = "|".join(
            [
                normalize_airfoil_name(airfoil),
                str(int(round(reynolds))),
                self._safe_float_token(mach),
                self._safe_float_token(self.alpha_min_deg),
                self._safe_float_token(self.alpha_max_deg),
                # Alpha resolution and transition model change the table itself,
                # so they must not share a cache entry with a coarser run.
                self._safe_float_token(self.alpha_step_deg),
                self._safe_float_token(self.n_crit),
            ]
        )
        # XFOIL silently truncates long filename components on Windows.
        digest = hashlib.sha1(key.encode("ascii")).hexdigest()[:16]
        return self.cache_directory / f"p{digest}.txt"

    def _safe_float_token(self, value: float) -> str:
        text = f"{value:.4g}".replace("-", "neg").replace(".", "p")
        return text.replace("+", "")

    def _configure_xfoil_output(self, xfoil: Xfoil, output_path: Path) -> None:
        _configure_xfoil_output_for_path(xfoil, output_path)


def _main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "--xfoil-mpi-worker":
        return _run_xfoil_mpi_worker_file(argv[1], argv[2])
    raise SystemExit("Usage: python -m pycopter.polars --xfoil-mpi-worker JOBS RESULTS")


if __name__ == "__main__":
    raise SystemExit(_main())
