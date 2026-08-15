"""Session-scoped run records for the Web UI run rail.

A hover calculation used to be a single mutable output: the GUI held the last
result and every new calculation overwrote it. The run rail treats a result as
a named, comparable record instead, so a design change can be compared against
the run it came from. Records are append-only within a browser session, hold
enough input state to re-solve, and are serializable so a session can be
exported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Iterable, Literal

from pycopter import CoaxialHoverResult, HoverResult


RunStatus = Literal["queued", "running", "done", "failed", "cancelled"]

# A session keeps this many records before the oldest non-baseline run is
# dropped, so a long parameter study cannot grow the rail without bound.
DEFAULT_MAX_RUNS = 200


@dataclass(frozen=True)
class RunRecord:
    """One persisted calculation in the session."""

    run_id: str
    label: str
    created_at: datetime
    # Full GUI.md key snapshot plus the blade station rows, which together are
    # enough to re-solve the run and to diff two runs field by field.
    inputs: dict[str, Any]
    result: HoverResult | CoaxialHoverResult | None = None
    status: RunStatus = "queued"
    error: str | None = None
    warnings: tuple[str, ...] = ()
    # Sweep points point at the sweep run that produced them.
    parent_id: str | None = None
    # (completed, total) while a multi-point run is in flight.
    progress: tuple[int, int] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    notes: tuple[str, ...] = ()

    @property
    def system_type(self) -> str:
        return str(self.inputs.get("rotor_system_type", "single"))

    @property
    def is_coaxial(self) -> bool:
        return isinstance(self.result, CoaxialHoverResult)

    @property
    def total_power_W(self) -> float | None:
        if self.result is None:
            return None
        if isinstance(self.result, CoaxialHoverResult):
            return self.result.total_power_W
        return self.result.power_W

    @property
    def total_thrust_N(self) -> float | None:
        if self.result is None:
            return None
        return self.result.total_thrust_N

    @property
    def figure_of_merit(self) -> float | None:
        if self.result is None:
            return None
        if isinstance(self.result, CoaxialHoverResult):
            # The pair has no single figure of merit; the upper rotor is the
            # reference rotor everywhere else in the UI, so report that.
            return self.result.upper.figure_of_merit
        return self.result.figure_of_merit

    @property
    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now()
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def eta_seconds(self) -> float | None:
        """Remaining time estimated from the points completed so far."""
        if self.progress is None or self.status != "running":
            return None
        done, total = self.progress
        elapsed = self.elapsed_seconds
        if not done or elapsed is None or total <= done:
            return None
        return elapsed / done * (total - done)

    def metrics(self) -> dict[str, float]:
        """Scalar outputs of this run, for export and for delta calculations."""
        if self.result is None:
            return {}
        if isinstance(self.result, CoaxialHoverResult):
            return {
                "total_thrust_N": self.result.total_thrust_N,
                "total_power_W": self.result.total_power_W,
                "net_aircraft_yaw_torque_Nm": self.result.net_aircraft_yaw_torque_Nm,
                "interference_power_delta_W": self.result.interference_power_delta_W,
                "interference_loss_ratio": self.result.interference_loss_ratio,
                "wake_radius_m": self.result.wake_radius_m,
                "wake_velocity_m_s": self.result.wake_velocity_m_s,
                "upper_collective_pitch_deg": self.result.upper.collective_pitch_deg,
                "lower_collective_pitch_deg": self.result.lower.collective_pitch_deg,
                "upper_thrust_N": self.result.upper.total_thrust_N,
                "lower_thrust_N": self.result.lower.total_thrust_N,
                "upper_power_W": self.result.upper.power_W,
                "lower_power_W": self.result.lower.power_W,
                "upper_figure_of_merit": self.result.upper.figure_of_merit,
                "lower_figure_of_merit": self.result.lower.figure_of_merit,
                "clamped_element_count": float(self.result.clamped_element_count),
            }
        return {
            "total_thrust_N": self.result.total_thrust_N,
            "power_W": self.result.power_W,
            "induced_power_W": self.result.induced_power_W,
            "profile_power_W": self.result.profile_power_W,
            "collective_pitch_deg": self.result.collective_pitch_deg,
            "figure_of_merit": self.result.figure_of_merit,
            "ct": self.result.ct,
            "cp": self.result.cp,
            "solidity": self.result.solidity,
            "mean_loss_factor": self.result.mean_loss_factor,
            "aircraft_yaw_torque_Nm": self.result.aircraft_yaw_torque_Nm,
            "clamped_element_count": float(self.result.clamped_element_count),
        }

    def to_payload(self) -> dict[str, Any]:
        """JSON-serializable form used by ``Export session runs``.

        Element load tables are deliberately excluded: they are exported
        separately as CSV, and a session of 60-element runs would otherwise
        produce a file dominated by them.
        """
        return {
            "run_id": self.run_id,
            "label": self.label,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "status": self.status,
            "error": self.error,
            "warnings": list(self.warnings),
            "parent_id": self.parent_id,
            "inputs": self.inputs,
            "metrics": self.metrics(),
        }


@dataclass
class RunStore:
    """Append-only, session-scoped collection of :class:`RunRecord`."""

    max_runs: int = DEFAULT_MAX_RUNS
    _records: list[RunRecord] = field(default_factory=list)
    _baseline_id: str | None = None
    _counter: int = 0

    # ---- reading -------------------------------------------------------

    @property
    def records(self) -> tuple[RunRecord, ...]:
        return tuple(self._records)

    @property
    def baseline_id(self) -> str | None:
        return self._baseline_id

    @property
    def baseline(self) -> RunRecord | None:
        return self.get(self._baseline_id) if self._baseline_id else None

    def get(self, run_id: str | None) -> RunRecord | None:
        if not run_id:
            return None
        for record in self._records:
            if record.run_id == run_id:
                return record
        return None

    def children_of(self, run_id: str) -> tuple[RunRecord, ...]:
        return tuple(record for record in self._records if record.parent_id == run_id)

    def newest_first(self) -> tuple[RunRecord, ...]:
        return tuple(reversed(self._records))

    def __len__(self) -> int:
        return len(self._records)

    # ---- writing -------------------------------------------------------

    def create(
        self,
        label: str,
        inputs: dict[str, Any],
        *,
        status: RunStatus = "queued",
        parent_id: str | None = None,
        progress: tuple[int, int] | None = None,
        notes: Iterable[str] = (),
    ) -> RunRecord:
        self._counter += 1
        record = RunRecord(
            run_id=f"R-{self._counter:02d}",
            label=label,
            created_at=datetime.now(),
            # Copy so later edits in the inspector cannot rewrite history.
            inputs=_deep_copy_inputs(inputs),
            status=status,
            parent_id=parent_id,
            progress=progress,
            started_at=datetime.now() if status == "running" else None,
            notes=tuple(notes),
        )
        self._records.append(record)
        if self._baseline_id is None and status in ("done", "running", "queued"):
            self._baseline_id = record.run_id
        self._trim()
        return record

    def update(self, run_id: str, **changes: Any) -> RunRecord | None:
        for index, record in enumerate(self._records):
            if record.run_id != run_id:
                continue
            if changes.get("status") == "running" and record.started_at is None:
                changes.setdefault("started_at", datetime.now())
            if changes.get("status") in ("done", "failed", "cancelled"):
                changes.setdefault("finished_at", datetime.now())
            if "warnings" in changes:
                changes["warnings"] = tuple(changes["warnings"])
            if "notes" in changes:
                changes["notes"] = tuple(changes["notes"])
            updated = replace(record, **changes)
            self._records[index] = updated
            return updated
        return None

    def set_baseline(self, run_id: str | None) -> None:
        if run_id is None or self.get(run_id) is not None:
            self._baseline_id = run_id

    def delete(self, run_id: str) -> int:
        """Delete a run and any sweep points that belong to it."""
        doomed = {run_id} | {child.run_id for child in self.children_of(run_id)}
        before = len(self._records)
        self._records = [record for record in self._records if record.run_id not in doomed]
        if self._baseline_id in doomed:
            self._baseline_id = self._records[-1].run_id if self._records else None
        return before - len(self._records)

    def clear(self) -> None:
        self._records = []
        self._baseline_id = None
        self._counter = 0

    def _trim(self) -> None:
        if len(self._records) <= self.max_runs:
            return
        keep_from = len(self._records) - self.max_runs
        dropped = self._records[:keep_from]
        self._records = self._records[keep_from:]
        if any(record.run_id == self._baseline_id for record in dropped):
            self._baseline_id = self._records[0].run_id if self._records else None

    # ---- export --------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self._baseline_id,
            "runs": [record.to_payload() for record in self._records],
        }

    def csv_rows(self) -> list[dict[str, Any]]:
        """One flat row per run, with every metric any run reported."""
        metric_names: list[str] = []
        for record in self._records:
            for name in record.metrics():
                if name not in metric_names:
                    metric_names.append(name)

        rows: list[dict[str, Any]] = []
        for record in self._records:
            metrics = record.metrics()
            row: dict[str, Any] = {
                "run_id": record.run_id,
                "label": record.label,
                "created_at": record.created_at.isoformat(timespec="seconds"),
                "status": record.status,
                "parent_id": record.parent_id or "",
                "baseline": "yes" if record.run_id == self._baseline_id else "",
                "warnings": " | ".join(record.warnings),
                "error": record.error or "",
            }
            for name in metric_names:
                row[name] = metrics.get(name, "")
            rows.append(row)
        return rows


def _deep_copy_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Snapshot a config dict, including its nested station rows."""
    copied = dict(inputs)
    rows = copied.get("station_rows")
    if isinstance(rows, list):
        copied["station_rows"] = [dict(row) for row in rows]
    return copied


def diff_inputs(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Field-by-field difference between two run input snapshots."""
    keys = sorted(set(left) | set(right))
    changes: list[dict[str, Any]] = []
    for key in keys:
        before = left.get(key)
        after = right.get(key)
        if before == after:
            continue
        changes.append({"key": key, "from": before, "to": after})
    return changes


