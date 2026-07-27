"""
Experiment tracking abstraction for the deployment training scripts.

PRD reference (§8.1 Training & Experiment Tracking):
    "Toggleable via a single environment variable... When enabled, log
    params/metrics/artifacts for each fold and the final held-out evaluation.
    If MLflow is not configured, fall back to writing the same metrics to a
    local JSON/JSONL file so nothing is silently lost."

Design notes (SOLID):
    - ExperimentTracker is the abstraction. train_combo() (combo_runner.py)
      depends only on this interface — it never imports mlflow and never
      branches on "is MLflow enabled" (Dependency Inversion). That branch
      happens exactly once, in build_tracker(), which is the only place
      that needs to change if a third backend is ever added (Open/Closed).
    - MLflowTracker and JsonFallbackTracker are interchangeable
      implementations (Liskov Substitution) — combo_runner.py calls the same
      four methods on either one and does not need to know which it has.
    - Each tracker owns exactly one concern: shipping metrics/params/artifacts
      to one destination (Single Responsibility).

Toggle: set the MLFLOW_TRACKING_URI environment variable to enable MLflow.
Leave it unset (or empty) to use the local JSON/JSONL fallback — nothing
else needs to change in the calling training script either way.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class ExperimentTracker(ABC):
    """Minimal experiment-tracking contract every backend must satisfy."""

    @abstractmethod
    def start_run(self, run_name: str, params: dict[str, Any]) -> None:
        """Open a new run and record its hyperparameters/config."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Record a batch of scalar metrics, optionally tagged with a step (e.g. fold or epoch)."""

    @abstractmethod
    def log_artifact(self, path: str) -> None:
        """Record the path to a file artifact (checkpoint, confusion matrix image, etc.)."""

    @abstractmethod
    def end_run(self) -> None:
        """Close the current run. Must be safe to call even if start_run failed partway."""

    # Context-manager sugar so call sites can `with tracker.run(name, params): ...`
    def run(self, run_name: str, params: dict[str, Any]) -> "_TrackedRun":
        return _TrackedRun(self, run_name, params)


class _TrackedRun:
    """Context manager returned by ExperimentTracker.run(); ensures end_run() always fires."""

    def __init__(self, tracker: ExperimentTracker, run_name: str, params: dict[str, Any]):
        self._tracker = tracker
        self._run_name = run_name
        self._params = params

    def __enter__(self) -> ExperimentTracker:
        self._tracker.start_run(self._run_name, self._params)
        return self._tracker

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._tracker.end_run()
        return False  # never swallow exceptions from the training loop


class MLflowTracker(ExperimentTracker):
    """Logs to a real MLflow tracking server/local store, self-hosted per PRD §8.1/§8.6."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        import mlflow  # imported lazily so JsonFallbackTracker users never need mlflow installed

        # Recent MLflow versions refuse the plain local filesystem store
        # ("./mlruns") unless MLFLOW_ALLOW_FILE_STORE=true is set, which would
        # silently defeat the simplest possible self-hosted setup (PRD §8.6:
        # "self-host later if desired" implies exactly this local-directory
        # case). Only auto-opt-in when the URI is clearly a bare local path —
        # never override this for an actual server/database URI, since that
        # signal should be respected as-is.
        is_bare_local_path = not tracking_uri.startswith(
            ("http://", "https://", "databricks", "sqlite:", "postgresql:", "mysql:", "mssql:")
        )
        if is_bare_local_path:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        self._mlflow = mlflow
        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment_name)
        self._active = False

    def start_run(self, run_name: str, params: dict[str, Any]) -> None:
        self._mlflow.start_run(run_name=run_name)
        self._active = True
        if params:
            # MLflow rejects non-primitive values; stringify anything exotic defensively.
            safe_params = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                            for k, v in params.items()}
            self._mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: str) -> None:
        if Path(path).exists():
            self._mlflow.log_artifact(path)

    def end_run(self) -> None:
        if self._active:
            self._mlflow.end_run()
            self._active = False


class JsonFallbackTracker(ExperimentTracker):
    """
    Zero-infrastructure fallback: appends one JSON object per event to a
    JSONL file. Used automatically whenever MLFLOW_TRACKING_URI is unset,
    and also used as a safety net if MLflow itself fails to initialize
    (see build_tracker below) — training must never crash, or silently lose
    metrics, just because tracking infrastructure is unavailable.
    """

    def __init__(self, output_path: Path):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._run_name: Optional[str] = None

    def _write(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), "run_name": self._run_name, **record}
        with open(self._path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def start_run(self, run_name: str, params: dict[str, Any]) -> None:
        self._run_name = run_name
        self._write({"event": "start_run", "params": params})

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        self._write({"event": "metrics", "step": step, "metrics": metrics})

    def log_artifact(self, path: str) -> None:
        self._write({"event": "artifact", "path": str(path)})

    def end_run(self) -> None:
        self._write({"event": "end_run"})
        self._run_name = None


def build_tracker(experiment_name: str, output_dir: Path) -> ExperimentTracker:
    """
    Factory (the one place that decides which backend to use).

    MLFLOW_TRACKING_URI set and non-empty  -> MLflowTracker
    MLFLOW_TRACKING_URI unset/empty        -> JsonFallbackTracker
    MLflow configured but fails to import/connect -> falls back to JSON
    rather than crashing training over a tracking-infrastructure problem.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    fallback_path = Path(output_dir) / f"{experiment_name}_metrics.jsonl"

    if not tracking_uri:
        return JsonFallbackTracker(fallback_path)

    try:
        return MLflowTracker(tracking_uri, experiment_name)
    except Exception as exc:  # pragma: no cover - defensive path
        print(
            f"[tracking] MLFLOW_TRACKING_URI is set ('{tracking_uri}') but MLflow "
            f"could not be initialized ({exc!r}). Falling back to local JSONL logging "
            f"at {fallback_path} so metrics are not lost."
        )
        return JsonFallbackTracker(fallback_path)
