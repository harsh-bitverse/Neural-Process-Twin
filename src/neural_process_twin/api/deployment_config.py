"""Deployment configuration, resource limits, and execution concurrency controls for Milestone 11.

Provides environment-configurable limits to protect server resources, prevent OOM,
bound concurrent CPU-intensive numerical and neural solves, and track system metrics.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeploymentSettings:
    """Runtime operational and deployment settings loaded from environment variables."""

    # Resource limits
    max_request_body_bytes: int = int(
        os.getenv("NPT_MAX_REQUEST_BODY_BYTES", str(10 * 1024 * 1024))
    )  # 10 MB
    max_grid_nx: int = int(os.getenv("NPT_MAX_GRID_NX", "256"))
    max_grid_ny: int = int(os.getenv("NPT_MAX_GRID_NY", "256"))
    max_total_cells: int = int(os.getenv("NPT_MAX_TOTAL_CELLS", "65536"))  # 256 x 256

    # Concurrency controls
    max_concurrent_numerical_solves: int = int(
        os.getenv("NPT_MAX_CONCURRENT_NUMERICAL_SOLVES", "4")
    )
    max_concurrent_neural_inferences: int = int(
        os.getenv("NPT_MAX_CONCURRENT_NEURAL_INFERENCES", "8")
    )
    concurrency_queue_timeout_seconds: float = float(
        os.getenv("NPT_CONCURRENCY_QUEUE_TIMEOUT_SECONDS", "5.0")
    )
    request_execution_timeout_seconds: float = float(
        os.getenv("NPT_REQUEST_EXECUTION_TIMEOUT_SECONDS", "30.0")
    )

    # Security & CORS
    cors_allowed_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("NPT_CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()
        ]
    )
    enforce_sha256_verification: bool = (
        os.getenv("NPT_ENFORCE_SHA256_VERIFICATION", "true").lower() == "true"
    )

    # Authorized model checksums
    authorized_a3_sha256: str = os.getenv(
        "NPT_AUTHORIZED_A3_SHA256",
        "7b412133904819bcc654a06525c334923454bd26378312623ce5251d9b5f0f1d",
    )

    # Observability
    enable_metrics: bool = os.getenv("NPT_ENABLE_METRICS", "true").lower() == "true"


class ConcurrencyLimitExceededError(Exception):
    """Raised when incoming requests exceed bounded concurrency queues."""


class ExecutionTimeoutError(Exception):
    """Raised when an operation exceeds maximum allowable execution duration."""


class ConcurrencyGovernor:
    """Thread-safe resource governor enforcing concurrency limits on expensive scientific tasks."""

    def __init__(self, settings: DeploymentSettings) -> None:
        self.settings = settings
        self._numerical_sem = threading.Semaphore(settings.max_concurrent_numerical_solves)
        self._neural_sem = threading.Semaphore(settings.max_concurrent_neural_inferences)
        self._active_numerical = 0
        self._active_neural = 0
        self._lock = threading.Lock()

    @property
    def active_numerical_solves(self) -> int:
        with self._lock:
            return self._active_numerical

    @property
    def active_neural_inferences(self) -> int:
        with self._lock:
            return self._active_neural

    def acquire_numerical(self, timeout: float | None = None) -> None:
        to = timeout if timeout is not None else self.settings.concurrency_queue_timeout_seconds
        acquired = self._numerical_sem.acquire(timeout=to)
        if not acquired:
            raise ConcurrencyLimitExceededError(
                f"Numerical solver concurrency limit ({self.settings.max_concurrent_numerical_solves}) "
                f"exceeded. Server is under heavy computational load."
            )
        with self._lock:
            self._active_numerical += 1

    def release_numerical(self) -> None:
        with self._lock:
            self._active_numerical = max(0, self._active_numerical - 1)
        self._numerical_sem.release()

    def acquire_neural(self, timeout: float | None = None) -> None:
        to = timeout if timeout is not None else self.settings.concurrency_queue_timeout_seconds
        acquired = self._neural_sem.acquire(timeout=to)
        if not acquired:
            raise ConcurrencyLimitExceededError(
                f"Neural inference concurrency limit ({self.settings.max_concurrent_neural_inferences}) "
                f"exceeded. Server is under heavy computational load."
            )
        with self._lock:
            self._active_neural += 1

    def release_neural(self) -> None:
        with self._lock:
            self._active_neural = max(0, self._active_neural - 1)
        self._neural_sem.release()


class MetricsTracker:
    """Thread-safe in-memory observability metrics tracker for operational telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.request_counts: dict[str, int] = {}
        self.mode_counts: dict[str, int] = {}
        self.status_code_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}
        self.latency_samples_ms: dict[str, list[float]] = {
            "total_request": [],
            "fvm_solve": [],
            "neural_forward": [],
            "physics_correction": [],
            "serialization": [],
        }

    def record_request(
        self, endpoint: str, mode: str | None, status_code: int, duration_ms: float
    ) -> None:
        with self._lock:
            self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
            if mode:
                self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
            sc_key = str(status_code)
            self.status_code_counts[sc_key] = self.status_code_counts.get(sc_key, 0) + 1

            samples = self.latency_samples_ms["total_request"]
            samples.append(duration_ms)
            if len(samples) > 1000:
                self.latency_samples_ms["total_request"] = samples[-1000:]

    def record_stage_latency(self, stage: str, duration_ms: float) -> None:
        with self._lock:
            if stage in self.latency_samples_ms:
                samples = self.latency_samples_ms[stage]
                samples.append(duration_ms)
                if len(samples) > 1000:
                    self.latency_samples_ms[stage] = samples[-1000:]

    def record_error(self, error_type: str) -> None:
        with self._lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.time() - self.start_time

            def stats_for(stage: str) -> dict[str, float]:
                s = self.latency_samples_ms.get(stage, [])
                if not s:
                    return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
                s_sorted = sorted(s)
                n = len(s_sorted)
                return {
                    "count": n,
                    "mean_ms": round(float(sum(s_sorted) / n), 2),
                    "p50_ms": round(float(s_sorted[int(0.50 * (n - 1))]), 2),
                    "p95_ms": round(float(s_sorted[int(0.95 * (n - 1))]), 2),
                    "p99_ms": round(float(s_sorted[int(0.99 * (n - 1))]), 2),
                }

            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": sum(self.request_counts.values()),
                "requests_by_endpoint": dict(self.request_counts),
                "requests_by_mode": dict(self.mode_counts),
                "requests_by_status": dict(self.status_code_counts),
                "errors_by_type": dict(self.error_counts),
                "latency_summary": {stage: stats_for(stage) for stage in self.latency_samples_ms},
            }


# Global singleton settings and governors
SETTINGS = DeploymentSettings()
GOVERNOR = ConcurrencyGovernor(SETTINGS)
METRICS = MetricsTracker()
