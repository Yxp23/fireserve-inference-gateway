"""
Real-time metrics collection with percentile tracking.

Tracks:
- Request counts (success/failure)
- Latency percentiles (p50, p95, p99)
- Throughput (RPS)
- Per-backend statistics
- Error rates with sliding windows
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field

from app.models import MetricsResponse


@dataclass
class BackendMetrics:
    """Per-backend metrics."""
    backend_id: str
    total_requests: int = 0
    failures: int = 0
    events: list = field(default_factory=list)


class MetricsCollector:
    """Thread-safe metrics collector with sliding window."""

    WINDOW_SIZE = 1000  # Keep last N latency samples

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._total_requests = 0
        self._successful = 0
        self._failed = 0
        self._latencies: deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._request_times: deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._backend_metrics: dict[str, BackendMetrics] = {}

    def record_request(self, latency_ms: float, success: bool):
        with self._lock:
            self._total_requests += 1
            self._latencies.append(latency_ms)
            self._request_times.append(time.monotonic())
            if success:
                self._successful += 1
            else:
                self._failed += 1

    def record_backend_event(self, backend_id: str, event: str):
        with self._lock:
            if backend_id not in self._backend_metrics:
                self._backend_metrics[backend_id] = BackendMetrics(backend_id=backend_id)
            bm = self._backend_metrics[backend_id]
            bm.events.append({"event": event, "time": time.monotonic()})
            if event == "success":
                bm.total_requests += 1
            elif event == "failure":
                bm.total_requests += 1
                bm.failures += 1

    def _percentile(self, data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * pct / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]

    def _calc_rps(self) -> float:
        """Requests per second over the last 60 seconds."""
        now = time.monotonic()
        cutoff = now - 60
        recent = [t for t in self._request_times if t > cutoff]
        if len(recent) < 2:
            return 0.0
        window = now - recent[0]
        return len(recent) / window if window > 0 else 0.0

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            latencies = list(self._latencies)
            uptime = time.monotonic() - self._start_time
            rps = self._calc_rps()
            error_rate = (
                self._failed / self._total_requests
                if self._total_requests > 0 else 0.0
            )

            backend_stats = [
                {
                    "backend_id": bm.backend_id,
                    "total_requests": bm.total_requests,
                    "failures": bm.failures,
                    "error_rate": round(
                        bm.failures / bm.total_requests
                        if bm.total_requests > 0 else 0.0, 4
                    ),
                }
                for bm in self._backend_metrics.values()
            ]

            return MetricsResponse(
                total_requests=self._total_requests,
                successful_requests=self._successful,
                failed_requests=self._failed,
                avg_latency_ms=round(
                    sum(latencies) / len(latencies) if latencies else 0.0, 2
                ),
                p50_latency_ms=round(self._percentile(latencies, 50), 2),
                p95_latency_ms=round(self._percentile(latencies, 95), 2),
                p99_latency_ms=round(self._percentile(latencies, 99), 2),
                requests_per_second=round(rps, 2),
                uptime_seconds=round(uptime, 2),
                error_rate=round(error_rate, 4),
                backend_stats=backend_stats,
            )

    def prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        s = self.snapshot()
        lines = [
            f"# HELP fireserve_requests_total Total requests processed",
            f"# TYPE fireserve_requests_total counter",
            f"fireserve_requests_total {s.total_requests}",
            f"fireserve_requests_successful {s.successful_requests}",
            f"fireserve_requests_failed {s.failed_requests}",
            f"",
            f"# HELP fireserve_latency_ms Request latency in milliseconds",
            f"# TYPE fireserve_latency_ms summary",
            f'fireserve_latency_ms{{quantile="0.5"}} {s.p50_latency_ms}',
            f'fireserve_latency_ms{{quantile="0.95"}} {s.p95_latency_ms}',
            f'fireserve_latency_ms{{quantile="0.99"}} {s.p99_latency_ms}',
            f"",
            f"# HELP fireserve_rps Requests per second",
            f"# TYPE fireserve_rps gauge",
            f"fireserve_rps {s.requests_per_second}",
            f"",
            f"# HELP fireserve_error_rate Error rate",
            f"# TYPE fireserve_error_rate gauge",
            f"fireserve_error_rate {s.error_rate}",
        ]
        return "\n".join(lines)
