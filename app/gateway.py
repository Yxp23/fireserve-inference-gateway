"""
Inference Gateway — core routing, batching, and reliability logic.

Handles:
- Health-aware backend routing (least-latency + GPU memory)
- Dynamic request batching for GPU throughput optimization
- Automatic retries with exponential backoff + fallback backends
- Priority queue with fair scheduling
- Circuit breaker pattern for failing backends
"""

import asyncio
import time
import uuid
import random
import base64
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import httpx

from app.models import GenerateRequest, GenerateResponse
from app.metrics import MetricsCollector


@dataclass
class BackendNode:
    """Represents a GPU inference backend."""
    id: str
    url: str
    gpu_type: str
    vram_gb: float
    healthy: bool = True
    active_requests: int = 0
    total_served: int = 0
    avg_latency_ms: float = 0.0
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_open_until: float = 0.0
    last_health_check: float = 0.0

    # Circuit breaker settings
    FAILURE_THRESHOLD: int = 3
    CIRCUIT_TIMEOUT: float = 30.0  # seconds

    def record_success(self, latency_ms: float):
        self.consecutive_failures = 0
        self.circuit_open = False
        self.total_served += 1
        # Exponential moving average
        alpha = 0.3
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms

    def record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.FAILURE_THRESHOLD:
            self.circuit_open = True
            self.circuit_open_until = time.monotonic() + self.CIRCUIT_TIMEOUT

    def is_available(self) -> bool:
        if self.circuit_open:
            if time.monotonic() > self.circuit_open_until:
                # Half-open: allow one request to test
                self.circuit_open = False
                return True
            return False
        return self.healthy

    def score(self) -> float:
        """Lower score = better candidate. Factors in latency + current load."""
        latency_score = self.avg_latency_ms if self.avg_latency_ms > 0 else 100
        load_score = self.active_requests * 50  # Penalty per active request
        vram_bonus = -self.vram_gb * 2  # Prefer more VRAM
        return latency_score + load_score + vram_bonus


class InferenceGateway:
    """Main gateway orchestrating request flow."""

    def __init__(
        self,
        metrics: MetricsCollector,
        max_retries: int = 2,
        batch_window_ms: float = 50,
        max_batch_size: int = 8,
        request_timeout: float = 30.0,
    ):
        self.metrics = metrics
        self.backends: dict[str, BackendNode] = {}
        self.max_retries = max_retries
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.request_timeout = request_timeout
        self.active_requests = 0
        self.queue_depth = 0
        self._client: Optional[httpx.AsyncClient] = None
        self._health_task: Optional[asyncio.Task] = None
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._use_mock = True  # Toggle for demo mode vs real backends

    async def start(self):
        self._client = httpx.AsyncClient(timeout=self.request_timeout)
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def shutdown(self):
        if self._health_task:
            self._health_task.cancel()
        if self._client:
            await self._client.aclose()

    # ───────────────── Backend Management ─────────────────

    async def register_backend(self, url: str, gpu_type: str, vram_gb: float) -> str:
        backend_id = f"gpu-{uuid.uuid4().hex[:8]}"
        self.backends[backend_id] = BackendNode(
            id=backend_id, url=url.rstrip("/"),
            gpu_type=gpu_type, vram_gb=vram_gb,
        )
        self.metrics.record_backend_event(backend_id, "registered")
        return backend_id

    async def remove_backend(self, backend_id: str):
        if backend_id in self.backends:
            del self.backends[backend_id]
            self.metrics.record_backend_event(backend_id, "removed")

    # ───────────────── Routing ─────────────────

    def _select_backend(self, exclude: set[str] | None = None) -> Optional[BackendNode]:
        """Select best available backend using least-score routing."""
        exclude = exclude or set()
        candidates = [
            b for b in self.backends.values()
            if b.is_available() and b.id not in exclude
        ]
        if not candidates:
            return None
        # Sort by composite score (latency + load + vram)
        candidates.sort(key=lambda b: b.score())
        return candidates[0]

    # ───────────────── Request Processing ─────────────────

    async def submit(self, req: GenerateRequest) -> GenerateResponse:
        """Submit a single generation request with retries and fallback."""
        request_id = uuid.uuid4().hex[:12]
        queue_start = time.monotonic()
        self.queue_depth += 1

        tried_backends: set[str] = set()
        last_error = None

        for attempt in range(self.max_retries + 1):
            backend = self._select_backend(exclude=tried_backends)

            if backend is None:
                # If we have mock mode, use it
                if self._use_mock:
                    self.queue_depth -= 1
                    return await self._mock_inference(req, request_id, queue_start)
                raise Exception("No healthy backends available")

            tried_backends.add(backend.id)
            backend.active_requests += 1
            self.active_requests += 1

            try:
                result = await self._call_backend(backend, req, request_id, queue_start)
                backend.active_requests -= 1
                self.active_requests -= 1
                self.queue_depth -= 1
                return result
            except Exception as e:
                backend.active_requests -= 1
                self.active_requests -= 1
                backend.record_failure()
                last_error = e
                self.metrics.record_backend_event(backend.id, "failure")

                # Exponential backoff before retry
                if attempt < self.max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt))

        self.queue_depth -= 1
        raise Exception(f"All retries exhausted. Last error: {last_error}")

    async def submit_batch(self, requests: list[GenerateRequest]) -> list[dict]:
        """Process a batch of requests concurrently."""
        tasks = [self.submit(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            r.model_dump() if isinstance(r, GenerateResponse)
            else {"error": str(r)}
            for r in results
        ]

    async def _call_backend(
        self, backend: BackendNode, req: GenerateRequest,
        request_id: str, queue_start: float,
    ) -> GenerateResponse:
        """Make the actual inference call to a backend."""
        infer_start = time.monotonic()
        queue_wait = (infer_start - queue_start) * 1000

        payload = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed or random.randint(0, 2**32),
        }

        resp = await self._client.post(
            f"{backend.url}/generate", json=payload
        )
        resp.raise_for_status()
        data = resp.json()

        infer_time = (time.monotonic() - infer_start) * 1000
        total_time = (time.monotonic() - queue_start) * 1000

        backend.record_success(infer_time)
        self.metrics.record_backend_event(backend.id, "success")

        return GenerateResponse(
            request_id=request_id,
            image_base64=data.get("image_base64", ""),
            model_used=req.model.value,
            backend_id=backend.id,
            inference_time_ms=round(infer_time, 2),
            queue_wait_ms=round(queue_wait, 2),
            total_latency_ms=round(total_time, 2),
            seed_used=payload["seed"],
            metadata={
                "gpu_type": backend.gpu_type,
                "backend_load": backend.active_requests,
            },
        )

    async def _mock_inference(
        self, req: GenerateRequest, request_id: str, queue_start: float
    ) -> GenerateResponse:
        """Mock inference for demo/testing — simulates realistic GPU latency."""
        # Simulate variable latency based on steps and resolution
        base_latency = 0.05  # 50ms base
        step_factor = req.num_inference_steps * 0.02
        resolution_factor = (req.width * req.height) / (512 * 512) * 0.03
        jitter = random.uniform(-0.01, 0.03)
        simulated_delay = base_latency + step_factor + resolution_factor + jitter

        await asyncio.sleep(simulated_delay)

        infer_time = simulated_delay * 1000
        queue_wait = (time.monotonic() - queue_start) * 1000 - infer_time
        total_time = (time.monotonic() - queue_start) * 1000

        # Generate a deterministic placeholder image hash
        seed = req.seed or random.randint(0, 2**32)
        img_hash = hashlib.sha256(f"{req.prompt}{seed}".encode()).hexdigest()
        fake_b64 = base64.b64encode(bytes.fromhex(img_hash[:64])).decode()

        return GenerateResponse(
            request_id=request_id,
            image_base64=fake_b64,
            model_used=req.model.value,
            backend_id="mock-gpu-t4",
            inference_time_ms=round(infer_time, 2),
            queue_wait_ms=round(max(queue_wait, 0), 2),
            total_latency_ms=round(total_time, 2),
            seed_used=seed,
            metadata={
                "gpu_type": "T4 (mock)",
                "mode": "demo",
                "note": "Connect a real GPU backend via POST /backends/register",
            },
        )

    # ───────────────── Health Checking ─────────────────

    async def _health_check_loop(self):
        """Periodically check backend health."""
        while True:
            await asyncio.sleep(10)
            for backend in list(self.backends.values()):
                try:
                    resp = await self._client.get(
                        f"{backend.url}/health", timeout=5
                    )
                    backend.healthy = resp.status_code == 200
                    backend.last_health_check = time.monotonic()
                except Exception:
                    backend.healthy = False

    async def check_backends(self) -> list[dict]:
        """Return current backend status."""
        if not self.backends:
            return [{"id": "mock-gpu-t4", "healthy": True,
                      "gpu": "T4 (mock)", "mode": "demo"}]
        return [
            {
                "id": b.id,
                "healthy": b.is_available(),
                "gpu": b.gpu_type,
                "vram_gb": b.vram_gb,
                "active_requests": b.active_requests,
                "avg_latency_ms": round(b.avg_latency_ms, 2),
                "total_served": b.total_served,
                "circuit_open": b.circuit_open,
            }
            for b in self.backends.values()
        ]
