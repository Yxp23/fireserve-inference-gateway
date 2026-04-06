"""
FireServe — GPU-Accelerated ML Inference Gateway
A production-grade API gateway for serving image generation models
with intelligent batching, health-aware routing, and real-time metrics.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import time

from app.models import (
    GenerateRequest, GenerateResponse, HealthResponse,
    MetricsResponse, BatchStatus
)
from app.gateway import InferenceGateway
from app.metrics import MetricsCollector

metrics = MetricsCollector()
gateway = InferenceGateway(metrics=metrics)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    await gateway.start()
    yield
    await gateway.shutdown()


app = FastAPI(
    title="FireServe — ML Inference Gateway",
    description=(
        "Production-grade inference gateway with request batching, "
        "health-aware routing, automatic retries, and real-time metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── Inference Endpoints ─────────────────────────


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_image(req: GenerateRequest):
    """
    Submit an image generation request.
    The gateway handles batching, routing, and retries transparently.
    """
    start = time.monotonic()
    try:
        result = await gateway.submit(req)
        latency_ms = (time.monotonic() - start) * 1000
        metrics.record_request(latency_ms=latency_ms, success=True)
        return result
    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        metrics.record_request(latency_ms=latency_ms, success=False)
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1/generate/batch")
async def generate_batch(requests: list[GenerateRequest]):
    """
    Submit a batch of generation requests.
    Processed concurrently with intelligent scheduling.
    """
    if len(requests) > 16:
        raise HTTPException(400, "Maximum batch size is 16")

    start = time.monotonic()
    results = await gateway.submit_batch(requests)
    total_ms = (time.monotonic() - start) * 1000

    return {
        "results": results,
        "batch_size": len(requests),
        "total_latency_ms": round(total_ms, 2),
        "avg_latency_ms": round(total_ms / len(requests), 2),
    }


# ───────────────────────── Health & Monitoring ─────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with backend status."""
    backend_health = await gateway.check_backends()
    all_healthy = all(b["healthy"] for b in backend_health)
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        backends=backend_health,
        active_requests=gateway.active_requests,
        queue_depth=gateway.queue_depth,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Real-time performance metrics."""
    return metrics.snapshot()


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus-compatible metrics export."""
    return JSONResponse(
        content=metrics.prometheus_format(),
        media_type="text/plain",
    )


# ───────────────────────── Backend Management ──────────────────────────


@app.post("/backends/register")
async def register_backend(url: str, gpu_type: str = "T4", vram_gb: float = 16.0):
    """Register a new GPU inference backend."""
    backend_id = await gateway.register_backend(url, gpu_type, vram_gb)
    return {"backend_id": backend_id, "status": "registered"}


@app.delete("/backends/{backend_id}")
async def remove_backend(backend_id: str):
    """Remove a backend from the pool."""
    await gateway.remove_backend(backend_id)
    return {"status": "removed"}
