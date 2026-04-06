"""
Test suite for FireServe Inference Gateway.
Covers: routing, batching, retries, circuit breaker, metrics.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from httpx import Response

from app.gateway import InferenceGateway, BackendNode
from app.metrics import MetricsCollector
from app.models import GenerateRequest


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def gateway(metrics):
    gw = InferenceGateway(metrics=metrics)
    gw._use_mock = True
    return gw


@pytest.fixture
def sample_request():
    return GenerateRequest(
        prompt="a red fox in a snowy forest",
        width=512,
        height=512,
        num_inference_steps=4,
    )


# ───────────────── Mock Inference ─────────────────


class TestMockInference:
    """Test the mock inference path (no real GPU needed)."""

    @pytest.mark.asyncio
    async def test_single_request(self, gateway, sample_request):
        await gateway.start()
        result = await gateway.submit(sample_request)

        assert result.request_id
        assert result.image_base64
        assert result.inference_time_ms > 0
        assert result.total_latency_ms > 0
        assert result.model_used == "sdxl-turbo"
        await gateway.shutdown()

    @pytest.mark.asyncio
    async def test_batch_request(self, gateway):
        await gateway.start()
        requests = [
            GenerateRequest(prompt=f"test prompt {i}")
            for i in range(4)
        ]
        results = await gateway.submit_batch(requests)

        assert len(results) == 4
        for r in results:
            assert "error" not in r
            assert r["request_id"]
        await gateway.shutdown()

    @pytest.mark.asyncio
    async def test_seed_reproducibility(self, gateway):
        await gateway.start()
        req1 = GenerateRequest(prompt="test", seed=42)
        req2 = GenerateRequest(prompt="test", seed=42)

        r1 = await gateway.submit(req1)
        r2 = await gateway.submit(req2)

        assert r1.image_base64 == r2.image_base64
        assert r1.seed_used == r2.seed_used == 42
        await gateway.shutdown()


# ───────────────── Backend Routing ─────────────────


class TestRouting:
    """Test health-aware backend selection."""

    def test_score_prefers_low_latency(self):
        fast = BackendNode(id="fast", url="", gpu_type="A100",
                           vram_gb=40, avg_latency_ms=50)
        slow = BackendNode(id="slow", url="", gpu_type="T4",
                           vram_gb=16, avg_latency_ms=200)
        assert fast.score() < slow.score()

    def test_score_penalizes_load(self):
        idle = BackendNode(id="idle", url="", gpu_type="T4",
                           vram_gb=16, avg_latency_ms=100, active_requests=0)
        busy = BackendNode(id="busy", url="", gpu_type="T4",
                           vram_gb=16, avg_latency_ms=100, active_requests=3)
        assert idle.score() < busy.score()

    def test_score_prefers_more_vram(self):
        big = BackendNode(id="big", url="", gpu_type="A100",
                          vram_gb=40, avg_latency_ms=100)
        small = BackendNode(id="small", url="", gpu_type="T4",
                            vram_gb=16, avg_latency_ms=100)
        assert big.score() < small.score()

    def test_select_best_backend(self, gateway):
        gateway.backends = {
            "slow": BackendNode(id="slow", url="http://slow",
                                gpu_type="T4", vram_gb=16, avg_latency_ms=300),
            "fast": BackendNode(id="fast", url="http://fast",
                                gpu_type="A100", vram_gb=40, avg_latency_ms=50),
        }
        selected = gateway._select_backend()
        assert selected.id == "fast"

    def test_excludes_backends(self, gateway):
        gateway.backends = {
            "a": BackendNode(id="a", url="http://a",
                             gpu_type="T4", vram_gb=16),
            "b": BackendNode(id="b", url="http://b",
                             gpu_type="T4", vram_gb=16),
        }
        selected = gateway._select_backend(exclude={"a"})
        assert selected.id == "b"


# ───────────────── Circuit Breaker ─────────────────


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_opens_after_threshold(self):
        node = BackendNode(id="test", url="", gpu_type="T4", vram_gb=16)
        for _ in range(3):
            node.record_failure()
        assert node.circuit_open is True
        assert node.is_available() is False

    def test_success_resets_failures(self):
        node = BackendNode(id="test", url="", gpu_type="T4", vram_gb=16)
        node.record_failure()
        node.record_failure()
        node.record_success(100)
        assert node.consecutive_failures == 0
        assert node.circuit_open is False

    def test_half_open_after_timeout(self):
        node = BackendNode(id="test", url="", gpu_type="T4", vram_gb=16)
        for _ in range(3):
            node.record_failure()
        # Simulate timeout passed
        node.circuit_open_until = 0
        assert node.is_available() is True


# ───────────────── Metrics ─────────────────


class TestMetrics:
    def test_records_requests(self, metrics):
        metrics.record_request(100.0, True)
        metrics.record_request(200.0, True)
        metrics.record_request(50.0, False)

        snap = metrics.snapshot()
        assert snap.total_requests == 3
        assert snap.successful_requests == 2
        assert snap.failed_requests == 1

    def test_percentiles(self, metrics):
        for i in range(100):
            metrics.record_request(float(i), True)

        snap = metrics.snapshot()
        assert snap.p50_latency_ms == pytest.approx(50, abs=2)
        assert snap.p95_latency_ms == pytest.approx(95, abs=2)
        assert snap.p99_latency_ms == pytest.approx(99, abs=2)

    def test_error_rate(self, metrics):
        for _ in range(7):
            metrics.record_request(100, True)
        for _ in range(3):
            metrics.record_request(100, False)

        snap = metrics.snapshot()
        assert snap.error_rate == pytest.approx(0.3, abs=0.01)

    def test_prometheus_format(self, metrics):
        metrics.record_request(100, True)
        output = metrics.prometheus_format()
        assert "fireserve_requests_total 1" in output
        assert "fireserve_rps" in output

    def test_backend_events(self, metrics):
        metrics.record_backend_event("gpu-1", "success")
        metrics.record_backend_event("gpu-1", "success")
        metrics.record_backend_event("gpu-1", "failure")

        snap = metrics.snapshot()
        assert len(snap.backend_stats) == 1
        assert snap.backend_stats[0]["total_requests"] == 3
        assert snap.backend_stats[0]["failures"] == 1


# ───────────────── Input Validation ─────────────────


class TestValidation:
    def test_prompt_required(self):
        with pytest.raises(Exception):
            GenerateRequest(prompt="")

    def test_valid_dimensions(self):
        req = GenerateRequest(prompt="test", width=256, height=1024)
        assert req.width == 256
        assert req.height == 1024

    def test_rejects_oversized(self):
        with pytest.raises(Exception):
            GenerateRequest(prompt="test", width=2048)

    def test_default_values(self):
        req = GenerateRequest(prompt="test")
        assert req.width == 512
        assert req.height == 512
        assert req.num_inference_steps == 4
        assert req.guidance_scale == 0.0
        assert req.priority == 1
