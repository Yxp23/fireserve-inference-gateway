"""Pydantic models for the inference gateway."""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class ModelType(str, Enum):
    SDXL_TURBO = "sdxl-turbo"
    SD_1_5 = "stable-diffusion-v1-5"
    CUSTOM = "custom"


class GenerateRequest(BaseModel):
    """Image generation request."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for image generation")
    negative_prompt: str = Field(default="", max_length=1000, description="Negative prompt to avoid")
    model: ModelType = Field(default=ModelType.SDXL_TURBO, description="Model to use")
    width: int = Field(default=512, ge=256, le=1024, description="Image width")
    height: int = Field(default=512, ge=256, le=1024, description="Image height")
    num_inference_steps: int = Field(default=4, ge=1, le=50, description="Denoising steps")
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0, description="Classifier-free guidance")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    priority: int = Field(default=1, ge=1, le=3, description="Priority: 1=normal, 2=high, 3=critical")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a futuristic cityscape at sunset, digital art",
                "model": "sdxl-turbo",
                "width": 512,
                "height": 512,
                "num_inference_steps": 4,
            }
        }


class GenerateResponse(BaseModel):
    """Image generation response."""
    request_id: str
    image_base64: str
    model_used: str
    backend_id: str
    inference_time_ms: float
    queue_wait_ms: float
    total_latency_ms: float
    seed_used: int
    metadata: dict = {}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    backends: list[dict]
    active_requests: int
    queue_depth: int


class MetricsResponse(BaseModel):
    """Metrics snapshot."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    uptime_seconds: float
    error_rate: float
    backend_stats: list[dict]


class BatchStatus(BaseModel):
    """Batch processing status."""
    batch_id: str
    total: int
    completed: int
    failed: int
    status: str
