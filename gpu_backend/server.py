"""
FireServe GPU Backend — Runs on Google Colab (Free T4 GPU)

This is the actual model-serving component that runs on a GPU.
The gateway routes requests to this backend for inference.

Usage (Google Colab):
    1. Open this file in Colab
    2. Enable GPU runtime (Runtime > Change runtime type > T4 GPU)
    3. Run all cells
    4. Copy the ngrok URL and register it with the gateway
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import base64
import io
import time
import uvicorn

app = FastAPI(title="FireServe GPU Backend")

# ── Global model reference ──
pipe = None
device = None


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    seed: int = 42


class GenerateResponse(BaseModel):
    image_base64: str
    inference_time_ms: float
    gpu_type: str
    model_id: str


def load_model():
    """Load SDXL-Turbo model onto GPU."""
    global pipe, device
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Loading model on: {gpu_name}")

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)

    # Warmup inference
    print("Running warmup inference...")
    _ = pipe("warmup", num_inference_steps=1, width=256, height=256)
    print(f"Model loaded and ready on {gpu_name}!")


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9 if gpu_available else 0
    vram_used = torch.cuda.memory_allocated(0) / 1e9 if gpu_available else 0

    return {
        "status": "healthy" if pipe is not None else "loading",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_total_gb": round(vram_total, 2),
        "vram_used_gb": round(vram_used, 2),
        "vram_free_gb": round(vram_total - vram_used, 2),
        "model_loaded": pipe is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if pipe is None:
        raise Exception("Model not loaded yet")

    generator = torch.Generator(device=device).manual_seed(req.seed)

    start = time.monotonic()
    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt if req.negative_prompt else None,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    ).images[0]
    inference_ms = (time.monotonic() - start) * 1000

    # Encode image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    return GenerateResponse(
        image_base64=img_b64,
        inference_time_ms=round(inference_ms, 2),
        gpu_type=gpu_name,
        model_id="stabilityai/sdxl-turbo",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
