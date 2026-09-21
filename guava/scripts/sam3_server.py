"""Guava SAM3 perception server.

Serves a text-prompt segmentation endpoint compatible with guava.perception.sam3.SAM3Solver.

Usage:
    .venv-sam3/bin/python -m guava.scripts.sam3_server
    .venv-sam3/bin/python -m guava.scripts.sam3_server --port 8114 --device cuda
"""
from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

_PROCESSOR: Any | None = None
_DEVICE: str = "cuda"
_GPU_SEMAPHORE = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# Request / response models  (must match guava.perception.sam3.SAM3Solver)
# ---------------------------------------------------------------------------

class SegmentRequest(BaseModel):
    image_b64: str
    text_prompt: str
    top_k: int = 1
    """Number of highest-scoring masks to return. SAM3 emits ~200 proposals and
    every consumer in guava uses only the best one, so returning them all costs
    ~2.6 s per request in Python-object conversion and JSON encoding alone.
    Set <= 0 to return every mask (the old behaviour)."""


class MaskResult(BaseModel):
    mask: list[list[bool]]   # H x W boolean grid
    score: float


class SegmentResponse(BaseModel):
    masks: list[MaskResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


async def _run_on_gpu(fn, *args, **kwargs):
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def _decode_image(b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _do_segment(pil_image: Image.Image, text_prompt: str,
                top_k: int = 1) -> SegmentResponse:
    dtype_ctx = (
        torch.autocast(_DEVICE, dtype=torch.bfloat16)
        if "cuda" in _DEVICE
        else torch.autocast("cpu")
    )
    with dtype_ctx:
        state = _PROCESSOR.set_image(pil_image)
        output = _PROCESSOR.set_text_prompt(state=state, prompt=text_prompt)

    masks_t = output.get("masks")
    scores_t = output.get("scores")
    if masks_t is None:
        return SegmentResponse(masks=[])

    # Rank and truncate BEFORE moving masks to the host. Scores are a handful of
    # floats, so ordering them is free; the masks are (N, 1, H, W) and every one
    # converted to nested Python lists costs ~3 ms plus JSON encoding.
    scores_np = _to_numpy(scores_t)
    order = np.argsort(-scores_np)
    if top_k > 0:
        order = order[:top_k]

    if hasattr(masks_t, "detach"):
        idx = torch.as_tensor(np.ascontiguousarray(order), device=masks_t.device)
        masks_np = _to_numpy(masks_t.index_select(0, idx))
    else:
        masks_np = np.asarray(masks_t)[order]
    scores_np = scores_np[order]

    # (N, 1, H, W) → (N, H, W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    # Already ordered by descending score.
    return SegmentResponse(masks=[
        MaskResult(mask=(masks_np[i] > 0).tolist(), score=float(scores_np[i]))
        for i in range(len(scores_np))
    ])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    if _PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not initialised")
    pil_image = _decode_image(req.image_b64)
    try:
        return await _run_on_gpu(_do_segment, pil_image, req.text_prompt, req.top_k)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    device: str = "cuda",
    port: int = 8114,
    host: str = "127.0.0.1",
) -> None:
    global _PROCESSOR, _DEVICE
    _DEVICE = device

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("Loading SAM3 model on %s...", device)
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model()
        if hasattr(model, "to"):
            model = model.to(device)
        _PROCESSOR = Sam3Processor(model, confidence_threshold=0.0)
    except Exception as e:
        logger.error("Failed to load SAM3 model: %s", e)
        raise

    logger.info("SAM3 ready. Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
