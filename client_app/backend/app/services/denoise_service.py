"""Denoise service layer — per API_CONTRACT.md § Step 5.

Loads the Canny output, applies connected-component area filtering,
saves the denoised result, and returns the accessible URL.

Multi-layer mode: when layer_index is not None, operates on the per-layer
lineart result (_canny_L{N}.png) and saves as _denoised_L{N}.png.
"""

import os
import time

import cv2
import numpy as np
from fastapi import HTTPException
from loguru import logger

from app.models.requests import DenoiseParams
from app.models.responses import PipelineStepResponse

OUTPUTS_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

# ── Lazy-import the denoise engine ──────────────────────────────
try:
    from app.utils.denoise import denoise_binary as _engine
except ImportError:
    _engine = None


def _find_canny_image(image_id: str) -> str:
    """Locate the Canny output for a given image_id.

    Denoise runs on the Canny result (Step 4 output).
    """
    path = os.path.join(OUTPUTS_DIR, f"{image_id}_canny.png")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"Canny result not found for image_id={image_id}. "
        f"Run edge detection (Step 4) first."
    )


def _find_canny_image_for_layer(image_id: str, layer_index: int) -> str:
    """Locate the per-layer Canny output."""
    path = os.path.join(
        OUTPUTS_DIR, f"{image_id}_canny_L{layer_index}.png"
    )
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"Canny result not found for image_id={image_id} layer_index={layer_index}. "
        f"Run edge detection (Step 4) on this layer first."
    )


def _find_original_image(image_id: str) -> str:
    """Locate the original uploaded image by image_id in outputs/."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(OUTPUTS_DIR, f"{image_id}_original{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Original image not found for image_id={image_id}")


def process_denoise(params: DenoiseParams) -> PipelineStepResponse:
    """Run the denoise pipeline step.

    Loads the Canny result as grayscale binary, applies denoise_binary(),
    and saves the result.
    """
    if _engine is None:
        raise HTTPException(
            status_code=501,
            detail="denoise engine not found — check app/utils/denoise.py",
        )

    t0 = time.perf_counter()

    # 1. Locate and load the Canny lineart image
    if params.layer_index is not None:
        try:
            input_path = _find_canny_image_for_layer(params.image_id, params.layer_index)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        try:
            input_path = _find_canny_image(params.image_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    logger.info("Denoise input: {}", input_path)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode: {input_path}")

    logger.info(
        "Denoise params | image_id={} shape={} min_component_area={} layer_index={}",
        params.image_id, img.shape, params.min_component_area, params.layer_index,
    )

    # 2. Run the denoise engine
    before_px = int((img > 0).sum())

    try:
        result = _engine(img, min_component_area=params.min_component_area)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not isinstance(result, np.ndarray):
        raise HTTPException(
            status_code=500,
            detail=f"Engine returned {type(result).__name__}, expected numpy.ndarray",
        )

    after_px = int((result > 0).sum())
    removed_px = before_px - after_px

    # 3. Save result (atomic write: tmp with .png extension → rename)
    if params.layer_index is not None:
        result_path = os.path.join(
            OUTPUTS_DIR, f"{params.image_id}_denoised_L{params.layer_index}.png"
        )
        tmp_path = os.path.join(
            OUTPUTS_DIR, f".tmp_{params.image_id}_denoised_L{params.layer_index}.png"
        )
        result_url = f"/outputs/{params.image_id}_denoised_L{params.layer_index}.png"
    else:
        result_path = os.path.join(OUTPUTS_DIR, f"{params.image_id}_denoised.png")
        tmp_path = os.path.join(OUTPUTS_DIR, f".tmp_{params.image_id}_denoised.png")
        result_url = f"/outputs/{params.image_id}_denoised.png"

    cv2.imwrite(tmp_path, result)
    os.replace(tmp_path, result_path)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Denoise saved | path={} shape={} before={}px after={}px removed={}px time={}ms",
        result_path, result.shape, before_px, after_px, removed_px, elapsed_ms,
    )

    return PipelineStepResponse(
        result_url=result_url,
        processing_time_ms=elapsed_ms,
    )
