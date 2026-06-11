"""Canny edge-detection service layer.

Receives image metadata, delegates to the canny_lineart engine,
saves the binary result to outputs/, and returns the accessible URL.

Engine: canny_lineart (CLAHE + Canny) — pure CPU, no GPU/model needed.
"""

import os
import time

import cv2
import numpy as np
from fastapi import HTTPException
from loguru import logger

from app.models.requests import LineArtParams
from app.models.responses import PipelineStepResponse

OUTPUTS_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

# ── Lazy-import the canny_lineart engine ──────────────────────────
try:
    from app.utils.canny_lineart import canny_lineart as _engine
except ImportError:
    _engine = None


def _find_original_image(image_id: str) -> str:
    """Locate the original uploaded image by image_id in outputs/."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(OUTPUTS_DIR, f"{image_id}_original{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Original image not found for image_id={image_id}")


def process_canny(params: LineArtParams) -> PipelineStepResponse:
    """Run the Canny edge-detection pipeline step.

    Loads the original image as an OpenCV BGR numpy array via cv2.imread(),
    passes it to canny_lineart(), and saves the binary result with cv2.imwrite().
    """
    if _engine is None:
        raise HTTPException(
            status_code=501,
            detail="canny_lineart engine not found — check app/utils/canny_lineart.py",
        )

    t0 = time.perf_counter()

    # 1. Locate and load the original image as BGR numpy array
    try:
        original_path = _find_original_image(params.image_id)
        logger.info(f"Loading original image: {original_path}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image = cv2.imread(original_path)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {original_path}")

    logger.info(
        "Canny input | image_id={} shape={} dtype={} low={} high={} smooth={}",
        params.image_id, image.shape, image.dtype,
        params.low, params.high, params.smooth_level,
    )

    # 2. Call the canny_lineart engine
    result = _engine(
        image,
        low=params.low,
        high=params.high,
        smooth_level=params.smooth_level,
    )

    # 3. Validate result
    if not isinstance(result, np.ndarray):
        raise HTTPException(
            status_code=500,
            detail=f"Engine returned {type(result).__name__}, expected numpy.ndarray",
        )

    # 4. Save result
    result_path = os.path.join(OUTPUTS_DIR, f"{params.image_id}_canny.png")
    tmp_path = result_path + ".tmp"
    cv2.imwrite(tmp_path, result)
    os.replace(tmp_path, result_path)
    logger.info("Canny result saved | path={} shape={} dtype={}", result_path, result.shape, result.dtype)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return PipelineStepResponse(
        result_url=f"/outputs/{params.image_id}_canny.png",
        processing_time_ms=elapsed_ms,
    )
