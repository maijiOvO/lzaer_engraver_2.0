"""Canny edge-detection service layer.

Receives image metadata, delegates to the canny_lineart engine,
saves the binary result to outputs/, and returns the accessible URL.

Engine: canny_lineart (CLAHE + Canny) — pure CPU, no GPU/model needed.

Multi-layer mode: when layer_index is not None, the mask for that layer
is loaded and used to isolate the layer region before edge detection.
"""

import os
import time

import cv2
import numpy as np
from fastapi import HTTPException
from loguru import logger

from app.models.requests import CannyParams
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


def _find_layer_mask(image_id: str, layer_index: int) -> str:
    """Locate the layer mask for a given layer_index."""
    path = os.path.join(OUTPUTS_DIR, f"{image_id}_mask_{layer_index}.png")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"Layer mask not found for image_id={image_id} layer_index={layer_index}. "
        f"Run segmentation (Step 2-3) first."
    )


def process_canny(params: CannyParams) -> PipelineStepResponse:
    """Run the Canny edge-detection pipeline step.

    Loads the original image as an OpenCV BGR numpy array via cv2.imread(),
    passes it to canny_lineart(), and saves the binary result with cv2.imwrite().

    When layer_index is not None: applies the layer mask to isolate the target
    region before edge detection, then crops to the layer bounding box plus
    frame margin.
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

    # ── Multi-layer: apply mask before edge detection ──────────────
    if params.layer_index is not None:
        try:
            mask_path = _find_layer_mask(params.image_id, params.layer_index)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=400, detail=f"Failed to decode mask: {mask_path}")

        # Apply mask: set non-layer pixels to white (background)
        masked_image = image.copy()
        masked_image[mask == 0] = 255

        logger.info(
            "Canny multi-layer | image_id={} layer_index={} shape={} mask={}",
            params.image_id, params.layer_index, image.shape, mask_path,
        )

        input_image = masked_image
    else:
        input_image = image

    logger.info(
        "Canny input | image_id={} shape={} dtype={} low={} high={} smooth={} layer_index={}",
        params.image_id, input_image.shape, input_image.dtype,
        params.low, params.high, params.smooth_level, params.layer_index,
    )

    # 2. Call the canny_lineart engine
    result = _engine(
        input_image,
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

    # 4. Save result (atomic write: tmp with .png extension → rename)
    if params.layer_index is not None:
        result_path = os.path.join(
            OUTPUTS_DIR, f"{params.image_id}_canny_L{params.layer_index}.png"
        )
        tmp_path = os.path.join(
            OUTPUTS_DIR, f".tmp_{params.image_id}_canny_L{params.layer_index}.png"
        )
        result_url = f"/outputs/{params.image_id}_canny_L{params.layer_index}.png"
    else:
        result_path = os.path.join(OUTPUTS_DIR, f"{params.image_id}_canny.png")
        tmp_path = os.path.join(OUTPUTS_DIR, f".tmp_{params.image_id}_canny.png")
        result_url = f"/outputs/{params.image_id}_canny.png"

    cv2.imwrite(tmp_path, result)
    os.replace(tmp_path, result_path)
    logger.info("Canny result saved | path={} shape={} dtype={}", result_path, result.shape, result.dtype)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return PipelineStepResponse(
        result_url=result_url,
        processing_time_ms=elapsed_ms,
    )
