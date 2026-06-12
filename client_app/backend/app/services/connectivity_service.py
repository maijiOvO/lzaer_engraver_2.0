"""Connectivity repair service layer.

Receives an image_id, locates the most recent pipeline output
(canny → connectivity chain), runs connectivity repair, saves result.

Multi-layer mode: when layer_index is not None, operates on per-layer
outputs and saves as _connected_L{N}.png.
"""

import os
import time

import cv2
import numpy as np
from fastapi import HTTPException
from loguru import logger

from app.models.requests import ConnectivityParams
from app.models.responses import ConnectivityResponse

OUTPUTS_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

# ── Lazy-import engine ──────────────────────────────────────────
try:
    from app.utils.connectivity import repair_connectivity as _engine
except ImportError:
    _engine = None


def _find_input_image(image_id: str) -> str:
    """Find the best input image for connectivity repair.

    Priority: _connected (re-run) > _denoised (Step 5) > _canny (skip denoise).
    """
    for suffix in ("_connected.png", "_denoised.png", "_canny.png"):
        path = os.path.join(OUTPUTS_DIR, f"{image_id}{suffix}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No canny or connected image found for image_id={image_id}"
    )


def _find_input_image_for_layer(image_id: str, layer_index: int) -> str:
    """Find the best per-layer input image for connectivity repair.

    Priority: _connected_L{N} > _denoised_L{N} > _canny_L{N}.
    """
    for suffix in (
        f"_connected_L{layer_index}.png",
        f"_denoised_L{layer_index}.png",
        f"_canny_L{layer_index}.png",
    ):
        path = os.path.join(OUTPUTS_DIR, f"{image_id}{suffix}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No canny/denoised/connected image found for "
        f"image_id={image_id} layer_index={layer_index}"
    )


def process_connectivity(params: ConnectivityParams) -> ConnectivityResponse:
    if _engine is None:
        raise HTTPException(
            status_code=501,
            detail="connectivity engine not found — check app/utils/connectivity.py",
        )

    t0 = time.perf_counter()

    # 1. Locate input
    if params.layer_index is not None:
        try:
            input_path = _find_input_image_for_layer(params.image_id, params.layer_index)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        try:
            input_path = _find_input_image(params.image_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    logger.info("Connectivity input: {}", input_path)

    # 2. Load as grayscale binary
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode: {input_path}")

    logger.info(
        "Connectivity params | image_id={} shape={} gap_tolerance={} layer_index={}",
        params.image_id, img.shape, params.gap_tolerance, params.layer_index,
    )

    # 3. Before/after pixel count for bridges
    before_px = int((img > 0).sum())

    # 4. Run engine
    result = _engine(img, gap_tolerance=params.gap_tolerance)

    if not isinstance(result, np.ndarray):
        raise HTTPException(
            status_code=500,
            detail=f"Engine returned {type(result).__name__}, expected ndarray",
        )

    after_px = int((result > 0).sum())
    bridges_built = after_px - before_px

    # 5. Save (overwrites if re-run)
    if params.layer_index is not None:
        result_path = os.path.join(
            OUTPUTS_DIR, f"{params.image_id}_connected_L{params.layer_index}.png"
        )
        tmp_path = os.path.join(
            OUTPUTS_DIR, f".tmp_{params.image_id}_connected_L{params.layer_index}.png"
        )
        result_url = f"/outputs/{params.image_id}_connected_L{params.layer_index}.png"
    else:
        result_path = os.path.join(OUTPUTS_DIR, f"{params.image_id}_connected.png")
        tmp_path = os.path.join(OUTPUTS_DIR, f".tmp_{params.image_id}_connected.png")
        result_url = f"/outputs/{params.image_id}_connected.png"

    cv2.imwrite(tmp_path, result)
    os.replace(tmp_path, result_path)
    logger.info(
        "Connectivity saved | path={} shape={} bridges={}",
        result_path, result.shape, bridges_built,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return ConnectivityResponse(
        result_url=result_url,
        bridges_built=bridges_built,
        processing_time_ms=elapsed_ms,
    )
