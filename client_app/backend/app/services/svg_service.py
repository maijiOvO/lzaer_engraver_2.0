"""SVG generation service layer.

Locates the connectivity-repaired image, generates a pure-stroke
cubic Bezier SVG, saves it to outputs/, and returns the URL + stats.

Multi-layer mode: when layer_index is not None, operates on per-layer
outputs and saves as _L{N}.svg. A separate process_svg_multi() function
generates all layers and merges them into a single tiled SVG.
"""

import os
import re
import time

import cv2
import numpy as np
from fastapi import HTTPException
from loguru import logger

from app.models.requests import SvgParams
from app.models.responses import SvgResponse, MultiLayerSvgResponse
from app.utils.layer_frame import compute_frame_width, merge_layer_svgs

OUTPUTS_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

try:
    from app.utils.svg_generator import generate_svg as _engine
except ImportError:
    _engine = None


def _find_input_image(image_id: str) -> str:
    """Find the best input image for SVG generation.

    Priority: _connected (Step 6) > _denoised (Step 5) > _canny (skip denoise).
    """
    for suffix in ("_connected.png", "_denoised.png", "_canny.png"):
        path = os.path.join(OUTPUTS_DIR, f"{image_id}{suffix}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No canny, denoised, or connected image found for image_id={image_id}"
    )


def _find_input_image_for_layer(image_id: str, layer_index: int) -> str:
    """Find the best per-layer input for SVG generation."""
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


def _find_original_image(image_id: str) -> str:
    """Locate the original uploaded image by image_id in outputs/."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(OUTPUTS_DIR, f"{image_id}_original{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Original image not found for image_id={image_id}")


def _count_svg_stats(svg_str: str) -> tuple[int, int]:
    """Count total paths and anchor points in an SVG string."""
    total_paths = svg_str.count("M ")
    total_points = len(re.findall(r'[MLC]\s+[\d.]+', svg_str))
    return total_paths, total_points


def process_svg(params: SvgParams) -> SvgResponse:
    """Generate SVG for a single image or single layer."""
    if _engine is None:
        raise HTTPException(
            status_code=501,
            detail="svg_generator engine not found — check app/utils/svg_generator.py",
        )

    t0 = time.perf_counter()

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

    logger.info("SVG input: {}", input_path)

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode: {input_path}")

    logger.info(
        "SVG params | image_id={} shape={} simplify_tolerance={} layer_index={}",
        params.image_id, img.shape, params.simplify_tolerance, params.layer_index,
    )

    svg_str = _engine(img, simplify_tolerance=params.simplify_tolerance)

    total_paths, total_points = _count_svg_stats(svg_str)

    # Save
    if params.layer_index is not None:
        result_path = os.path.join(
            OUTPUTS_DIR, f"{params.image_id}_L{params.layer_index}.svg"
        )
        tmp_path = os.path.join(
            OUTPUTS_DIR, f".tmp_{params.image_id}_L{params.layer_index}.svg"
        )
        svg_url = f"/outputs/{params.image_id}_L{params.layer_index}.svg"
    else:
        result_path = os.path.join(OUTPUTS_DIR, f"{params.image_id}.svg")
        tmp_path = os.path.join(OUTPUTS_DIR, f".tmp_{params.image_id}.svg")
        svg_url = f"/outputs/{params.image_id}.svg"

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    os.replace(tmp_path, result_path)
    logger.info(
        "SVG saved | path={} paths={} points={} bytes={}",
        result_path, total_paths, total_points, len(svg_str),
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return SvgResponse(
        svg_url=svg_url,
        total_paths=total_paths,
        total_points=total_points,
        processing_time_ms=elapsed_ms,
    )


def process_svg_multi(image_id: str, n_layers: int, simplify_tolerance: float) -> MultiLayerSvgResponse:
    """Generate per-layer SVGs and merge into one tiled multi-layer SVG.

    Args:
        image_id: UUID of the original image.
        n_layers: number of layers to process.
        simplify_tolerance: Douglas-Peucker epsilon.

    Returns:
        MultiLayerSvgResponse with merged SVG URL and per-layer stats.
    """
    if _engine is None:
        raise HTTPException(
            status_code=501,
            detail="svg_generator engine not found — check app/utils/svg_generator.py",
        )

    t0 = time.perf_counter()

    # Load original image to compute frame width
    try:
        original_path = _find_original_image(image_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image = cv2.imread(original_path)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Failed to decode: {original_path}")

    frame_w = compute_frame_width(image.shape)
    logger.info("Multi-layer SVG | image_id={} n_layers={} frame_w={}", image_id, n_layers, frame_w)

    layer_svgs: list[str] = []
    layer_bboxes: list[tuple[int, int, int, int]] = []
    per_layer_paths: list[int] = []
    per_layer_points: list[int] = []

    for li in range(n_layers):
        try:
            input_path = _find_input_image_for_layer(image_id, li)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Layer {li} input not found: {e}",
            )

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Failed to decode: {input_path}")

        logger.info(
            "SVG layer {} | image_id={} shape={}",
            li, image_id, img.shape,
        )

        svg_str = _engine(img, simplify_tolerance=simplify_tolerance)
        layer_svgs.append(svg_str)

        total_paths, total_points = _count_svg_stats(svg_str)
        per_layer_paths.append(total_paths)
        per_layer_points.append(total_points)

        # Compute bounding box of foreground pixels
        ys, xs = np.where(img > 0)
        if len(ys) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        else:
            bbox = (0, 0, img.shape[1], img.shape[0])
        layer_bboxes.append(bbox)

    # Merge into single tiled SVG (horizontal by default)
    merged_svg = merge_layer_svgs(
        layer_svgs,
        layer_bboxes,
        frame_w,
        direction="horizontal",
    )

    # Save merged SVG
    result_path = os.path.join(OUTPUTS_DIR, f"{image_id}_multi.svg")
    tmp_path = os.path.join(OUTPUTS_DIR, f".tmp_{image_id}_multi.svg")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(merged_svg)
    os.replace(tmp_path, result_path)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Multi-layer SVG saved | path={} layers={} total_paths={} total_points={} bytes={} time={}ms",
        result_path, n_layers, sum(per_layer_paths), sum(per_layer_points), len(merged_svg), elapsed_ms,
    )

    return MultiLayerSvgResponse(
        svg_url=f"/outputs/{image_id}_multi.svg",
        per_layer_paths=per_layer_paths,
        total_paths=sum(per_layer_paths),
        total_points=sum(per_layer_points),
        processing_time_ms=elapsed_ms,
    )
