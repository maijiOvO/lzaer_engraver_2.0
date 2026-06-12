"""SAM segmentation endpoint — per API_CONTRACT.md § Step 2-3."""

import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.requests import SegmentParams
from app.models.responses import SegmentResponse
from app.services.segmentation_service import process_segment

router = APIRouter(tags=["segmentation"])


@router.post("/pipeline/segment", response_model=SegmentResponse)
async def segment_endpoint(params: SegmentParams):
    logger.info(
        "POST /pipeline/segment | image_id={} n_layers={} depth_mode={}",
        params.image_id, params.n_layers, params.depth_mode,
    )
    try:
        # Offload CPU-bound SAM inference to a thread so the event loop
        # stays responsive for concurrent uploads and health checks.
        return await asyncio.to_thread(process_segment, params)
    except HTTPException:
        # Business-logic errors (400/501/500 with structured detail)
        # already carry step-location info — pass through unchanged.
        raise
    except Exception as e:
        logger.exception(
            "Unhandled error in SAM segment endpoint | image_id={}",
            params.image_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"SAM分割失败 | 阶段: 路由层(未预期异常) | "
                f"原因: {type(e).__name__}: {e}"
            ),
        )
