"""Canny edge-detection API route (Step 4 of the pipeline).

Per API_CONTRACT.md § Step 4:
  POST /pipeline/canny
"""

from fastapi import APIRouter
from loguru import logger

from app.models.requests import CannyParams
from app.models.responses import PipelineStepResponse
from app.services.canny_service import process_canny

router = APIRouter(tags=["pipeline"])


@router.post("/pipeline/canny", response_model=PipelineStepResponse)
async def canny_endpoint(params: CannyParams):
    logger.info("POST /pipeline/canny | image_id={}", params.image_id)
    return process_canny(params)
