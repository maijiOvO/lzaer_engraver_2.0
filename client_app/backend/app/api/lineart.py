"""Line-art extraction API route (Step 4 of the pipeline).

Per API_CONTRACT.md § Step 4:
  POST /pipeline/lineart
"""

from fastapi import APIRouter
from loguru import logger

from app.models.requests import LineArtParams
from app.models.responses import PipelineStepResponse
from app.services.lineart_service import process_lineart

router = APIRouter(tags=["pipeline"])


@router.post("/pipeline/lineart", response_model=PipelineStepResponse)
async def lineart_endpoint(params: LineArtParams):
    logger.info("POST /pipeline/lineart | image_id={}", params.image_id)
    return process_lineart(params)
