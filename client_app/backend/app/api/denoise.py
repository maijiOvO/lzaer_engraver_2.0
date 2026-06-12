"""Denoise endpoint — per API_CONTRACT.md § Step 5."""

from fastapi import APIRouter
from loguru import logger

from app.models.requests import DenoiseParams
from app.models.responses import PipelineStepResponse
from app.services.denoise_service import process_denoise

router = APIRouter(tags=["denoise"])


@router.post("/pipeline/denoise", response_model=PipelineStepResponse)
def denoise_endpoint(params: DenoiseParams):
    logger.info(
        "POST /pipeline/denoise | image_id={} min_component_area={}",
        params.image_id, params.min_component_area,
    )
    return process_denoise(params)
