"""SVG generation endpoints — per API_CONTRACT.md § Step 7."""

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from app.models.requests import SvgParams
from app.models.responses import SvgResponse, MultiLayerSvgResponse
from app.services.svg_service import process_svg, process_svg_multi

router = APIRouter(tags=["svg"])


class MultiLayerSvgRequest(BaseModel):
    image_id: str = Field(..., description="UUID of the uploaded original image")
    n_layers: int = Field(..., ge=2, le=5, description="Number of layers to generate SVGs for")
    simplify_tolerance: float = Field(1.0, ge=0.1, le=10.0, description="Douglas-Peucker epsilon in pixels")


@router.post("/pipeline/svg", response_model=SvgResponse)
def svg_endpoint(params: SvgParams):
    logger.info(
        "POST /pipeline/svg | image_id={} simplify_tolerance={} layer_index={}",
        params.image_id, params.simplify_tolerance, params.layer_index,
    )
    return process_svg(params)


@router.post("/pipeline/svg/multi-layer", response_model=MultiLayerSvgResponse)
def svg_multi_endpoint(params: MultiLayerSvgRequest):
    """Generate per-layer SVGs and merge into a single tiled multi-layer SVG."""
    logger.info(
        "POST /pipeline/svg/multi-layer | image_id={} n_layers={} simplify_tolerance={}",
        params.image_id, params.n_layers, params.simplify_tolerance,
    )
    return process_svg_multi(
        image_id=params.image_id,
        n_layers=params.n_layers,
        simplify_tolerance=params.simplify_tolerance,
    )
