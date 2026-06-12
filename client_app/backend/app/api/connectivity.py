"""Connectivity repair endpoint — per API_CONTRACT.md § Step 6."""

from fastapi import APIRouter
from loguru import logger

from app.models.requests import ConnectivityParams
from app.models.responses import ConnectivityResponse
from app.services.connectivity_service import process_connectivity

router = APIRouter(tags=["connectivity"])


@router.post("/pipeline/connectivity", response_model=ConnectivityResponse)
def connectivity_endpoint(params: ConnectivityParams):
    logger.info(
        "POST /pipeline/connectivity | image_id={} gap_tolerance={}",
        params.image_id, params.gap_tolerance,
    )
    return process_connectivity(params)
