"""Request models for the Laser Engraver pipeline."""

from typing import Optional
from pydantic import BaseModel, Field


class LineArtParams(BaseModel):
    """Per API_CONTRACT.md § Step 4: Canny LineArt extraction parameters.

    Uses CLAHE + Canny edge detection (pure CPU, no GPU/model download).
    """
    image_id: str = Field(..., description="UUID of the uploaded original image")
    layer_index: Optional[int] = Field(None, description="Layer index (null for single-layer mode)")
    low: int = Field(50, ge=0, le=255, description="Canny low threshold — smaller = more edges")
    high: int = Field(150, ge=0, le=255, description="Canny high threshold — recommended high ≈ low × 3")
    smooth_level: int = Field(0, ge=0, le=2, description="Pre-smoothing: 0=default, 1=light, 2=medium")
