"""Request models for the Laser Engraver pipeline."""

from typing import Optional
from pydantic import BaseModel, Field


class LineArtParams(BaseModel):
    """Per API_CONTRACT.md § Step 4: lineart extraction parameters."""
    image_id: str = Field(..., description="UUID of the uploaded original image")
    layer_index: Optional[int] = Field(None, description="Layer index (null for single-layer mode)")
    detect_resolution: int = Field(768, ge=128, le=2048, description="Detection resolution for the lineart model")
    line_strength: int = Field(55, ge=0, le=255, description="Line strength / sensitivity")
    thin: bool = Field(True, description="Whether to apply thinning to lines")
