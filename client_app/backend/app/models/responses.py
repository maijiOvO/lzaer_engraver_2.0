"""Response models for the Laser Engraver pipeline."""

from pydantic import BaseModel, Field


class PipelineStepResponse(BaseModel):
    """Standard response for intermediate pipeline steps (lineart, denoise, etc.)."""
    result_url: str = Field(..., description="URL path to the result image, e.g. /outputs/uuid_lineart.png")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
