"""Response models for the Laser Engraver pipeline."""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Per API_CONTRACT.md § Step 1: image upload response."""
    image_id: str = Field(..., description="UUID of the uploaded image")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    original_url: str = Field(..., description="Accessible URL, e.g. /outputs/uuid_original.jpg")


class PipelineStepResponse(BaseModel):
    """Standard response for intermediate pipeline steps (canny, denoise, etc.)."""
    result_url: str = Field(..., description="URL path to the result image, e.g. /outputs/uuid_canny.png")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class ConnectivityResponse(BaseModel):
    """Per API_CONTRACT.md § Step 6: connectivity repair response."""
    result_url: str = Field(..., description="URL path to the connectivity-repaired image")
    bridges_built: int = Field(..., description="Number of pixel bridges drawn")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class SvgResponse(BaseModel):
    """Per API_CONTRACT.md § Step 7: SVG generation response."""
    svg_url: str = Field(..., description="URL path to the generated SVG file")
    total_paths: int = Field(..., description="Number of contour paths in the SVG")
    total_points: int = Field(..., description="Total anchor points across all paths")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class LayerInfo(BaseModel):
    """Per API_CONTRACT.md § Step 2-3: single layer in SegmentResponse."""
    layer_index: int = Field(..., description="Zero-based layer index")
    mask_url: str = Field(..., description="URL path to the binary mask image")
    frame_url: str = Field(..., description="URL path to the mask image with outer frame border")


class MultiLayerSvgResponse(BaseModel):
    """Response for multi-layer SVG generation — all layers merged into one SVG."""
    svg_url: str = Field(..., description="URL path to the merged multi-layer SVG file")
    per_layer_paths: list[int] = Field(..., description="Path count per layer")
    total_paths: int = Field(..., description="Total contour paths across all layers")
    total_points: int = Field(..., description="Total anchor points across all paths")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class SegmentResponse(BaseModel):
    """Per API_CONTRACT.md § Step 2-3: SAM segmentation response."""
    overlay_url: str = Field(..., description="URL path to the colored overlay image")
    layers: list[LayerInfo] = Field(..., description="Ordered list of layer masks (front→back)")
