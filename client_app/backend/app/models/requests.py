"""Request models for the Laser Engraver pipeline."""

from typing import Optional
from pydantic import BaseModel, Field


class CannyParams(BaseModel):
    """Per API_CONTRACT.md § Step 4: Canny edge-detection parameters.

    Uses CLAHE + Canny edge detection (pure CPU, no GPU/model download).
    """
    image_id: str = Field(..., description="UUID of the uploaded original image")
    layer_index: Optional[int] = Field(None, description="Layer index (null for single-layer mode)")
    low: int = Field(50, ge=0, le=255, description="Canny low threshold — smaller = more edges")
    high: int = Field(150, ge=0, le=255, description="Canny high threshold — recommended high ≈ low × 3")
    smooth_level: int = Field(0, ge=0, le=2, description="Pre-smoothing: 0=default, 1=light, 2=medium")


class DenoiseParams(BaseModel):
    """Per API_CONTRACT.md § Step 5: noise reduction parameters."""
    image_id: str = Field(..., description="UUID of the uploaded original image")
    layer_index: Optional[int] = Field(None, description="Layer index (null for single-layer mode)")
    min_component_area: int = Field(4, ge=1, le=100, description="Remove connected components smaller than this pixel area")


class SegmentParams(BaseModel):
    """Per API_CONTRACT.md § Step 2-3: 深度引导结构分层。

    Depth-Anything-V2 估计单目深度 → 等距量化为 N 层 →
    连通性校验 + 桥接修复 → (可选) SAM 逐层边界精修。
    """
    image_id: str = Field(..., description="UUID of the uploaded original image")
    n_layers: int = Field(3, ge=2, le=5, description="Number of depth layers to produce")
    sam_quality: str = Field("standard", description="SAM quality: 'draft' (skip refine), 'standard' (refine), 'fine' (refine + edge snap)")
    force_recompute: bool = Field(False, description="If true, skip depth cache and re-run inference")

    # ── 结构分层参数 ──────────────────────────────────────────
    frame_width: int = Field(50, ge=20, le=200, description="外层固定边框宽度（像素）")
    min_island_area: int = Field(100, ge=10, le=5000, description="低于此面积(px)的孤立岛直接丢弃")

    # ── 向后兼容（已废弃，保留以兼容旧前端请求）─────────────
    depth_mode: Optional[str] = Field(None, description="[DEPRECATED] 由深度估计替代，不再使用")
    merge_sensitivity: Optional[float] = Field(None, description="[DEPRECATED] 不再使用")
    min_layer_area_pct: Optional[float] = Field(None, description="[DEPRECATED] 不再使用")


class ConnectivityParams(BaseModel):
    """Per API_CONTRACT.md § Step 6: connectivity repair parameters."""
    image_id: str = Field(..., description="UUID of the uploaded original image")
    layer_index: Optional[int] = Field(None, description="Layer index (null for single-layer mode)")
    gap_tolerance: int = Field(5, ge=1, le=20, description="Max pixel gap to bridge between fragments")


class SvgParams(BaseModel):
    """Per API_CONTRACT.md § Step 7: SVG generation parameters."""
    image_id: str = Field(..., description="UUID of the uploaded original image")
    layer_index: Optional[int] = Field(None, description="Layer index (null for single-layer mode)")
    simplify_tolerance: float = Field(1.0, ge=0.1, le=10.0, description="Douglas-Peucker epsilon in pixels")
