/** Types matching API_CONTRACT.md response models. */

/** GET /api/health */
export interface HealthResponse {
  status: string;
  version: string;
}

/** POST /api/upload — Step 1 */
export interface UploadResponse {
  image_id: string;
  width: number;
  height: number;
  original_url: string;
}

/** POST /api/pipeline/segment — Steps 2-3 (Depth-guided structural layering) */
export interface SegmentParams {
  image_id: string;
  n_layers?: number;          // [DEPRECATED] 由后端深度模型自动推断，不再由前端发送
  sam_quality: string;        // "draft" | "standard" | "fine", default "standard"
  force_recompute?: boolean;   // skip depth cache, default false
  // ── 结构分层参数 (2026-06-23 新增) ──
  frame_width: number;        // 外框宽度(px), 20-200, default 50
  min_island_area: number;    // 孤立岛丢弃阈值(px), 10-5000, default 100
  // ── 已废弃（向后兼容） ──
  depth_mode?: string;         // [DEPRECATED]
  merge_sensitivity?: number;  // [DEPRECATED]
  min_layer_area_pct?: number; // [DEPRECATED]
}

export interface LayerInfo {
  layer_index: number;
  mask_url: string;
  frame_url: string;  // mask with outer clamping border
}

export interface SegmentResponse {
  overlay_url: string;
  layers: LayerInfo[];
}

/** POST /api/pipeline/canny — Step 4 */
export interface CannyParams {
  image_id: string;
  layer_index: number | null;
  low: number;
  high: number;
  smooth_level: number;
}

export interface PipelineStepResponse {
  result_url: string;
  processing_time_ms: number;
}

/** POST /api/pipeline/denoise — Step 5 */
export interface DenoiseParams {
  image_id: string;
  layer_index: number | null;  // null = whole image, 0..N-1 = single layer
  min_component_area: number;  // default 4, range 1-100
}

/** POST /api/pipeline/connectivity — Step 6 */
export interface ConnectivityParams {
  image_id: string;
  layer_index: number | null;  // null = whole image
  gap_tolerance: number;
}

export interface ConnectivityResponse {
  result_url: string;
  bridges_built: number;
  processing_time_ms: number;
}

/** POST /api/pipeline/svg — Step 7 */
export interface SvgParams {
  image_id: string;
  layer_index: number | null;  // null = whole image
  simplify_tolerance: number;
}

export interface SvgResponse {
  svg_url: string;
  total_paths: number;
  total_points: number;
  processing_time_ms: number;
}

/** POST /api/pipeline/svg/multi-layer — Step 7 multi-layer */
export interface MultiLayerSvgParams {
  image_id: string;
  n_layers: number;
  simplify_tolerance: number;
}

export interface MultiLayerSvgResponse {
  svg_url: string;
  per_layer_paths: number[];
  total_paths: number;
  total_points: number;
  processing_time_ms: number;
}

/* ── Frontend-only types (not from API contract) ────────────── */

/** Organised per-layer data for multi-layer canvas rendering. */
export interface MultiLayerInfo {
  layerIndex: number;
  maskUrl: string;          // SAM mask
  frameUrl: string;         // mask with outer frame border
  cannyUrl: string | null;
  denoiseUrl: string | null;
  connectivityUrl: string | null;
  label: string;            // "图层 1", "图层 2", ...
}

export type MultiViewMode = 'vertical' | 'horizontal' | 'overlay';
