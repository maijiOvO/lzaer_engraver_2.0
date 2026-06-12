"""MobileSAM automatic mask generation engine.

Loads MobileSAM from /app/models/mobile_sam.pt (persistent Docker volume),
runs AutomaticMaskGenerator with paper_sculpture preset, returns fragment
masks and a region_map label image.

Pure functions — no HTTP, no file I/O except for model loading.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any

import cv2
import numpy as np
from loguru import logger

# ── Constants ───────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MODEL_PATH = os.path.join(MODEL_DIR, "mobile_sam.pt")
MODEL_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

# paper_sculpture preset — highest quality, suitable for engraving masks
PAPER_SCULPTURE_PRESET: dict[str, Any] = {
    "points_per_side": 64,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "crop_n_layers": 1,
    "min_mask_region_area": 50,
    "crop_n_points_downscale_factor": 2,
    "max_sam_dim": 1200,
}

# ── Global model singleton ──────────────────────────────────────────
_sam_model: Any = None
_device: str | None = None


def _resolve_device() -> str:
    """Resolve the best available torch device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _download_model() -> str:
    """Download MobileSAM weights if not already present.

    Returns the path to the model file.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        logger.info("[SAM] 使用缓存模型: {}", MODEL_PATH)
        return MODEL_PATH

    logger.info("[SAM] 下载 MobileSAM 权重: {}", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("[SAM] 下载完成 → {}", MODEL_PATH)
    return MODEL_PATH


def _get_sam_model():
    """Lazy-load MobileSAM singleton.

    Downloads weights to /app/models/ on first call (persistent
    Docker volume — survives container restarts).
    """
    global _sam_model, _device

    if _sam_model is None:
        _download_model()

        import torch
        from mobile_sam import sam_model_registry

        # Use all available CPU cores for intra-op parallelism.
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)

        _device = _resolve_device()
        logger.info("[SAM] 加载 MobileSAM | device={} threads={}", _device, cpu_count)
        _sam_model = sam_model_registry["vit_t"](checkpoint=MODEL_PATH)
        _sam_model.to(device=_device)
        _sam_model.eval()

        # Optimise the image encoder for CPU inference.
        # Strategy: try torch.jit.trace first (no C++ compiler needed),
        # fall back to torch.compile if available and g++ is present.
        if _device == "cpu":
            _original_encoder = _sam_model.image_encoder
            try:
                # torch.jit.trace — lightweight, no compiler dependency.
                dummy = torch.randn(1, 3, 1024, 1024)
                with torch.inference_mode():
                    _sam_model.image_encoder = torch.jit.trace(
                        _original_encoder, dummy
                    )
                # The traced module loses non-forward attributes;
                # copy img_size back — MobileSAM's mask generator needs it.
                if hasattr(_original_encoder, "img_size"):
                    _sam_model.image_encoder.img_size = _original_encoder.img_size
                logger.info("[SAM] 图像编码器已追踪 (torch.jit)")
            except Exception as exc:
                _sam_model.image_encoder = _original_encoder
                logger.warning("[SAM] jit.trace 失败 ({})", exc)
                # Try torch.compile as fallback (requires g++ in Docker).
                if hasattr(torch, "compile"):
                    try:
                        _sam_model.image_encoder = torch.compile(
                            _original_encoder, mode="reduce-overhead",
                        )
                        with torch.inference_mode():
                            _sam_model.image_encoder(dummy)
                        logger.info("[SAM] 图像编码器已编译 (torch.compile)")
                    except Exception as exc2:
                        _sam_model.image_encoder = _original_encoder
                        logger.warning("[SAM] torch.compile 也失败 ({})，退回 eager 模式", exc2)
                else:
                    logger.info("[SAM] torch.compile 不可用，使用 eager 模式")

    return _sam_model


def _create_generator(**overrides: Any):
    """Create a SamAutomaticMaskGenerator with paper_sculpture defaults."""
    from mobile_sam import SamAutomaticMaskGenerator

    params = dict(PAPER_SCULPTURE_PRESET)
    params.update(overrides)

    model = _get_sam_model()
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=params.get("points_per_side", 64),
        pred_iou_thresh=params.get("pred_iou_thresh", 0.88),
        stability_score_thresh=params.get("stability_score_thresh", 0.95),
        stability_score_offset=1.0,
        box_nms_thresh=0.7,
        crop_n_layers=params.get("crop_n_layers", 1),
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=params.get(
            "crop_n_points_downscale_factor", 2
        ),
        min_mask_region_area=params.get("min_mask_region_area", 50),
        output_mode="binary_mask",
    )


def _preprocess_image(image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Resize image so its largest dimension ≤ max_dim, keeping aspect ratio.

    Returns the (possibly resized) RGB image for SAM ingestion.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _build_region_map(masks: list[dict], h: int, w: int) -> np.ndarray:
    """Build a fragment-id label map from SAM output masks.

    Masks are stacked with priority: earlier masks override later masks
    (SAM generates confident masks first).

    Returns:
        region_map: (H, W) int32 label map. 0 = background, 1..N = fragments.
    """
    region_map = np.zeros((h, w), dtype=np.int32)

    for i, mask_data in enumerate(masks):
        seg = mask_data["segmentation"]
        # Resize mask to target dimensions if needed
        if seg.shape[:2] != (h, w):
            seg_uint8 = seg.astype(np.uint8) * 255
            seg_uint8 = cv2.resize(
                seg_uint8, (w, h), interpolation=cv2.INTER_NEAREST
            )
            seg = seg_uint8 > 127

        region_map[seg] = i + 1  # fragment IDs start at 1

    return region_map


def _upscale_mask_smooth(
    mask: np.ndarray,
    target_h: int,
    target_w: int,
    blur_sigma: float = 0.8,
) -> np.ndarray:
    """Upscale a binary mask with anti-aliased edges.

    Bicubic interpolation produces soft probability edges, Gaussian
    blur smooths remaining micro-stairsteps, and thresholding restores
    a clean binary mask with sub-pixel-accurate boundaries.
    """
    mask_f32 = mask.astype(np.float32)
    upscaled = cv2.resize(
        mask_f32, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=blur_sigma)
    return blurred > 0.5


def _build_region_map_from_masks(
    masks: list[np.ndarray],
    h: int,
    w: int,
) -> np.ndarray:
    """Rebuild region_map from already-upscaled bool masks.

    Earlier masks (lower index = higher confidence) take priority
    where masks overlap — matching SAM's native ordering.
    """
    region_map = np.zeros((h, w), dtype=np.int32)
    for i, seg in enumerate(masks):
        region_map[seg] = i + 1
    return region_map


def _snap_mask_to_edges(
    mask: np.ndarray,         # bool (H, W) — already upscaled to original res
    original_bgr: np.ndarray,  # BGR uint8 (H, W) — original image
    edge_band: int = 3,
) -> np.ndarray:
    """Snap mask boundaries to the original image's natural edges.

    After smooth upscaling, mask edges are anti-aliased but blind to
    image content.  This step uses the original image gradient to
    nudge boundaries toward real edges, improving edge precision
    especially for draft-mode (low SAM resolution) masks.
    """
    if mask.sum() == 0 or mask.sum() == mask.size:
        return mask  # trivial mask — nothing to refine

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Edge strength via Sobel gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.sqrt(gx ** 2 + gy ** 2)
    edge_strength /= edge_strength.max() + 1e-8  # normalise [0, 1]

    # Boundary band: dilate XOR erode
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((edge_band, edge_band), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel)
    eroded = cv2.erode(mask_u8, kernel)
    boundary = dilated != eroded

    # In the boundary band, bias the mask toward edge alignment:
    #   strong edge → sharpen (push toward 0 or 1)
    #   weak edge   → smooth  (push toward 0.5, letting threshold handle it)
    mask_f32 = mask.astype(np.float32)
    edge_bias = (edge_strength - 0.5) * 0.4
    refined = mask_f32.copy()
    refined[boundary] = np.clip(
        mask_f32[boundary] + edge_bias[boundary], 0.0, 1.0,
    )

    return refined > 0.5


def run_sam_automatic(
    image: np.ndarray,
    max_dim: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Run MobileSAM AutomaticMaskGenerator on an image.

    Args:
        image: BGR numpy array (H, W, 3) uint8 — as loaded by cv2.imread.
        max_dim: maximum dimension for SAM processing. Default 1200
            (paper_sculpture preset). Set lower for faster processing.

    Returns:
        (masks, region_map) where:
        - masks: list of SAM mask dicts, each with keys:
            segmentation (bool H×W), area, bbox, stability_score, ...
        - region_map: (H, W) int32 label map. 0=background, 1..N=fragments.
    """
    from loguru import logger

    _max_dim = max_dim or PAPER_SCULPTURE_PRESET.get("max_sam_dim", 1200)

    # ── Step A: Preprocess (resize) ─────────────────────────────
    logger.debug("[SAM引擎] 步骤A: 图像预处理 (max_dim={})", _max_dim)
    try:
        sam_image = _preprocess_image(image, max_dim=_max_dim)
    except Exception as e:
        raise RuntimeError(
            f"[SAM引擎] 图像预处理失败 — 原因: {type(e).__name__}: {e}"
        ) from e

    # ── Step B: BGR → RGB conversion ────────────────────────────
    logger.debug("[SAM引擎] 步骤B: 色彩空间转换 BGR→RGB")
    try:
        if sam_image.shape[-1] == 3:
            sam_rgb = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        else:
            sam_rgb = sam_image
    except Exception as e:
        raise RuntimeError(
            f"[SAM引擎] 色彩空间转换失败 — 输入shape={sam_image.shape} — "
            f"原因: {type(e).__name__}: {e}"
        ) from e

    # ── Step C: Create generator / load model ───────────────────
    logger.debug("[SAM引擎] 步骤C: 创建SAM生成器")
    import torch

    try:
        generator = _create_generator()
    except Exception as e:
        raise RuntimeError(
            f"[SAM引擎] 模型加载失败 — 原因: {type(e).__name__}: {e}"
        ) from e

    # ── Step D: SAM inference ───────────────────────────────────
    logger.debug("[SAM引擎] 步骤D: MobileSAM推理 (image={}x{})",
                 sam_rgb.shape[1], sam_rgb.shape[0])
    try:
        with torch.inference_mode():
            masks = generator.generate(sam_rgb)
    except Exception as e:
        raise RuntimeError(
            f"[SAM引擎] MobileSAM推理失败 — 原因: {type(e).__name__}: {e}"
        ) from e

    if not masks:
        logger.warning("[SAM引擎] SAM推理返回0个mask — 图像可能过于均匀")
        return [], np.zeros(image.shape[:2], dtype=np.int32)

    logger.debug("[SAM引擎] 步骤D完成: 生成{}个mask", len(masks))

    # ── Step E: Build region map at SAM resolution ──────────────
    sam_h, sam_w = sam_image.shape[:2]
    logger.debug("[SAM引擎] 步骤E: 构建区域图 ({}×{})", sam_w, sam_h)
    try:
        region_map_sam = _build_region_map(masks, sam_h, sam_w)
    except Exception as e:
        raise RuntimeError(
            f"[SAM引擎] 构建区域图失败 — 原因: {type(e).__name__}: {e}"
        ) from e

    orig_h, orig_w = image.shape[:2]

    if (sam_h, sam_w) == (orig_h, orig_w):
        # No upscaling needed — image was already ≤ max_dim
        logger.debug("[SAM引擎] 无需上采样 (SAM分辨率=原始分辨率)")
        return masks, region_map_sam

    # ── Step F: Per-mask upscale + edge snap ───────────────────
    logger.debug("[SAM引擎] 步骤F: 逐mask上采样+边缘精修 ({}→{}×{})",
                 (sam_w, sam_h), orig_w, orig_h)
    orig_segs: list[np.ndarray] = []
    total = len(masks)
    for i, mask_data in enumerate(masks):
        # F1: Upscale
        try:
            seg_smooth = _upscale_mask_smooth(
                mask_data["segmentation"], orig_h, orig_w,
            )
        except Exception as e:
            raise RuntimeError(
                f"[SAM引擎] 第{i+1}/{total}个mask上采样失败 — "
                f"原因: {type(e).__name__}: {e}"
            ) from e

        # F2: Edge snap
        try:
            seg_smooth = _snap_mask_to_edges(seg_smooth, image)
        except Exception as e:
            raise RuntimeError(
                f"[SAM引擎] 第{i+1}/{total}个mask边缘精修失败 — "
                f"原因: {type(e).__name__}: {e}"
            ) from e

        mask_data["segmentation"] = seg_smooth
        mask_data["area"] = int(np.count_nonzero(seg_smooth))
        orig_segs.append(seg_smooth)

    # ── Step G: Rebuild region map at original resolution ───────
    logger.debug("[SAM引擎] 步骤G: 重建区域图 (原始分辨率 {}×{})", orig_w, orig_h)
    try:
        region_map = _build_region_map_from_masks(orig_segs, orig_h, orig_w)
    except Exception as e:
        raise RuntimeError(
            f"[SAM引擎] 重建区域图失败 — 原因: {type(e).__name__}: {e}"
        ) from e

    logger.debug("[SAM引擎] 完成: {}个mask, region_map shape={}",
                 len(masks), region_map.shape)
    return masks, region_map


def refine_mask(
    image: np.ndarray,
    rough_mask: np.ndarray,
    edge_band: int = 3,
) -> np.ndarray:
    """用 SAM 沿原图边缘精修深度蒙版。

    以深度估计输出的粗蒙版为 mask prompt，SAM 预测器沿真实图像
    边缘收紧边界，解决深度图边界模糊的问题。

    Args:
        image: BGR 原始图像 (H, W, 3) uint8。
        rough_mask: (H, W) 二值蒙版（bool 或 0/255），来自深度分层。
        edge_band: 传给 _snap_mask_to_edges 的边缘带宽。

    Returns:
        (H, W) bool 精修后的二值蒙版。
    """
    from loguru import logger
    import torch

    orig_h, orig_w = image.shape[:2]
    rough_bin = rough_mask.astype(bool) if rough_mask.dtype != bool else rough_mask

    if not rough_bin.any() or rough_bin.all():
        logger.debug("[SAM精修] 蒙版为空或全满，跳过精修")
        return rough_bin

    logger.debug(
        "[SAM精修] 输入 | image={}x{} mask_fg={:.1f}%",
        orig_w, orig_h,
        100 * rough_bin.sum() / rough_bin.size,
    )

    # ── 预处理：缩放图像到 1024×1024（SAM 原生分辨率）───────
    sam_longest = 1024
    scale = sam_longest / max(orig_h, orig_w)
    sam_h = int(orig_h * scale)
    sam_w = int(orig_w * scale)

    sam_image = cv2.resize(image, (sam_w, sam_h), interpolation=cv2.INTER_AREA)
    sam_mask = cv2.resize(
        rough_bin.astype(np.uint8) * 255,
        (sam_w, sam_h),
        interpolation=cv2.INTER_NEAREST,
    )
    sam_mask_bin = sam_mask > 127

    # ── RGB 转换 ─────────────────────────────────────────────
    sam_rgb = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)

    # ── 创建 SAM 预测器 ─────────────────────────────────────
    model = _get_sam_model()

    from mobile_sam import SamPredictor
    predictor = SamPredictor(model)
    predictor.set_image(sam_rgb)

    # ── 将蒙版转为 SAM mask_input logits ──────────────────
    # SAM 需要 (1, 256, 256) 的 logits 张量
    mask_tensor = torch.as_tensor(sam_mask_bin, dtype=torch.float32, device=_device)
    # 缩放到 256×256
    mask_256 = torch.nn.functional.interpolate(
        mask_tensor[None, None, :, :].float(),
        size=(256, 256),
        mode="bilinear",
    )
    # 转为 logits: 正例 logits=+4, 负例 logits=-4
    mask_logits = (mask_256 - 0.5) * 8.0

    # ── SAM 推理 ───────────────────────────────────────────
    with torch.inference_mode():
        masks, scores, _logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            mask_input=mask_logits,
            multimask_output=False,
        )

    # masks[0] 是 (1, sam_H, sam_W) bool tensor
    refined_sam = masks[0].cpu().numpy()
    score = float(scores[0])
    logger.debug("[SAM精修]   置信度={:.3f}", score)

    # ── 上采样回原始分辨率 ─────────────────────────────────
    if (sam_h, sam_w) != (orig_h, orig_w):
        refined = _upscale_mask_smooth(refined_sam, orig_h, orig_w)
    else:
        refined = refined_sam

    # ── 边缘精修（Sobel 梯度吸附） ─────────────────────────
    refined = _snap_mask_to_edges(refined, image, edge_band=edge_band)

    logger.debug(
        "[SAM精修] 完成 | refined_fg={:.1f}%",
        100 * refined.sum() / refined.size,
    )
    return refined


def get_sam_info() -> dict:
    """Return SAM runtime info (device, model status)."""
    return {
        "device": _device or _resolve_device(),
        "model_loaded": _sam_model is not None,
        "model_path": MODEL_PATH,
        "preset": "paper_sculpture",
    }
