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
import torch
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
    "crop_n_layers": 0,  # 关闭裁剪推理，消除网格裁切伪影
    "min_mask_region_area": 50,
    "crop_n_points_downscale_factor": 2,
    "max_sam_dim": 1200,
}

# ── Global model singleton ──────────────────────────────────────────
_sam_model: Any = None
_device: str | None = None
_onnx_encoder_enabled: bool | None = None  # None = not yet probed


def _resolve_device() -> str:
    """Resolve the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _probe_onnx_dml_encoder() -> bool:
    """Detect whether ONNX Runtime + DirectML encoder is available.

    Caches the result so probing only happens once per process.
    """
    global _onnx_encoder_enabled
    if _onnx_encoder_enabled is not None:
        return _onnx_encoder_enabled

    try:
        from app.utils.onnx_engine import is_dml_available, get_sam_encoder_session

        if not is_dml_available():
            logger.info("[SAM] DirectML 不可用，使用 PyTorch CPU 编码器")
            _onnx_encoder_enabled = False
            return False

        # Trigger session creation early to catch init errors
        get_sam_encoder_session()
        logger.info("[SAM] ONNX Runtime + DirectML 可用，将启用 GPU 加速编码器")
        _onnx_encoder_enabled = True
        return True
    except Exception as exc:
        logger.warning("[SAM] ONNX 编码器探测失败 ({}), 回退到 PyTorch CPU", exc)
        _onnx_encoder_enabled = False
        return False


def _wrap_encoder_for_onnx(original_encoder):
    """Wrap the PyTorch image_encoder so calls go through ONNX Runtime DML.

    Preserves ``img_size`` attribute required by MobileSAM's mask generator.
    """
    import numpy as np

    from app.utils.onnx_engine import run_sam_encoder

    class _ONNXEncoderWrapper:
        """Thin wrapper that delegates image_encoder calls to ONNX Runtime."""

        def __init__(self, orig):
            self.img_size = getattr(orig, "img_size", 1024)

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            # x is already preprocessed by model.preprocess() —
            # shape (1, 3, 1024, 1024), float32, on CPU.
            x_np = x.cpu().numpy().astype(np.float32)
            result = run_sam_encoder(x_np)  # → (1, 256, 64, 64)
            return torch.from_numpy(result)

    return _ONNXEncoderWrapper(original_encoder)


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

        # ── Encoder optimization ─────────────────────────────────
        if _device == "cpu" and _probe_onnx_dml_encoder():
            # Priority 1: ONNX Runtime + DirectML GPU accelerator
            # Use object.__setattr__ to bypass nn.Module.__setattr__ type check
            _original_encoder = _sam_model.image_encoder
            object.__setattr__(
                _sam_model, "image_encoder",
                _wrap_encoder_for_onnx(_original_encoder),
            )
            logger.info("[SAM] 图像编码器 → ONNX Runtime + DirectML (GPU)")
        elif _device == "cpu":
            # Priority 2: PyTorch JIT / compile (pure CPU)
            _original_encoder = _sam_model.image_encoder
            try:
                dummy = torch.randn(1, 3, 1024, 1024)
                with torch.inference_mode():
                    _sam_model.image_encoder = torch.jit.trace(
                        _original_encoder, dummy
                    )
                if hasattr(_original_encoder, "img_size"):
                    _sam_model.image_encoder.img_size = _original_encoder.img_size
                logger.info("[SAM] 图像编码器已追踪 (torch.jit)")
            except Exception as exc:
                _sam_model.image_encoder = _original_encoder
                logger.warning("[SAM] jit.trace 失败 ({})", exc)
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
    mask: np.ndarray,
    original_bgr: np.ndarray,
    edge_band: int = 5,
) -> np.ndarray:
    """
    使用导向滤波 (Guided Filter) 实现像素级的物理边缘吸附。
    利用高分原图作为引导，将掩码边缘完美对齐到复杂的真实结构（如铁塔钢架、天线）上，
    彻底消除 GrabCut 带来的块状伪影。
    """
    if mask.sum() == 0 or mask.sum() == mask.size:
        return mask

    try:
        # 0. 尺寸对齐：mask 可能被 frame padding 放大，guide 必须匹配其尺寸
        mask_h, mask_w = mask.shape[:2]
        guide_h, guide_w = original_bgr.shape[:2]

        if (mask_h, mask_w) != (guide_h, guide_w):
            # mask 大于原图 → 白边 padding（frame 区域）
            if mask_h >= guide_h and mask_w >= guide_w:
                pad_top = (mask_h - guide_h) // 2
                pad_bottom = mask_h - guide_h - pad_top
                pad_left = (mask_w - guide_w) // 2
                pad_right = mask_w - guide_w - pad_left
                guide_bgr = cv2.copyMakeBorder(
                    original_bgr,
                    pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255),
                )
            else:
                # mask 小于原图 → 裁剪 guide 中心区域
                crop_top = (guide_h - mask_h) // 2
                crop_left = (guide_w - mask_w) // 2
                guide_bgr = original_bgr[
                    crop_top:crop_top + mask_h,
                    crop_left:crop_left + mask_w,
                ]
        else:
            guide_bgr = original_bgr

        # 1. 预处理：向导图 (Guide) 必须是 float32 格式，归一化到 [0, 1]
        guide = guide_bgr.astype(np.float32) / 255.0

        # 2. 预处理：将二值蒙版转为软蒙版 (Soft Mask)
        # 给予边界一定的灰度渐变，让导向滤波有将其"推/拉"到真实物理边缘的空间
        mask_f32 = mask.astype(np.float32)
        blur_size = max(3, edge_band * 2 + 1)
        soft_mask = cv2.GaussianBlur(mask_f32, (blur_size, blur_size), 0)

        # 3. 创建并执行导向滤波
        # radius: 搜索邻域，eps: 正则化参数（越小对微弱边缘越敏感）
        # 这里使用 1e-5，以确保像铁塔钢架这样强对比的高频边缘被死死咬住
        gf = cv2.ximgproc.createGuidedFilter(
            guide=guide,
            radius=max(3, edge_band),
            eps=1e-5
        )
        refined_f32 = gf.filter(soft_mask)

        # 4. 二值化输出（在 0.5 处切断，获得极其凌厉的真实边界）
        return refined_f32 > 0.5

    except AttributeError:
        from loguru import logger
        logger.error("[边缘吸附] 缺少 opencv-contrib-python 库，无法使用导向滤波，回退到原始蒙版！")
        return mask
    except Exception as exc:
        from loguru import logger
        logger.warning(f"[边缘吸附] 导向滤波失败 — fallback 原始蒙版 | {exc}")
        return mask


def run_sam_automatic(
    image: np.ndarray,
    max_dim: int | None = None,
    cache_path: str | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Run MobileSAM AutomaticMaskGenerator on an image.

    Args:
        image: BGR numpy array (H, W, 3) uint8 — as loaded by cv2.imread.
        max_dim: maximum dimension for SAM processing. Default 1200
            (paper_sculpture preset). Set lower for faster processing.
        cache_path: if provided, save/load compressed region_map cache
            (.npz) to avoid re-running SAM on the same image.

    Returns:
        (masks, region_map) where:
        - masks: list of SAM mask dicts, each with keys:
            segmentation (bool H×W), area, bbox, stability_score, ...
        - region_map: (H, W) int32 label map. 0=background, 1..N=fragments.
    """
    from loguru import logger

    # ── Cache hit: load from disk ────────────────────────────
    if cache_path and os.path.exists(cache_path):
        logger.info("[SAM引擎] 缓存命中 | path={}", cache_path)
        try:
            data = np.load(cache_path, allow_pickle=True)
            region_map = data["region_map"]
            mask_meta = data["mask_meta"]
            masks = _reconstruct_masks(region_map, mask_meta)
            logger.info("[SAM引擎] 从缓存重建 {} 个mask", len(masks))
            return masks, region_map
        except Exception as e:
            logger.warning("[SAM引擎] 缓存损坏，重新推理 | {}", e)

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
            seg_smooth = _snap_mask_to_edges(seg_smooth, image, edge_band=5)
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

    # ── Cache save ──────────────────────────────────────────
    if cache_path:
        try:
            mask_meta = []
            for m in masks:
                mask_meta.append({
                    "area": int(m.get("area", 0)),
                    "bbox": [int(v) for v in m.get("bbox", [0, 0, 0, 0])],
                })
            tmp = cache_path + ".tmp.npz"
            np.savez_compressed(tmp, region_map=region_map, mask_meta=np.array(mask_meta, dtype=object))
            os.replace(tmp, cache_path)
            logger.info("[SAM引擎] 缓存已保存 | path={}", cache_path)
        except Exception as e:
            logger.warning("[SAM引擎] 缓存保存失败 | {}", e)

    return masks, region_map


def _reconstruct_masks(region_map: np.ndarray, mask_meta: np.ndarray) -> list[dict]:
    """从 region_map 和 metadata 重建 SAM masks 列表。"""
    masks: list[dict] = []
    n = region_map.max()
    for i in range(1, n + 1):
        seg = region_map == i
        meta = mask_meta[i - 1] if i <= len(mask_meta) else {}
        masks.append({
            "segmentation": seg,
            "area": int(meta.get("area", seg.sum())),
            "bbox": [int(v) for v in meta.get("bbox", [0, 0, 0, 0])],
        })
    return masks


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
    # squeeze 掉多余的 batch dim — predictor 内部会自行添加
    # (1, 1, 256, 256) → (1, 256, 256)，否则 predictor 二次包装成 5D 报错
    mask_logits = mask_logits.squeeze(0)

    # ── SAM 推理 ───────────────────────────────────────────
    with torch.inference_mode():
        masks, scores, _logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            mask_input=mask_logits,
            multimask_output=False,
        )

    # masks[0] 可能是 torch tensor 或 numpy array（取决于设备/追踪路径）
    refined_sam = np.asarray(masks[0])
    score = float(scores[0])
    logger.debug("[SAM精修]   置信度={:.3f}", score)

    # ── 上采样回原始分辨率 ─────────────────────────────────
    if (sam_h, sam_w) != (orig_h, orig_w):
        refined = _upscale_mask_smooth(refined_sam, orig_h, orig_w)
    else:
        refined = refined_sam

    # ── 边缘精修（导向滤波） ─────────────────────────
    refined = _snap_mask_to_edges(refined, image, edge_band=max(5, edge_band))

    logger.debug(
        "[SAM精修] 完成 | refined_fg={:.1f}%",
        100 * refined.sum() / refined.size,
    )

    # ── 质量门：SAM 不可信时退回原始蒙版 ───────────────────
    refined_fg_pct = refined.sum() / refined.size
    rough_fg_pct = rough_bin.sum() / rough_bin.size
    if score < 0.5 or refined_fg_pct < rough_fg_pct * 0.1:
        logger.debug(
            "[SAM精修] 不可信 (置信度={:.3f} refined={:.1f}% rough={:.1f}%) → 使用原始蒙版",
            score, 100 * refined_fg_pct, 100 * rough_fg_pct,
        )
        return rough_bin

    return refined


def get_sam_info() -> dict:
    """Return SAM runtime info (device, model status)."""
    return {
        "device": _device or _resolve_device(),
        "model_loaded": _sam_model is not None,
        "model_path": MODEL_PATH,
        "preset": "paper_sculpture",
    }
