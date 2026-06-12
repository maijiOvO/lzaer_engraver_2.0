"""结构分层服务层 — 深度引导 + SAM 精修。

用 Depth-Anything-V2 估计单目深度，将连续深度量化为 N 层
物理可支撑的结构蒙版，可选 SAM 逐层边界精修。

完全替代旧的 K-means 聚类方案（2026-06-23 架构升级）。
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from fastapi import HTTPException
from loguru import logger

from app.models.requests import SegmentParams
from app.models.responses import SegmentResponse, LayerInfo

OUTPUTS_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

# ── Lazy-import 引擎 ────────────────────────────────────────────────
try:
    from app.utils.depth_engine import estimate_depth as _depth_engine
except ImportError:
    _depth_engine = None

try:
    from app.utils.structural_segmentation import build_structural_layers
except ImportError:
    build_structural_layers = None

try:
    from app.utils.sam_engine import refine_mask as _sam_refine
except ImportError:
    _sam_refine = None

# ── Overlay colors (BGR) for up to 5 layers ──────────────────────────
LAYER_COLORS = [
    (231, 76, 60),    # red
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
]

# ── SAM quality presets ──────────────────────────────────────────────
# 现在 quality 控制的是是否启用 SAM 精修，不再是 max_dim
SAM_QUALITY_PRESETS: dict[str, dict] = {
    "draft":    {"enable_refine": False, "label": "快速预览 (无SAM精修)"},
    "standard": {"enable_refine": True,  "label": "标准质量"},
    "fine":     {"enable_refine": True,  "label": "精细导出 (增强边缘)"},
}
DEFAULT_QUALITY = "standard"


# ── 深度缓存 ────────────────────────────────────────────────────────

def _depth_cache_path(image_id: str) -> str:
    """深度图缓存文件路径。"""
    return os.path.join(OUTPUTS_DIR, f"{image_id}_depth.npy")


def _load_depth_cache(image_id: str) -> np.ndarray | None:
    """加载缓存的深度图。"""
    path = _depth_cache_path(image_id)
    if not os.path.exists(path):
        return None
    try:
        depth = np.load(path)
        logger.info("深度缓存 HIT | image_id={} shape={}", image_id, depth.shape)
        return depth
    except Exception:
        logger.warning("深度缓存损坏，将重新推理 | path={}", path)
        return None


def _save_depth_cache(image_id: str, depth_map: np.ndarray):
    """持久化深度图。"""
    path = _depth_cache_path(image_id)
    tmp = path + ".tmp.npy"
    np.save(tmp, depth_map)
    os.replace(tmp, path)
    logger.info("深度缓存已保存 | path={}", path)


# ── 图像定位 ────────────────────────────────────────────────────────

def _find_original_image(image_id: str) -> str:
    """在 outputs/ 中定位上传的原图。"""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(OUTPUTS_DIR, f"{image_id}_original{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Original image not found for image_id={image_id}"
    )


# ── 渲染 ────────────────────────────────────────────────────────────

def _render_overlay(
    image: np.ndarray,
    layer_masks: list[np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    """将每层蒙版以不同颜色叠加到原图上。"""
    h, w = image.shape[:2]
    canvas = image.copy().astype(np.float32)

    for i, mask in enumerate(layer_masks):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        color_arr = np.array(color, dtype=np.float32)
        fg = mask > 0
        canvas[fg] = canvas[fg] * (1 - alpha) + color_arr * alpha

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ── 主流程 ──────────────────────────────────────────────────────────

def process_segment(params: SegmentParams) -> SegmentResponse:
    """深度引导结构分层 + SAM 精修。

    Pipeline:
    1. 加载原图
    2. 深度估计（或从缓存加载）
    3. 深度量化 + 外框生成 + 连通性修复
    4. 可选：SAM 逐层边界精修
    5. 渲染叠加图 + 保存输出
    6. 返回 SegmentResponse
    """
    t0 = time.perf_counter()

    # ── 引擎状态检查 ─────────────────────────────────────────────
    if _depth_engine is None:
        raise HTTPException(
            status_code=501,
            detail="深度引擎未安装 — 检查 app/utils/depth_engine.py",
        )
    if build_structural_layers is None:
        raise HTTPException(
            status_code=501,
            detail="结构分层引擎未安装 — 检查 app/utils/structural_segmentation.py",
        )

    # ── 1. 加载原图 ─────────────────────────────────────────────
    try:
        original_path = _find_original_image(params.image_id)
        logger.info("[结构分层] 输入: {}", original_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image = cv2.imread(original_path)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"无法解码图像: {original_path}",
        )

    orig_h, orig_w = image.shape[:2]

    # 质量预设
    quality = params.sam_quality if params.sam_quality in SAM_QUALITY_PRESETS else DEFAULT_QUALITY
    qconf = SAM_QUALITY_PRESETS[quality]
    enable_refine = qconf["enable_refine"]

    logger.info(
        "[结构分层] 参数 | image_id={} shape=({},{}) n_layers={} "
        "frame={}px min_island={}px quality={} refine={}",
        params.image_id, orig_w, orig_h, params.n_layers,
        params.frame_width, params.min_island_area,
        quality, enable_refine,
    )

    # ── 2. 深度估计 ─────────────────────────────────────────────
    force = getattr(params, "force_recompute", False)
    depth_map = None

    if not force:
        depth_map = _load_depth_cache(params.image_id)

    if depth_map is None:
        logger.info("[结构分层] 运行深度估计 …")
        depth_t0 = time.perf_counter()
        try:
            depth_map = _depth_engine(image)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"深度估计失败 | 原因: {type(e).__name__}: {e}",
            )
        depth_elapsed = int((time.perf_counter() - depth_t0) * 1000)
        logger.info("[结构分层] 深度估计完成 | time={}ms", depth_elapsed)

        try:
            _save_depth_cache(params.image_id, depth_map)
        except Exception as e:
            logger.warning("[结构分层] 深度缓存保存失败（不影响结果）| {}", e)
    else:
        depth_elapsed = 0

    # ── 3. 结构分层 ─────────────────────────────────────────────
    logger.info("[结构分层] 构建{}层结构蒙版 …", params.n_layers)
    layer_t0 = time.perf_counter()
    try:
        layer_masks, frame_mask, layer_stats = build_structural_layers(
            depth_map,
            n_layers=params.n_layers,
            frame_width=params.frame_width,
            min_island_area=params.min_island_area,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"结构分层失败 | 原因: {type(e).__name__}: {e}",
        )
    layer_elapsed = int((time.perf_counter() - layer_t0) * 1000)

    for s in layer_stats:
        logger.info(
            "[结构分层]   层{} | fg={}px({}%) bridges={} erased={}",
            s["layer_index"], s["fg_pixels"], s["fg_pct"],
            s["bridges_built"], s["islands_erased"],
        )

    # ── 4. SAM 逐层精修 ─────────────────────────────────────────
    refine_ms = 0
    if enable_refine and _sam_refine is not None:
        logger.info("[结构分层] SAM 逐层边界精修 …")
        refine_t0 = time.perf_counter()

        edge_band = 5 if quality == "fine" else 3

        for i, mask in enumerate(layer_masks):
            if not mask.any():
                continue  # 空层跳过
            try:
                refined = _sam_refine(image, mask, edge_band=edge_band)
                # 保留边框（SAM 精修不改边框区域）
                frame_bin = frame_mask > 0
                refined_u8 = refined.astype(np.uint8) * 255
                refined_u8[frame_bin] = 255
                layer_masks[i] = refined_u8
                layer_stats[i]["fg_pixels"] = int(np.count_nonzero(refined_u8))
            except Exception as e:
                logger.warning(
                    "[结构分层] 层{} SAM精修失败（使用原始蒙版）| {}",
                    i, e,
                )

        refine_ms = int((time.perf_counter() - refine_t0) * 1000)
        logger.info("[结构分层] SAM精修完成 | time={}ms", refine_ms)
    elif enable_refine and _sam_refine is None:
        logger.warning("[结构分层] SAM精修引擎未安装，跳过")

    # ── 5. 渲染叠加图 ──────────────────────────────────────────
    logger.debug("[结构分层] 渲染叠加图")
    try:
        overlay = _render_overlay(image, layer_masks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"渲染叠加图失败 | {type(e).__name__}: {e}",
        )

    # ── 6. 保存输出 ────────────────────────────────────────────
    # 叠加图
    overlay_path = os.path.join(
        OUTPUTS_DIR, f"{params.image_id}_segmented.png"
    )
    tmp_overlay = os.path.join(
        OUTPUTS_DIR, f".tmp_{params.image_id}_segmented.png"
    )
    try:
        cv2.imwrite(tmp_overlay, overlay)
        os.replace(tmp_overlay, overlay_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"保存叠加图失败 | {type(e).__name__}: {e}",
        )
    logger.info("[结构分层] 叠加图已保存 | path={}", overlay_path)

    # 每层蒙版（含边框版）
    layers_info: list[LayerInfo] = []
    for rank, mask in enumerate(layer_masks):
        mask_path = os.path.join(
            OUTPUTS_DIR, f"{params.image_id}_mask_{rank}.png"
        )
        tmp_mask = os.path.join(
            OUTPUTS_DIR, f".tmp_{params.image_id}_mask_{rank}.png"
        )
        try:
            cv2.imwrite(tmp_mask, mask)
            os.replace(tmp_mask, mask_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"保存第{rank}层蒙版失败 | {type(e).__name__}: {e}",
            )

        # 不含边框的纯内容版（用于 layer_labeler 等工具对比）
        pure_mask = mask.copy()
        pure_mask[frame_mask > 0] = 0
        frame_path = os.path.join(
            OUTPUTS_DIR, f"{params.image_id}_frame_{rank}.png"
        )
        tmp_frame = os.path.join(
            OUTPUTS_DIR, f".tmp_{params.image_id}_frame_{rank}.png"
        )
        try:
            cv2.imwrite(tmp_frame, pure_mask)
            os.replace(tmp_frame, frame_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"保存第{rank}层边框文件失败 | {type(e).__name__}: {e}",
            )

        layers_info.append(LayerInfo(
            layer_index=rank,
            mask_url=f"/outputs/{params.image_id}_mask_{rank}.png",
            frame_url=f"/outputs/{params.image_id}_frame_{rank}.png",
        ))
        logger.info(
            "[结构分层] 层{} 已保存 | mask={} frame={} fg={}px",
            rank, mask_path, frame_path, layer_stats[rank]["fg_pixels"],
        )

    # ── 7. 汇总 ─────────────────────────────────────────────────
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[结构分层] 完成 | layers={} total={}ms "
        "(depth={}ms layers={}ms refine={}ms)",
        params.n_layers, elapsed_ms,
        depth_elapsed, layer_elapsed, refine_ms,
    )

    return SegmentResponse(
        overlay_url=f"/outputs/{params.image_id}_segmented.png",
        layers=layers_info,
    )
