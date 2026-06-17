"""边界精修引擎：深度等距量化 → 提取层间边界带 → SAM 框选精修 → 深度投票回填。

将深度图的"宏观层归属"能力与 SAM 的"原图精度对象边界"能力分离协同：
  深度模型 → "这片属于第 N 层"（层级归属，518px 足够）
  SAM     → "边界在这条线上"（精确边缘，原图精度）

仅在 dev_tools/labeler 中使用，不涉及 client_app。
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from loguru import logger


# ── 公开 API ────────────────────────────────────────────────────────


def extract_boundary_zones(
    layer_masks: list[np.ndarray],
    band_width: int = 5,
    min_component_area: int = 100,
) -> list[dict]:
    """从相邻层蒙版之间提取"需要 SAM 判断归属"的边界带。

    对每一对相邻层 (i, i+1)：
      zone = dilate(layer_i, band_width) ∩ dilate(layer_{i+1}, band_width)

    这个区域是深度图"认为"的过渡带——深度值模糊、等距阈值附近。
    SAM 在这个区域内重新画精确的边界线。

    Args:
        layer_masks: N 个 (H, W) uint8 二值蒙版，255=该层。
        band_width: 边界带膨胀宽度（像素）。
        min_component_area: 跳过小于此面积（px²）的连通分量。

    Returns:
        [{layer_pair, component_id, bbox, mask, area}, ...]
        按面积降序排列。
    """
    h, w = layer_masks[0].shape[:2]
    n_layers = len(layer_masks)
    components: list[dict] = []

    for i in range(n_layers - 1):
        mask_a = layer_masks[i] > 0
        mask_b = layer_masks[i + 1] > 0

        # 膨胀后求交集 = 两层的过渡带
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1))
        dilated_a = cv2.dilate(mask_a.astype(np.uint8), kernel)
        dilated_b = cv2.dilate(mask_b.astype(np.uint8), kernel)
        zone = (dilated_a & dilated_b) > 0

        if not zone.any():
            continue

        # 连通分量分析
        zone_u8 = zone.astype(np.uint8) * 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zone_u8, connectivity=8)

        for lid in range(1, n_labels):
            area = int(stats[lid, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue

            x = int(stats[lid, cv2.CC_STAT_LEFT])
            y = int(stats[lid, cv2.CC_STAT_TOP])
            bw = int(stats[lid, cv2.CC_STAT_WIDTH])
            bh = int(stats[lid, cv2.CC_STAT_HEIGHT])

            components.append({
                "layer_pair": (i, i + 1),
                "component_id": f"L{i}_{lid}",
                "bbox": (x, y, bw, bh),
                "mask": labels == lid,
                "area": area,
            })

    # 按面积降序（大边界优先）
    components.sort(key=lambda c: c["area"], reverse=True)
    logger.info(
        "[边界精修] 提取 {} 个边界带分量 (band={}px min_area={})",
        len(components), band_width, min_component_area,
    )
    return components


def sam_box_refine(
    component: dict,
    sam_predictor: Any,
    sam_h: int,
    sam_w: int,
    box_padding: int = 10,
) -> np.ndarray | None:
    """对单个边界带分量运行 SAM 框选精修。

    SamPredictor 已调用 set_image() 设置了全图。
    predict() 返回的 mask 是全图尺寸，不需要裁剪。

    Args:
        component: extract_boundary_zones 返回的单个分量（bbox 已是 SAM 坐标）。
        sam_predictor: 已调用 set_image() 的 SamPredictor 实例。
        sam_h, sam_w: SAM 推理图像尺寸。
        box_padding: bbox 外扩像素。

    Returns:
        (sam_h, sam_w) bool 精修后的二值蒙版，失败时返回 None。
    """
    import torch

    x, y, bw, bh = component["bbox"]

    # 外扩 bbox
    x1 = max(0, x - box_padding)
    y1 = max(0, y - box_padding)
    x2 = min(sam_w, x + bw + box_padding)
    y2 = min(sam_h, y + bh + box_padding)

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    box_xyxy = np.array([x1, y1, x2, y2])

    try:
        with torch.inference_mode():
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy[None, :],
                multimask_output=False,
            )
    except Exception:
        return None

    refined = np.asarray(masks[0])
    score = float(scores[0])

    if score < 0.5:
        return None

    return refined.astype(bool)


def depth_vote_assign(
    sam_fragment: np.ndarray,
    depth_map: np.ndarray,
    n_layers: int,
) -> int:
    """对 SAM 精修片段做深度投票，决定归属哪一层。

    取片段覆盖区域的深度中位数，映射到等距量化的层级。

    Args:
        sam_fragment: (H, W) bool，精修后的蒙版片段。
        depth_map: (H, W) float32 深度图。
        n_layers: 总层数。

    Returns:
        0..n_layers-1 的层索引。
    """
    if not sam_fragment.any():
        return 0

    # 取片段覆盖区域的深度中位数
    fragment_depth = depth_map[sam_fragment]
    median_depth = float(np.median(fragment_depth))

    # 映射到等距层级
    layer_idx = int(median_depth * n_layers)
    layer_idx = max(0, min(n_layers - 1, layer_idx))
    return layer_idx


def refine_layers(
    image: np.ndarray,
    layer_masks_raw: list[np.ndarray],
    depth_map: np.ndarray,
    band_width: int = 5,
    min_component_area: int = 100,
    box_padding: int = 10,
    max_sam_dim: int = 1200,
) -> tuple[list[np.ndarray], list[dict]]:
    """完整边界精修流程。

    1. 从等距量化 raw 蒙版提取层间边界带
    2. 对每个边界带分量运行 SAM 框选精修
    3. 深度投票决定 SAM 片段归属
    4. 回填到层蒙版

    Args:
        image: BGR 原图 (H_orig, W_orig, 3)。
        layer_masks_raw: 等距量化的 N 层原始二值蒙版（不含外框）。
        depth_map: (H_depth, W_depth) float32 深度图，分辨率可能与 image 不同。
        band_width: 边界带膨胀宽度。
        min_component_area: 跳过过小的分量。
        box_padding: SAM bbox 外扩像素。
        max_sam_dim: SAM 推理时缩放的最大边长。

    Returns:
        (refined_masks, stats)
        - refined_masks: N 个 (H_orig, W_orig) uint8 二值蒙版
        - stats: 每层 + 精修统计
    """
    import torch
    from app.utils.sam_engine import _get_sam_model, _preprocess_image
    from mobile_sam import SamPredictor

    t0 = time.perf_counter()
    h_img, w_img = image.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]
    n_layers = len(layer_masks_raw)

    # ── 将等距蒙版缩放到原图分辨率（如果深度图分辨率不同）──
    if (depth_h, depth_w) != (h_img, w_img):
        layer_masks_full = [
            cv2.resize(m.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            for m in layer_masks_raw
        ]
        depth_scaled = cv2.resize(depth_map, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    else:
        layer_masks_full = [m.astype(np.uint8) for m in layer_masks_raw]
        depth_scaled = depth_map

    layer_masks_bool = [m > 0 for m in layer_masks_full]

    # ── Step 1: 提取边界带 ──────────────────────────────────
    components = extract_boundary_zones(layer_masks_full, band_width, min_component_area)

    if not components:
        logger.info("[边界精修] 无边界带分量需要精修，返回原始等距量化结果")
        elapsed = int((time.perf_counter() - t0) * 1000)
        stats = _build_stats(layer_masks_full, 0, 0)
        return layer_masks_full, stats

    # ── Step 2: 设置 SAM 预测器 ─────────────────────────────
    sam_t0 = time.perf_counter()
    sam_image_rgb = _preprocess_image(image, max_dim=max_sam_dim)
    sam_image_rgb = cv2.cvtColor(sam_image_rgb, cv2.COLOR_BGR2RGB)

    model = _get_sam_model()
    predictor = SamPredictor(model)
    predictor.set_image(sam_image_rgb)
    sam_setup_ms = int((time.perf_counter() - sam_t0) * 1000)
    logger.debug("[边界精修] SAM 预测器就绪 | setup={}ms", sam_setup_ms)

    # ── Step 3: 逐分量 SAM 精修 ─────────────────────────────
    # SAM 在原图分辨率下工作，但 predictor 使用缩放后的坐标
    # 需要转换坐标
    sam_h, sam_w = sam_image_rgb.shape[:2]
    scale_x = sam_w / w_img
    scale_y = sam_h / h_img

    refined_fragments: list[np.ndarray] = []
    fragment_assignments: list[int] = []
    stats_components = {"total": len(components), "refined": 0, "failed": 0, "skipped": 0}

    for comp in components:
        # 将 bbox 转换到 SAM 坐标
        x, y, bw, bh = comp["bbox"]
        x1_sam = int(x * scale_x)
        y1_sam = int(y * scale_y)
        x2_sam = int((x + bw) * scale_x)
        y2_sam = int((y + bh) * scale_y)

        # 构建 SAM 坐标的分量
        comp_sam = {
            **comp,
            "bbox": (x1_sam, y1_sam, x2_sam - x1_sam, y2_sam - y1_sam),
        }

        result = sam_box_refine(comp_sam, predictor, sam_h, sam_w, box_padding=int(box_padding * min(scale_x, scale_y)))

        if result is None:
            stats_components["failed"] += 1
            continue

        # SAM 返回的是 SAM 分辨率的 bool mask，需要缩放回原图
        result_u8 = result.astype(np.uint8) * 255
        result_full = cv2.resize(result_u8, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        refined_bool = result_full > 0

        # 深度投票：决定这个片段属于哪层
        layer_idx = depth_vote_assign(refined_bool, depth_scaled, n_layers)

        refined_fragments.append(refined_bool)
        fragment_assignments.append(layer_idx)
        stats_components["refined"] += 1

    # ── Step 4: 回填 ────────────────────────────────────────
    # 起始状态：等距量化 raw 结果（bool）
    refined = [mask.copy() for mask in layer_masks_bool]

    for fragment, layer_idx in zip(refined_fragments, fragment_assignments):
        # 从所有层中移除该区域
        for i in range(n_layers):
            refined[i][fragment] = False
        # 加入目标层
        refined[layer_idx][fragment] = True

    # 转回 uint8
    refined_u8 = [r.astype(np.uint8) * 255 for r in refined]

    elapsed = int((time.perf_counter() - t0) * 1000)
    stats = _build_stats(refined_u8, stats_components["refined"], stats_components["failed"])
    stats[0]["refine_stats"] = stats_components

    logger.info(
        "[边界精修] 完成 | refined={} failed={} total={}ms",
        stats_components["refined"], stats_components["failed"], elapsed,
    )
    return refined_u8, stats


# ── 内部辅助 ────────────────────────────────────────────────────────


def _build_stats(
    layer_masks: list[np.ndarray],
    refined_count: int,
    failed_count: int,
) -> list[dict]:
    """构建每层统计。"""
    h, w = layer_masks[0].shape[:2]
    stats: list[dict] = []
    for i, mask in enumerate(layer_masks):
        fg = int(np.count_nonzero(mask))
        stats.append({
            "layer_index": i,
            "fg_pixels": fg,
            "fg_pct": round(fg / (h * w) * 100, 2),
            "bridges_built": 0,
            "islands_erased": 0,
            "sam_fragments_refined": refined_count if i == 0 else 0,
            "sam_fragments_failed": failed_count if i == 0 else 0,
        })
    return stats
