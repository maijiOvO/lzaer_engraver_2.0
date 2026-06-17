"""
SLIC 超像素 + 深度投票 分割引擎。

基于 RGB 图像的 SLIC 超像素提供精确边缘（不切建筑物），
Depth-Anything 深度图为每个超像素投票分配层级。

仅用于 dev_tools/labeler，不涉及 client_app。
"""

from __future__ import annotations

import numpy as np
import cv2


def slic_depth_layers(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    n_layers: int = 3,
    n_segments: int = 300,
    compactness: float = 10.0,
    border_width: int = 4,
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """SLIC 超像素分割 + 深度投票 → 结构蒙版。

    流程：
    1. SLIC 将图像分为 ~n_segments 个超像素（边界跟随 RGB 边缘）
    2. 每超像素取深度中位数 → 投票深度值
    3. 按分位数将超像素分入 N 层
    4. 生成每层二值蒙版 + 外框

    Args:
        image_bgr: BGR 图像 (H, W, 3)。分辨率应与 depth_map 一致。
        depth_map: float32 深度图 (H, W)，值域 [0, 1]。
        n_layers: 目标层数。
        n_segments: 超像素大约数量。
        compactness: SLIC compactness。越大越方，越小越贴合边缘。
        border_width: 外框宽度（像素）。

    Returns:
        (layer_masks, frame_mask, stats)
        - layer_masks: N 个 (H, W) uint8 二值蒙版，255=该层内容+边框
        - frame_mask: (H, W) uint8 边框蒙版
        - stats: 每层统计
    """
    from skimage.segmentation import slic

    h, w = depth_map.shape[:2]

    # ── Step 1: SLIC 超像素 ──────────────────────────────────
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    segments = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=2,
    )
    n_segs_actual = segments.max() + 1

    # ── Step 2: 每超像素深度投票（中位数）───────────────────
    seg_depths = np.full(n_segs_actual, np.nan, dtype=np.float32)
    for sid in range(n_segs_actual):
        mask = segments == sid
        if mask.any():
            seg_depths[sid] = np.median(depth_map[mask])

    # 对没有深度值的 segment（不应出现），填入全局中位数
    nan_mask = np.isnan(seg_depths)
    if nan_mask.any():
        seg_depths[nan_mask] = np.median(seg_depths[~nan_mask])

    # ── Step 3: 分位数分层 ──────────────────────────────────
    percentiles = np.linspace(0, 100, n_layers + 1)
    thresholds = np.percentile(seg_depths, percentiles)
    # 确保阈值严格递增
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] + 1e-6

    # ── Step 4: 生成层蒙版 ──────────────────────────────────
    layer_masks: list[np.ndarray] = []
    for i in range(n_layers):
        lo = thresholds[i]
        hi = thresholds[i + 1]

        # 最后一层包含上限
        if i == n_layers - 1:
            seg_in_layer = (seg_depths >= lo) & (seg_depths <= hi)
        else:
            seg_in_layer = (seg_depths >= lo) & (seg_depths < hi)

        # 找到属于该层的 segment IDs
        layer_seg_ids = np.where(seg_in_layer)[0]

        # 构建蒙版
        mask = np.zeros((h, w), dtype=np.uint8)
        for sid in layer_seg_ids:
            mask[segments == sid] = 255

        layer_masks.append(mask)

    # ── Step 5: 外框蒙版 ────────────────────────────────────
    frame_mask = np.zeros((h, w), dtype=np.uint8)
    frame_mask[:border_width, :] = 255
    frame_mask[-border_width:, :] = 255
    frame_mask[:, :border_width] = 255
    frame_mask[:, -border_width:] = 255

    # 每层都包含外框（结构完整性）
    for mask in layer_masks:
        mask[frame_mask > 0] = 255

    # ── Step 6: 统计 ────────────────────────────────────────
    stats: list[dict] = []
    for i, mask in enumerate(layer_masks):
        fg = int(np.count_nonzero(mask))
        lo = thresholds[i]
        hi = thresholds[i + 1]
        if i == n_layers - 1:
            in_layer = int(((seg_depths >= lo) & (seg_depths <= hi)).sum())
        else:
            in_layer = int(((seg_depths >= lo) & (seg_depths < hi)).sum())
        stats.append({
            "layer_index": i,
            "fg_pixels": fg,
            "fg_pct": round(fg / (h * w) * 100, 2),
            "bridges_built": 0,
            "islands_erased": 0,
            "n_segments_in_layer": in_layer,
        })

    return layer_masks, frame_mask, stats
