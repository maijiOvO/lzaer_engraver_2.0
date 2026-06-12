"""结构分层引擎：深度图 → N 层物理可支撑的结构蒙版。

将单目深度估计输出的连续深度图转化为 N 层二值蒙版，
对每层做连通性校验，确保内容与外框物理相连（不掉落），
并剔除微小孤立噪声（策略 C）。

纯函数 — 无 HTTP、无文件 I/O。
"""

from __future__ import annotations

import cv2
import numpy as np


def quantize_depth(
    depth_map: np.ndarray,
    n_layers: int,
) -> list[np.ndarray]:
    """将连续深度图均匀量化为 N 层二值蒙版。

    Args:
        depth_map: (H, W) float32，值域 [0, 1]。0=近，1=远。
        n_layers: 目标层数 N ∈ [2, 5]。

    Returns:
        N 个 (H, W) uint8 二值蒙版，255=该层，0=其他。
        索引 0 = 前景（最近），索引 N-1 = 背景（最远）。
    """
    from loguru import logger

    h, w = depth_map.shape[:2]
    masks: list[np.ndarray] = []

    for i in range(n_layers):
        lo = i / n_layers
        hi = (i + 1) / n_layers

        if i == n_layers - 1:
            # 最后一层包含上限
            mask = (depth_map >= lo) & (depth_map <= hi)
        else:
            mask = (depth_map >= lo) & (depth_map < hi)

        masks.append(mask.astype(np.uint8) * 255)

    logger.info(
        "[结构分层] 深度量化完成 | layers={} depth_range=[{:.3f},{:.3f}]",
        n_layers, float(depth_map.min()), float(depth_map.max()),
    )
    for i, m in enumerate(masks):
        logger.debug(
            "[结构分层]   层{} | fg_px={} ({}%)",
            i, int(np.count_nonzero(m)), float(np.count_nonzero(m) / (h * w) * 100),
        )

    return masks


def generate_frame_mask(
    h: int,
    w: int,
    frame_width: int = 50,
) -> np.ndarray:
    """生成外层固定边框蒙版。

    边框是一个空心矩形环，位于图像边缘内侧 frame_width 像素。

    Args:
        h, w: 图像尺寸（像素）。
        frame_width: 边框宽度（像素）。

    Returns:
        (H, W) uint8 二值蒙版，255=边框区域。
    """
    frame = np.zeros((h, w), dtype=np.uint8)
    # 四个边
    frame[:frame_width, :] = 255          # 上边
    frame[-frame_width:, :] = 255         # 下边
    frame[:, :frame_width] = 255          # 左边
    frame[:, -frame_width:] = 255         # 右边
    return frame


def _bressenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham 画线——返回直线上所有像素坐标。"""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        points.append((y0, x0))  # (row, col) 顺序
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return points


def repair_layer_mask(
    layer_mask: np.ndarray,
    frame_mask: np.ndarray,
    min_island_area: int = 100,
) -> tuple[np.ndarray, int, int]:
    """策略 C：确保层内容与外框连通，丢弃微小孤立岛。

    1. 找到所有层内连通分量
    2. 与外框不相连的分量：
       - 面积 >= min_island_area → Bresenham 画桥连到外框
       - 面积 <  min_island_area → 擦除
    3. 返回修复后的蒙版 + 统计信息

    Args:
        layer_mask: (H, W) uint8 二值蒙版，255=层内容。
        frame_mask: (H, W) uint8 二值蒙版，255=边框区域。
        min_island_area: 低于此面积（px）的孤立岛直接丢弃。

    Returns:
        (repaired_mask, bridges_built, islands_erased)
    """
    from loguru import logger

    h, w = layer_mask.shape[:2]

    # ── 分离边框和内容 ──────────────────────────────────────────
    frame_bin = frame_mask > 0
    layer_bin = layer_mask > 0

    # 有层内容的区域 = 图层 AND NOT 边框
    content_bin = layer_bin & ~frame_bin

    if not content_bin.any():
        # 该层没有内容（全空层）—— 返回只有边框的蒙版
        logger.debug("[结构分层]   空层 — 无内容")
        return (frame_bin.astype(np.uint8) * 255, 0, 0)

    # ── 连通分量分析 ────────────────────────────────────────────
    content_u8 = content_bin.astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        content_u8, connectivity=8
    )

    if n_labels <= 1:
        # 只有一个分量（背景）—— 内容全空
        return (frame_bin.astype(np.uint8) * 255, 0, 0)

    repaired = layer_bin.copy()  # 在原始蒙版上做修改
    bridges_built = 0
    islands_erased = 0

    for lid in range(1, n_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        comp_mask = labels == lid

        # 检查该分量是否与外框邻接
        # 用膨胀 1px 检查是否碰到 frame
        dilated = cv2.dilate(
            comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8),
        )
        touches_frame = np.any(dilated & frame_bin)

        if touches_frame:
            continue  # 已连通，保留

        if area < min_island_area:
            # 太小 → 直接擦除
            repaired[comp_mask] = False
            islands_erased += 1
            continue

        # ── 大岛 → 桥接到外框 ──────────────────────────────────
        # 用距离变换找到岛上距离外框最近的点
        frame_dist = cv2.distanceTransform(
            (~frame_bin).astype(np.uint8), cv2.DIST_L2, 5,
        )
        # 岛上各点到外框的距离
        island_dist = frame_dist[comp_mask]
        min_idx = np.argmin(island_dist)
        island_coords = np.argwhere(comp_mask)
        island_pt = tuple(island_coords[min_idx])  # (y, x)

        # 在 frame 上找距离 island_pt 最近的点
        frame_coords = np.argwhere(frame_bin)
        diff = frame_coords - np.array(island_pt)
        dists = np.sum(diff ** 2, axis=1)
        frame_pt = tuple(frame_coords[np.argmin(dists)])

        # Bresenham 画桥（1px 宽）
        line_pts = _bressenham_line(
            island_pt[1], island_pt[0],
            frame_pt[1], frame_pt[0],
        )
        for r, c in line_pts:
            if 0 <= r < h and 0 <= c < w:
                repaired[r, c] = True

        bridges_built += 1

    logger.debug(
        "[结构分层]   修复 | bridges={} erased={} components={}",
        bridges_built, islands_erased, n_labels - 1,
    )

    # ── 合并：repaired = 修复后的内容 | 边框 ───────────────────
    final = (repaired | frame_bin).astype(np.uint8) * 255
    return final, bridges_built, islands_erased


def build_structural_layers(
    depth_map: np.ndarray,
    n_layers: int,
    frame_width: int = 50,
    min_island_area: int = 100,
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """从深度图构建 N 层物理可支撑的结构蒙版。

    完整流程：
    1. 深度图等距量化 → N 层二值蒙版
    2. 生成外框蒙版
    3. 逐层连通性修复（策略 C）
    4. 返回蒙版列表 + 边框 + 统计

    Args:
        depth_map: (H, W) float32，[0,1] 归一化深度。
        n_layers: 目标层数。
        frame_width: 边框宽度（像素）。
        min_island_area: 孤立岛丢弃阈值（像素面积）。

    Returns:
        (layer_masks, frame_mask, stats) 其中：
        - layer_masks: N 个 (H, W) uint8 二值蒙版（含边框）
        - frame_mask: (H, W) uint8 边框蒙版
        - stats: 每层统计 dict {layer_index, fg_pixels, bridges, erased}
    """
    from loguru import logger

    h, w = depth_map.shape[:2]

    # Step 1: 深度量化
    raw_masks = quantize_depth(depth_map, n_layers)

    # Step 2: 生成外框
    frame_mask = generate_frame_mask(h, w, frame_width)

    # Step 3: 逐层修复
    layer_masks: list[np.ndarray] = []
    stats: list[dict] = []

    for i, raw in enumerate(raw_masks):
        repaired, bridges, erased = repair_layer_mask(
            raw, frame_mask, min_island_area,
        )
        layer_masks.append(repaired)

        fg = int(np.count_nonzero(repaired))
        stats.append({
            "layer_index": i,
            "fg_pixels": fg,
            "fg_pct": round(fg / (h * w) * 100, 2),
            "bridges_built": bridges,
            "islands_erased": erased,
        })

    logger.info(
        "[结构分层] 完成 | layers={} frame={}px "
        "bridges={} erased={}",
        n_layers, frame_width,
        sum(s["bridges_built"] for s in stats),
        sum(s["islands_erased"] for s in stats),
    )

    return layer_masks, frame_mask, stats
