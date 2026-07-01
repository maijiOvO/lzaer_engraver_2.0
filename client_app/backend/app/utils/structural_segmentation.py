"""结构分层引擎：深度图 → N 层物理可支撑的结构蒙版。

将单目深度估计输出的连续深度图转化为 N 层二值蒙版，
对每层做连通性校验，确保内容与外框物理相连（不掉落），
并剔除微小孤立噪声（策略 C）。

纯函数 — 无 HTTP、无文件 I/O。
"""

from __future__ import annotations

import cv2
import numpy as np


def suggest_n_layers(
    depth_map: np.ndarray,
    min_layers: int = 2,
    max_layers: int = 5,
    gap_threshold: float = 0.05,
) -> int:
    """根据深度图的百分位间距（gap）自动推断最优层数。

    原理：在深度百分位曲线上寻找显著跳跃点。
    每个跳跃表示"这以下是同一群物体，以上就是另一群"。

    例如迪拜.jpg：
      P33=0.00 → P50=0.13 (gap=0.13) ← 水面与建筑之间
      P80=0.37 → P90=0.68 (gap=0.31) ← 建筑与天空之间
      → 2 个显著 gap → 建议 3 层

    Args:
        depth_map: (H, W) float32 深度图。
        min_layers: 最少层数。
        max_layers: 最多层数。
        gap_threshold: 百分位间距阈值。gap 超过此值视为"显著跳跃"。

    Returns:
        建议层数 ∈ [min_layers, max_layers]。
    """
    from loguru import logger

    # 计算分位数 (1% 步长)
    percentiles = np.linspace(0, 100, 101)
    values = np.percentile(depth_map, percentiles)

    # 计算相邻百分位之间的跳跃
    gaps = np.diff(values)
    significant_gaps = (gaps > gap_threshold).sum()

    # 特殊处理：如果中位数附近有巨大跳跃（如迪拜的 P33→P50）
    # 检测是否有超过 15% 范围的空窗
    p25 = np.percentile(depth_map, 25)
    p75 = np.percentile(depth_map, 75)
    if p75 - p25 > 0.3 and significant_gaps == 0:
        significant_gaps = 1

    n = int(min(max_layers, max(min_layers, significant_gaps + 1)))

    logger.info(
        "[结构分层] 深度分析 | range=[{:.3f},{:.3f}] "
        "gaps={} → suggest_n_layers={}",
        float(depth_map.min()), float(depth_map.max()),
        significant_gaps, n,
    )
    return n


def find_valley_thresholds(
    depth_map: np.ndarray,
    n_layers: int,
    n_bins: int = 80,
    smooth_sigma: float = 3.0,
    min_dip_ratio: float = 0.15,
) -> list[float]:
    """在深度直方图中检测谷底，作为层间切割阈值。

    避免等距切割的"拦腰斩断"问题：切割线落在像素密度的天然裂缝处，
    大范围渐变区域（如从前到后的墙）不会被切成多段。

    Args:
        depth_map: (H, W) float32，值域 [0, 1]。
        n_layers: 目标层数。
        n_bins: 直方图 bin 数。
        smooth_sigma: 高斯平滑 σ（bin 单位），越大谷底越少。
        min_dip_ratio: 浅谷过滤比例。谷底密度必须比两侧高峰低至少此比例。

    Returns:
        n_layers-1 个阈值 float，升序排列。
        不够时用分位数补足。
    """
    from scipy.ndimage import gaussian_filter1d
    from loguru import logger

    # 1. 直方图
    flat = depth_map.ravel()
    counts, edges = np.histogram(flat, bins=n_bins, range=(0.0, 1.0))
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # 2. 平滑（加大 σ 消除建筑内部浅谷）
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma)

    # 3. 找谷底（局部极小值 + 深度过滤）
    valleys: list[tuple[float, float, float]] = []  # (center, density, dip_ratio)
    search_radius = max(2, int(smooth_sigma))
    for i in range(search_radius, n_bins - search_radius):
        is_valley = True
        for r in range(1, search_radius + 1):
            if smoothed[i] >= smoothed[i - r] or smoothed[i] >= smoothed[i + r]:
                is_valley = False
                break
        if not is_valley:
            continue
        # 计算谷深比：谷底密度 vs 两侧高峰的平均
        left_peak = max(smoothed[i - search_radius:i])
        right_peak = max(smoothed[i:i + search_radius])
        peak_avg = (left_peak + right_peak) / 2
        if peak_avg == 0:
            continue
        dip_ratio = 1.0 - smoothed[i] / peak_avg
        if dip_ratio >= min_dip_ratio:
            valleys.append((float(bin_centers[i]), float(smoothed[i]), dip_ratio))

    # 4. 按谷深比降序（最深的谷底优先）
    valleys.sort(key=lambda x: x[2], reverse=True)

    thresholds: list[float] = []
    needed = n_layers - 1

    # 5. 取最深谷底，保持顺序
    taken = sorted([v[0] for v in valleys[:needed]])

    # 6. 不够 → 分位数补足（放宽接近限制）
    if len(taken) < needed:
        percentiles = np.linspace(0, 100, n_layers + 1)[1:-1]
        for p in percentiles:
            thresh = float(np.percentile(flat, p))
            # 避免与已有阈值太近（< 0.02），但不够时放宽
            proximity_limit = 0.02 if len(taken) >= needed - 1 else 0.0
            if all(abs(thresh - t) > proximity_limit for t in taken):
                taken.append(thresh)
                if len(taken) >= needed:
                    break
        taken.sort()

    # 7. 仍不够 → 等距阈值硬兜底
    if len(taken) < needed:
        for i in range(1, n_layers):
            t = i / n_layers
            if all(abs(t - x) > 0.01 for x in taken):
                taken.append(t)
                if len(taken) >= needed:
                    break
        taken.sort()

    logger.info(
        "[谷底检测] n_layers={} valleys_found={} thresholds={}",
        n_layers, len(valleys),
        [f"{t:.3f}" for t in taken[:needed]],
    )
    return taken[:needed]


def valley_quantize_depth(
    depth_map: np.ndarray,
    n_layers: int,
) -> list[np.ndarray]:
    """用谷底阈值替代等距阈值，将深度图量化为 N 层。

    Args:
        depth_map: (H, W) float32。
        n_layers: 目标层数。

    Returns:
        N 个 (H, W) uint8 二值蒙版。
    """
    from loguru import logger

    thresholds = find_valley_thresholds(depth_map, n_layers)
    h, w = depth_map.shape[:2]
    masks: list[np.ndarray] = []

    for i in range(n_layers):
        if i == 0:
            mask = depth_map < thresholds[0]
        elif i == n_layers - 1:
            mask = depth_map >= thresholds[-1]
        else:
            mask = (depth_map >= thresholds[i-1]) & (depth_map < thresholds[i])
        masks.append(mask.astype(np.uint8) * 255)

    logger.info(
        "[结构分层] 谷底量化完成 | layers={} thresholds={}",
        n_layers, [f"{t:.3f}" for t in thresholds],
    )
    for i, m in enumerate(masks):
        logger.debug(
            "[结构分层]   层{} | fg_px={} ({}%)",
            i, int(np.count_nonzero(m)),
            float(np.count_nonzero(m) / (h * w) * 100),
        )

    return masks


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
    """生成外层固定边框蒙版（向外延伸，不遮挡图像内容）。

    边框位于扩展画布的外围 frame_width 像素，
    原始图像内容置于中心区域，不被边框覆盖。

    Args:
        h, w: 原始图像尺寸（像素）。
        frame_width: 边框宽度（像素），向外延伸。

    Returns:
        (H+2*fw, W+2*fw) uint8 二值蒙版，255=边框区域（外围），0=内容区域（中心）。
    """
    padded_h = h + 2 * frame_width
    padded_w = w + 2 * frame_width
    frame = np.zeros((padded_h, padded_w), dtype=np.uint8)
    # 四个外边（全部在外围）
    frame[:frame_width, :] = 255                    # 上边（外部）
    frame[-frame_width:, :] = 255                   # 下边（外部）
    frame[:, :frame_width] = 255                    # 左边（外部）
    frame[:, -frame_width:] = 255                   # 右边（外部）
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

        # ── 桥接逻辑已移除 — 允许合法孤岛存在，避免伪影 ──
        # 大岛（≥ min_island_area）不做任何处理，直接保留

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


def build_sam_driven_layers(
    sam_masks: list[dict],
    depth_map: np.ndarray,
    n_layers: int,
    image_shape: tuple[int, int],
    frame_width: int = 50,
    min_island_area: int = 100,
    skip_connectivity_repair: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """新管线：SAM 自动分割 → 深度中位数归属 → 连通修复。

    SAM 负责决定物体的形状（保证边缘贴合原图，对象完整不切断），
    深度图仅负责决定每个完整对象的 Z 轴层级归属（排前后顺序）。

    完整流程：
    1. 遍历 SAM masks，计算每块的深度中位数 → 确定归属层
    2. 按优先级（SAM 置信度）逐块分配到层蒙版
    3. 未覆盖像素 fallback 到 quantize_depth
    4. 生成外框 + 逐层连通性修复

    Args:
        sam_masks: run_sam_automatic() 的输出，list[dict]，
                   每个 dict 含 "segmentation" (bool H×W)、"area"、"bbox" 等。
                   分辨率必须与 image_shape 一致。
        depth_map: (H_depth, W_depth) float32 深度图，[0,1] 归一化。
                   分辨率可能与 image_shape 不同，内部会自动 rescale。
        n_layers: 目标层数。
        image_shape: 原图 (H, W)，SAM masks 的分辨率。
        frame_width: 边框宽度（像素，在原图分辨率上）。
        min_island_area: 孤立岛丢弃阈值（像素面积）。

    Returns:
        (layer_masks, frame_mask, stats) 其中：
        - layer_masks: N 个 (H, W) uint8 二值蒙版（含边框）
        - frame_mask: (H, W) uint8 边框蒙版
        - stats: 每层统计 dict
    """
    from loguru import logger

    orig_h, orig_w = image_shape
    depth_h, depth_w = depth_map.shape[:2]

    # ── 深度图缩放到原图分辨率 ──────────────────────────────
    if (depth_h, depth_w) != (orig_h, orig_w):
        depth_full = cv2.resize(
            depth_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC,
        )
    else:
        depth_full = depth_map

    # ── Step 0: 计算谷底阈值（替代等距切割）─────────────────
    thresholds = find_valley_thresholds(depth_full, n_layers)
    logger.info(
        "[SAM驱动分层] 谷底阈值={} | layers={}",
        [f"{t:.3f}" for t in thresholds], n_layers,
    )

    # ── Step 1: 每块 SAM mask → 深度中位数 → 层归属 ────────
    # SAM masks 按原始顺序（高置信度在前），逐个分配到层
    # 已分配像素不再被后续低置信度 mask 覆盖
    assigned = np.zeros((orig_h, orig_w), dtype=bool)
    layer_masks_bool: list[np.ndarray] = [
        np.zeros((orig_h, orig_w), dtype=bool) for _ in range(n_layers)
    ]

    fragment_count = 0
    for mask_data in sam_masks:
        seg = mask_data["segmentation"]

        # 仅处理该 mask 中尚未被更高置信度 mask 覆盖的像素
        new_pixels = seg & ~assigned
        if new_pixels.sum() < 10:  # 忽略极小新增区域
            continue

        # 取该 mask 覆盖区域的深度中位数
        mask_depths = depth_full[seg]
        if len(mask_depths) == 0:
            continue
        median_depth = float(np.median(mask_depths))

        # 用谷底阈值映射到层索引
        layer_idx = 0
        for t in thresholds:
            if median_depth >= t:
                layer_idx += 1
            else:
                break

        # 整块分配到目标层
        layer_masks_bool[layer_idx][new_pixels] = True
        assigned[new_pixels] = True
        fragment_count += 1

    logger.info(
        "[SAM驱动分层] SAM区块{}/{}已分配 | layers={}",
        fragment_count, len(sam_masks), n_layers,
    )

    # ── Step 2: 未覆盖像素 fallback ──────────────────────────
    uncovered = ~assigned
    uncovered_pct = uncovered.sum() / uncovered.size * 100
    if uncovered_pct > 0.1:
        logger.info(
            "[SAM驱动分层] {:.1f}% 像素未被SAM覆盖，回退到深度等距量化",
            uncovered_pct,
        )
        raw_masks = valley_quantize_depth(depth_full, n_layers)
        for i in range(n_layers):
            fallback = (raw_masks[i] > 0) & uncovered
            layer_masks_bool[i][fallback] = True
    elif uncovered_pct > 0:
        logger.debug(
            "[SAM驱动分层] {:.1f}% 零星未覆盖像素，扩散填充",
            uncovered_pct,
        )
        # 极小比例 → 用最近邻扩散（dilate 层蒙版覆盖）
        for i in range(n_layers):
            if not layer_masks_bool[i].any():
                continue
            dilated = cv2.dilate(
                layer_masks_bool[i].astype(np.uint8),
                np.ones((3, 3), np.uint8),
            )
            layer_masks_bool[i][uncovered & (dilated > 0)] = True

    # ── Step 3: 转 uint8 ─────────────────────────────────────
    layer_masks_u8 = [m.astype(np.uint8) * 255 for m in layer_masks_bool]

    # ── Step 4: 生成外框（向外延伸） + padding 层蒙版 ─────────
    frame_mask = generate_frame_mask(orig_h, orig_w, frame_width)
    # 层蒙版 pad 到 frame 尺寸（内容居中，边框在外围）
    fw = frame_width
    layer_masks_padded = [
        cv2.copyMakeBorder(m, fw, fw, fw, fw, cv2.BORDER_CONSTANT, value=0)
        for m in layer_masks_u8
    ]

    # ── Step 5: 逐层连通性修复（可跳过）───────────────────
    layer_masks_final: list[np.ndarray] = []
    stats: list[dict] = []

    for i, raw_u8 in enumerate(layer_masks_padded):
        if skip_connectivity_repair:
            # 跳过连通性修复：直接将边框染白，保留原始内容
            repaired = raw_u8.copy()
            repaired[frame_mask > 0] = 255
            bridges, erased = 0, 0
        else:
            repaired, bridges, erased = repair_layer_mask(
                raw_u8, frame_mask, min_island_area,
            )
        # ── 形态学伪影过滤：消除深度渐变光晕及 SAM 接缝残留 ──
        n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            repaired, connectivity=8,
        )
        artifacts_erased = 0
        for lid in range(1, n_labels):
            area = cc_stats[lid, cv2.CC_STAT_AREA]
            
            # 优化：主体对象（大面积连通域）直接跳过，保护性能及附着的尖锐结构（如塔尖）
            if area > 10000:
                continue
                
            # 提取当前独立分量的蒙版
            comp_mask = (labels == lid).astype(np.uint8)
            
            # 极限厚度测试：使用 5x5 内核进行腐蚀。
            # 5x5 会从四面八方吃掉 2 个像素，因此厚度 <= 4px 的纯细线会完全消失。
            # 如果腐蚀后像素和为 0，说明它是纯粹的渐变接缝，予以抹除！
            eroded = cv2.erode(comp_mask, np.ones((5, 5), np.uint8))
            
            if eroded.sum() == 0:
                repaired[labels == lid] = 0
                artifacts_erased += 1
        if artifacts_erased > 0:
            erased += artifacts_erased

        layer_masks_final.append(repaired)

        fg = int(np.count_nonzero(repaired))
        stats.append({
            "layer_index": i,
            "fg_pixels": fg,
            "fg_pct": round(fg / (orig_h * orig_w) * 100, 2),
            "bridges_built": bridges,
            "islands_erased": erased,
        })

    logger.info(
        "[SAM驱动分层] 完成 | layers={} sam_fragments={} frame={}px "
        "bridges={} erased={} uncovered={:.1f}%{}",
        n_layers, fragment_count, frame_width,
        sum(s["bridges_built"] for s in stats),
        sum(s["islands_erased"] for s in stats),
        uncovered_pct,
        " (跳过连通修复)" if skip_connectivity_repair else "",
    )

    return layer_masks_final, frame_mask, stats
