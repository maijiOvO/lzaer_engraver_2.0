"""分割结果自动评分引擎 — 5 维量化 + 综合评分。

纯函数，零副作用。输入分割统计，输出评分字典。
供 dev_tools 脚本内部使用。
"""

from __future__ import annotations

import numpy as np


def score_segmentation(
    layer_stats: list[dict],
    image_shape: tuple[int, int],
) -> dict:
    """对分割结果进行 5 维自动评分。

    Args:
        layer_stats: build_structural_layers 返回的 stats 列表。
                    每项含: layer_index, fg_pixels, bridges_built, islands_erased。
        image_shape: (height, width) 原图尺寸。

    Returns:
        {layer_balance, bridge_efficiency, island_purity, coverage_efficiency, combined_score}
    """
    n = len(layer_stats)
    if n == 0:
        return _zero_score()

    h, w = image_shape
    total_px = h * w

    fg_pixels = np.array([s["fg_pixels"] for s in layer_stats], dtype=np.float64)
    bridges = np.array([s["bridges_built"] for s in layer_stats], dtype=np.float64)
    erased = np.array([s["islands_erased"] for s in layer_stats], dtype=np.float64)

    total_fg = fg_pixels.sum()
    total_bridges = bridges.sum()
    total_erased = erased.sum()

    if total_fg == 0:
        return _zero_score()

    # 1. 层面积均衡度
    fg_pcts = fg_pixels / total_px
    mean_pct = fg_pcts.mean()
    cv = fg_pcts.std() / mean_pct if mean_pct > 0 else 1.0
    layer_balance = float(1.0 - min(cv, 1.0))

    # 2. 桥接效率 (越少越好)
    bridge_ratio = total_bridges / total_fg
    bridge_efficiency = float(1.0 - min(bridge_ratio * 10, 1.0))

    # 3. 碎片纯净度 (越少越好)
    island_ratio = total_erased / total_fg
    island_purity = float(1.0 - min(island_ratio * 5, 1.0))

    # 4. 有效覆盖率
    effective_fg = total_fg - total_bridges
    coverage_efficiency = float(effective_fg / total_fg)

    # 5. 综合评分 (加权乘积)
    combined = float(
        0.35 * layer_balance
        + 0.25 * bridge_efficiency
        + 0.20 * island_purity
        + 0.20 * coverage_efficiency
    )

    return {
        "layer_balance": round(layer_balance, 4),
        "bridge_efficiency": round(bridge_efficiency, 4),
        "island_purity": round(island_purity, 4),
        "coverage_efficiency": round(coverage_efficiency, 4),
        "combined_score": round(combined, 4),
    }


def _zero_score() -> dict:
    return dict.fromkeys([
        "layer_balance", "bridge_efficiency", "island_purity",
        "coverage_efficiency", "combined_score",
    ], 0.0)
