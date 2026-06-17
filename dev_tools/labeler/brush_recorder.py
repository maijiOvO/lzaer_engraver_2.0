"""笔刷事件记录模块 — 每次笔刷 SAM 精修自动记录，为 ML 积累训练数据。

记录内容：
  - 笔画类型（纳入/排除）、坐标、时间戳
  - 笔画覆盖区域的局部特征（深度梯度、RGB 边缘、纹理等）
  - 关联的层级、图片哈希

存储：dev_tools/data/brush_events/{image_hash}_{timestamp}.json

用途：
  - 训练"精修触发预测器"——哪些边界段需要 SAM 精修
  - 失败模式聚类——哪些特征组合最容易出错
  - 数据质量审计——哪些标定经过了充分人工审核
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ── 路径 ────────────────────────────────────────────────────────────

def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

DATA_DIR = _get_project_root() / "dev_tools" / "data"
EVENTS_DIR = DATA_DIR / "brush_events"
os.makedirs(EVENTS_DIR, exist_ok=True)


# ── 数据模型 ────────────────────────────────────────────────────────

@dataclass
class BrushEvent:
    """一次笔刷 SAM 精修事件。"""
    image_hash: str
    image_name: str
    layer_index: int
    brush_type: str           # "include" | "exclude"
    point_count: int          # 笔画采样点数
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) 原图坐标
    timestamp: str

    # SAM 精修结果
    sam_score: float          # SAM 置信度
    fg_pct_before: float | None = None  # 精修前该层 fg%
    fg_pct_after: float | None = None   # 精修后该层 fg%

    # 局部特征（该笔刷区域的图像/深度特征）
    local_features: dict[str, float] = field(default_factory=dict)


# ── 局部特征提取 ─────────────────────────────────────────────────────

def _compute_local_features(
    image: np.ndarray,
    depth_map: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> dict[str, float]:
    """计算笔刷覆盖区域的局部特征。

    Args:
        image: BGR 原图 (H, W, 3)。
        depth_map: float32 深度图 (H_depth, W_depth)，可能与 image 分辨率不同。
        bbox: (x1, y1, x2, y2) 原图坐标。

    Returns:
        dict of feature_name → float value。
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img, x2); y2 = min(h_img, y2)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return {"_empty": 1.0}

    features: dict[str, float] = {}

    # ── RGB 边缘强度 ──────────────────────────────
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    features["rgb_edge_mean"] = float(np.mean(edges > 0))

    # ── 纹理复杂度 ─────────────────────────────────
    # 局部窗口标准差的中位数
    gray_f = gray.astype(np.float32)
    kernel_size = 8
    mean = cv2.blur(gray_f, (kernel_size, kernel_size))
    sq_mean = cv2.blur(gray_f ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    features["texture_complexity"] = float(np.median(local_std)) / 128.0  # 归一化

    # ── 深度特征 ────────────────────────────────────
    if depth_map is not None:
        depth_h, depth_w = depth_map.shape[:2]
        scale_x = depth_w / w_img
        scale_y = depth_h / h_img
        dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
        dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)
        dx1, dy1 = max(0, dx1), max(0, dy1)
        dx2, dy2 = min(depth_w, dx2), min(depth_h, dy2)

        if dx2 - dx1 >= 2 and dy2 - dy1 >= 2:
            depth_crop = depth_map[dy1:dy2, dx1:dx2]
            gy, gx = np.gradient(depth_crop)
            g = np.sqrt(gy ** 2 + gx ** 2)
            features["depth_gradient_mean"] = float(np.mean(g))
            features["depth_median"] = float(np.median(depth_crop))
            features["depth_std"] = float(np.std(depth_crop))
        else:
            features["depth_gradient_mean"] = 0.0
            features["depth_median"] = 0.0
            features["depth_std"] = 0.0
    else:
        features["depth_gradient_mean"] = -1.0
        features["depth_median"] = -1.0
        features["depth_std"] = -1.0

    # ── 区域大小 ────────────────────────────────────
    features["component_size"] = float((x2 - x1) * (y2 - y1))

    return features


# ── 事件记录 API ─────────────────────────────────────────────────────

def record_event(
    image_hash: str,
    image_name: str,
    layer_index: int,
    brush_type: str,
    point_count: int,
    bbox: tuple[int, int, int, int],
    sam_score: float,
    fg_pct_before: float | None = None,
    fg_pct_after: float | None = None,
    image: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
) -> str:
    """记录一次笔刷事件并保存到文件。

    Args:
        image_hash: 图片 SHA256。
        image_name: 文件名。
        layer_index: 被修正的层。
        brush_type: "include" 或 "exclude"。
        point_count: 笔画采样点数。
        bbox: (x1, y1, x2, y2) 原图坐标。
        sam_score: SAM 精修置信度。
        fg_pct_before/after: 精修前后该层前景占比。
        image: 原图（可选，用于计算局部特征）。
        depth_map: 深度图（可选）。

    Returns:
        保存的事件文件路径。
    """
    # 计算局部特征
    local_features = {}
    if image is not None:
        local_features = _compute_local_features(image, depth_map, bbox)

    event = BrushEvent(
        image_hash=image_hash,
        image_name=image_name,
        layer_index=layer_index,
        brush_type=brush_type,
        point_count=point_count,
        bbox=bbox,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        sam_score=sam_score,
        fg_pct_before=fg_pct_before,
        fg_pct_after=fg_pct_after,
        local_features=local_features,
    )

    timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{image_hash[:12]}_{timestamp_slug}.json"
    filepath = EVENTS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(asdict(event), f, ensure_ascii=False, indent=2)

    return str(filepath)


def get_events(image_hash: str) -> list[dict]:
    """获取某张图片的所有笔刷事件。"""
    events: list[dict] = []
    prefix = image_hash[:12]
    for p in sorted(EVENTS_DIR.glob(f"{prefix}_*.json")):
        try:
            with open(p) as f:
                events.append(json.load(f))
        except Exception:
            continue
    return events


def get_event_count(image_hash: str) -> int:
    """获取某张图片的笔刷事件数量。"""
    prefix = image_hash[:12]
    return len(list(EVENTS_DIR.glob(f"{prefix}_*.json")))


def export_events(image_hash: str | None = None) -> list[dict]:
    """导出笔刷事件为训练格式。

    Args:
        image_hash: 指定图片，None = 全部。

    Returns:
        [{features: {...}, label: "needs_refine"|"no_refine"}, ...]
    """
    records: list[dict] = []

    if image_hash:
        event_files = list(EVENTS_DIR.glob(f"{image_hash[:12]}_*.json"))
    else:
        event_files = sorted(EVENTS_DIR.glob("*.json"))

    for p in event_files:
        try:
            with open(p) as f:
                event = json.load(f)
        except Exception:
            continue

        features = event.get("local_features", {})
        if not features or features.get("_empty"):
            continue

        # 特征 + 标签：开发者在此处涂抹过 → 需要精修
        records.append({
            "features": features,
            "label": "needs_refine",
            "image_hash": event.get("image_hash", ""),
            "layer_index": event.get("layer_index", 0),
            "brush_type": event.get("brush_type", ""),
        })

    return records


def get_statistics() -> dict:
    """汇总所有笔刷事件的统计信息。"""
    total = 0
    by_layer: dict[int, int] = {}
    by_type: dict[str, int] = {"include": 0, "exclude": 0}
    images_with_events: set[str] = set()

    for p in sorted(EVENTS_DIR.glob("*.json")):
        try:
            with open(p) as f:
                event = json.load(f)
        except Exception:
            continue
        total += 1
        by_layer[event.get("layer_index", 0)] = by_layer.get(event.get("layer_index", 0), 0) + 1
        bt = event.get("brush_type", "")
        if bt in by_type:
            by_type[bt] += 1
        images_with_events.add(event.get("image_hash", ""))

    return {
        "total_events": total,
        "unique_images": len(images_with_events),
        "by_layer": by_layer,
        "by_type": by_type,
    }
