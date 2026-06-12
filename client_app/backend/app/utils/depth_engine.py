"""Depth-Anything-V2 单目深度估计引擎。

加载 Depth-Anything-V2-Small-hf 模型，输出逐像素归一化深度图。
纯函数 — 无 HTTP、无文件 I/O（模型下载除外）。

架构：HuggingFace Transformers pipeline
模型：depth-anything/Depth-Anything-V2-Small-hf
"""

from __future__ import annotations

import os
import time
from typing import Any

import cv2
import numpy as np

# ── 模型单例 ────────────────────────────────────────────────────────
_depth_pipe: Any = None
_device: str | None = None

# 首选设备：与 SAM 保持一致
_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def _resolve_device() -> str:
    """解析最优 torch 设备。"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_depth_pipeline():
    """惰性加载 Depth-Anything-V2 pipeline 单例。

    首次调用时从 HuggingFace Hub 拉取模型（约 100 MiB），
    后续调用复用缓存。
    """
    global _depth_pipe, _device

    if _depth_pipe is not None:
        return _depth_pipe

    from loguru import logger
    import torch
    from PIL import Image

    _device = _resolve_device()
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)

    logger.info(
        "[深度引擎] 加载 Depth-Anything-V2-Small | device={} threads={}",
        _device, cpu_count,
    )

    # 使用 transformers pipeline（自动处理预处理 / 推理 / 后处理）
    try:
        from transformers import pipeline

        _depth_pipe = pipeline(
            task="depth-estimation",
            model=_MODEL_ID,
            device=_device if _device != "mps" else -1,
            # 模型默认下载到 HF_HOME 缓存目录
        )
    except Exception as e:
        raise RuntimeError(
            f"[深度引擎] 模型加载失败 | model={_MODEL_ID} | {type(e).__name__}: {e}"
        ) from e

    logger.info("[深度引擎] 模型就绪")
    return _depth_pipe


def estimate_depth(
    image: np.ndarray,
    max_dim: int | None = None,
) -> np.ndarray:
    """对 BGR 图像逐像素估计深度。

    Args:
        image: BGR uint8 numpy 数组 (H, W, 3)。
        max_dim: 推理前缩放的最大边长。None = 使用模型原生分辨率
                 （Small 模型轻量，一般不需缩放）。

    Returns:
        (H, W) float32 深度图，值域 [0, 1]。
        0 = 最近（前景），1 = 最远（背景）。
    """
    from loguru import logger
    from PIL import Image

    t0 = time.perf_counter()
    h, w = image.shape[:2]
    logger.debug("[深度引擎] 输入 | shape=({},{},{})", h, w, image.shape[2])

    # Step 1: BGR → RGB → PIL
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Step 2: 可选缩放
    if max_dim is not None and max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
        logger.debug("[深度引擎] 缩放 | {}→{}x{}", image.shape[:2][::-1], new_w, new_h)

    # Step 3: 推理
    pipe = _get_depth_pipeline()
    try:
        import torch
        with torch.inference_mode():
            result = pipe(pil_image)
    except Exception as e:
        raise RuntimeError(
            f"[深度引擎] 推理失败 | {type(e).__name__}: {e}"
        ) from e

    # Step 4: 提取深度图 → numpy → 缩放到原图分辨率
    depth_pil: Image.Image = result["depth"]  # PIL Image, 灰度
    depth_raw = np.array(depth_pil, dtype=np.float32)

    if depth_raw.shape[:2] != (h, w):
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_CUBIC)

    # Step 5: 归一化到 [0, 1]
    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max - d_min > 1e-8:
        depth_norm = (depth_raw - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_raw)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[深度引擎] 完成 | shape=({},{}) range=[{:.3f},{:.3f}] time={:.0f}ms",
        h, w, float(d_min), float(d_max), elapsed,
    )
    return depth_norm.astype(np.float32)


def get_depth_info() -> dict:
    """返回深度引擎运行时信息。"""
    return {
        "device": _device or _resolve_device(),
        "model_loaded": _depth_pipe is not None,
        "model_id": _MODEL_ID,
    }
