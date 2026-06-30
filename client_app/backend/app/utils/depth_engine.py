"""Depth-Anything-V2 单目深度估计引擎。

加载 Depth-Anything-V2-Small-hf 模型，输出逐像素归一化深度图。
纯函数 — 无 HTTP、无文件 I/O（模型下载除外）。

架构：ONNX Runtime + DirectML (GPU) 或 HuggingFace Transformers pipeline (CPU 回退)
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
_depth_onnx_session: Any = None
_device: str | None = None
_use_onnx: bool | None = None  # None = not yet probed

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


def _probe_onnx_dml_depth() -> bool:
    """检测 ONNX Runtime + DirectML 是否可用于深度推理。

    缓存结果，每个进程只探测一次。
    """
    global _use_onnx
    if _use_onnx is not None:
        return _use_onnx

    try:
        from app.utils.onnx_engine import is_dml_available, get_depth_session

        if not is_dml_available():
            from loguru import logger
            logger.info("[深度引擎] DirectML 不可用，使用 HuggingFace pipeline")
            _use_onnx = False
            return False

        get_depth_session()
        from loguru import logger
        logger.info("[深度引擎] ONNX Runtime + DirectML 可用，将启用 GPU 加速")
        _use_onnx = True
        return True
    except Exception as exc:
        from loguru import logger
        logger.warning("[深度引擎] ONNX 探测失败 ({}), 回退到 HuggingFace pipeline", exc)
        _use_onnx = False
        return False


def _get_depth_pipeline():
    """惰性加载 Depth-Anything-V2 pipeline 单例。

    首次调用时从 HuggingFace Hub 拉取模型（约 100 MiB），
    后续调用复用缓存。

    仅在 ONNX DML 不可用时使用。
    """
    global _depth_pipe, _device

    if _depth_pipe is not None:
        return _depth_pipe

    from loguru import logger
    import torch

    _device = _resolve_device()
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)

    logger.info(
        "[深度引擎] 加载 Depth-Anything-V2-Small (HuggingFace) | device={} threads={}",
        _device, cpu_count,
    )

    try:
        from transformers import pipeline

        _depth_pipe = pipeline(
            task="depth-estimation",
            model=_MODEL_ID,
            device=_device if _device != "mps" else -1,
        )
    except Exception as e:
        raise RuntimeError(
            f"[深度引擎] 模型加载失败 | model={_MODEL_ID} | {type(e).__name__}: {e}"
        ) from e

    logger.info("[深度引擎] 模型就绪 (HuggingFace)")
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

    # Step 3: 推理（ONNX DML 优先，HuggingFace pipeline 回退）
    if _probe_onnx_dml_depth():
        # ── ONNX Runtime + DirectML 路径 ─────────────────
        depth_raw = _infer_depth_onnx(pil_image, h, w)
    else:
        # ── HuggingFace pipeline 路径 ────────────────────
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


def _infer_depth_onnx(
    pil_image,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """使用 ONNX Runtime + DirectML 进行深度推理。

    Args:
        pil_image: PIL RGB 图像，可能已被缩放到 max_dim。
        orig_h: 原始图像高度。
        orig_w: 原始图像宽度。

    Returns:
        (H, W) float32 原始深度图（未归一化）。
    """
    from loguru import logger
    from app.utils.onnx_engine import run_depth_estimation
    from transformers import AutoImageProcessor

    # 1. HuggingFace 预处理器（与 ONNX 导出时用同一 processor）
    processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
    inputs = processor(images=pil_image, return_tensors="pt")

    # 2. ONNX Runtime DML 推理
    pixel_values_np = inputs["pixel_values"].numpy().astype(np.float32)
    ort_output = run_depth_estimation(pixel_values_np)  # shape (1, H_pred, W_pred)

    # 3. 提取深度图并缩放到输入 PIL 尺寸
    depth_pred = ort_output[0]  # (H_pred, W_pred)
    pil_w, pil_h = pil_image.size

    if depth_pred.shape[:2] != (pil_h, pil_w):
        depth_raw = cv2.resize(depth_pred, (pil_w, pil_h), interpolation=cv2.INTER_CUBIC)
    else:
        depth_raw = depth_pred

    logger.debug(
        "[深度引擎] ONNX DML 推理完成 | pred_shape={}→target=({},{})",
        depth_pred.shape, pil_h, pil_w,
    )
    return depth_raw.astype(np.float32)


def get_depth_info() -> dict:
    """返回深度引擎运行时信息。"""
    return {
        "device": _device or _resolve_device(),
        "model_loaded": _depth_pipe is not None,
        "model_id": _MODEL_ID,
    }
