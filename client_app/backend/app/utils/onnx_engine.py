"""ONNX Runtime + DirectML 统一推理引擎。

为 MobileSAM image_encoder 和 Depth-Anything-V2 提供
基于 ONNX Runtime + DirectML Execution Provider 的 GPU 加速推理。

目标硬件: Intel Arc B370 (Battlemage, Ultra 5 338H 核显)
DirectML 后端: onnxruntime-directml==1.24.4
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

import onnxruntime as ort

# ── 模型路径 ────────────────────────────────────────────────────────
# 与 sam_engine.py / depth_engine.py 使用同一个 MODEL_DIR 约定
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")


def _resolve_model_path(filename: str) -> Path:
    """将模型文件名解析为绝对路径。

    优先检查环境变量 MODEL_DIR 下的路径，其次尝试相对于本文件的位置。
    """
    path = Path(MODEL_DIR) / filename
    if path.exists():
        return path

    # 回退：相对于本文件的 ../../models/
    fallback = Path(__file__).resolve().parents[2] / "models" / filename
    if fallback.exists():
        return fallback

    # 即使不存在也返回首选路径，session 创建时会报可读错误
    return path


def create_session(
    model_path: str,
    use_gpu: bool = True,
    graph_optimization_level: Optional[ort.GraphOptimizationLevel] = None,
) -> ort.InferenceSession:
    """创建 ONNX Runtime 推理会话。

    Args:
        model_path: ONNX 模型文件路径。
        use_gpu: 是否优先使用 DirectML (GPU)。设为 False 强制使用 CPU。
        graph_optimization_level: 图优化级别，默认 ORT_ENABLE_ALL。

    Returns:
        配置好的 ort.InferenceSession。

    Raises:
        FileNotFoundError: 模型文件不存在。
        RuntimeError: DML 会话创建失败且 use_gpu=True 时，回退到 CPU。
    """
    model_path = str(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

    if graph_optimization_level is None:
        graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = graph_optimization_level
    # 限制 GPU 内存使用，避免核显 VRAM 溢出
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_cpu_mem_arena = True

    if use_gpu:
        # DirectML 需要在 provider_options 中指定 device_id
        dml_opts = {"device_id": "0"}
        providers = [
            ("DmlExecutionProvider", dml_opts),
            "CPUExecutionProvider",
        ]
        try:
            session = ort.InferenceSession(
                model_path, sess_options=sess_opts, providers=providers
            )
            actual_provider = session.get_providers()[0]
            logger.info(
                "[ONNX] 创建 DML 会话成功 | provider={} | model={}",
                actual_provider,
                Path(model_path).name,
            )
            return session
        except Exception as exc:
            logger.warning(
                "[ONNX] DML 会话创建失败 ({}), 回退到 CPU | model={}",
                exc,
                Path(model_path).name,
            )
            # 回退到 CPU
            return _create_cpu_session(model_path, sess_opts)
    else:
        return _create_cpu_session(model_path, sess_opts)


def _create_cpu_session(
    model_path: str, sess_opts: ort.SessionOptions
) -> ort.InferenceSession:
    """创建纯 CPU 推理会话。"""
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info("[ONNX] 创建 CPU 会话 | model={}", Path(model_path).name)
    return session


# ── 单例缓存 ─────────────────────────────────────────────────────────
_encoder_session: Optional[ort.InferenceSession] = None
_depth_session: Optional[ort.InferenceSession] = None


def get_sam_encoder_session(use_gpu: bool = True) -> ort.InferenceSession:
    """获取 MobileSAM ViT-Tiny image_encoder 的 ONNX Runtime 会话（单例）。

    Args:
        use_gpu: 是否优先使用 DirectML GPU 加速。
    """
    global _encoder_session
    if _encoder_session is None:
        model_path = _resolve_model_path("mobile_sam_encoder.onnx")
        _encoder_session = create_session(str(model_path), use_gpu=use_gpu)
    return _encoder_session


def get_depth_session(use_gpu: bool = True) -> ort.InferenceSession:
    """获取 Depth-Anything-V2-Small 的 ONNX Runtime 会话（单例）。

    Args:
        use_gpu: 是否优先使用 DirectML GPU 加速。
    """
    global _depth_session
    if _depth_session is None:
        model_path = _resolve_model_path("depth_anything_v2_small.onnx")
        _depth_session = create_session(str(model_path), use_gpu=use_gpu)
    return _depth_session


def run_sam_encoder(image: np.ndarray) -> np.ndarray:
    """使用 ONNX Runtime 运行 MobileSAM image_encoder。

    Args:
        image: shape (1, 3, 1024, 1024) float32 numpy array。

    Returns:
        image_embeddings: shape (1, 256, 64, 64) float32 numpy array。
    """
    session = get_sam_encoder_session()
    ort_inputs = {"image": image}
    ort_outputs = session.run(None, ort_inputs)
    return ort_outputs[0]


def run_depth_estimation(pixel_values: np.ndarray) -> np.ndarray:
    """使用 ONNX Runtime 运行 Depth-Anything-V2。

    Args:
        pixel_values: 经过 HuggingFace AutoImageProcessor 预处理后的 tensor，
                      转成 numpy，shape (1, 3, H, W) float32。

    Returns:
        predicted_depth: shape (1, H, W) float32 numpy array。
    """
    session = get_depth_session()
    ort_inputs = {"pixel_values": pixel_values}
    ort_outputs = session.run(None, ort_inputs)
    return ort_outputs[0]


def is_dml_available() -> bool:
    """检测 DirectML Execution Provider 是否可用。"""
    try:
        available = ort.get_available_providers()
        return "DmlExecutionProvider" in available
    except Exception:
        return False


def reset_sessions() -> None:
    """重置所有单例会话（用于测试或热重载）。"""
    global _encoder_session, _depth_session
    _encoder_session = None
    _depth_session = None
    logger.info("[ONNX] 所有会话已重置")