#!/usr/bin/env python3
"""全流程 GPU 加速测试脚本。

用法:
    python dev_tools/scripts/test_gpu_full_pipeline.py [image_path]

默认使用 dev_tools/test_imgs/lorem_512.jpg。
"""

import sys
import os
import time
from pathlib import Path

# ── 路径注入 ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("MODEL_DIR", str(BACKEND_DIR / "models"))

import cv2
import numpy as np
from loguru import logger

# ── 测试图像 ────────────────────────────────────────────────────────
default_img = PROJECT_ROOT / "dev_tools" / "test_imgs" / "lorem_512.jpg"
img_path = sys.argv[1] if len(sys.argv) > 1 else str(default_img)

if not os.path.exists(img_path):
    logger.error("图像不存在: {}", img_path)
    sys.exit(1)

image = cv2.imread(img_path)
if image is None:
    logger.error("无法读取图像: {}", img_path)
    sys.exit(1)

h, w = image.shape[:2]
logger.info("=" * 60)
logger.info("测试图像: {} ({}×{})", Path(img_path).name, w, h)
logger.info("=" * 60)

# ── 1. 深度估计 ─────────────────────────────────────────────────
logger.info("[测试] 深度估计...")
from app.utils.depth_engine import estimate_depth

t0 = time.perf_counter()
depth = estimate_depth(image)
t_depth = (time.perf_counter() - t0) * 1000
logger.info(
    "[测试] 深度完成 | shape={} range=[{:.3f}, {:.3f}] time={:.0f}ms",
    depth.shape, float(depth.min()), float(depth.max()), t_depth,
)

# ── 2. SAM 自动分割 ────────────────────────────────────────────
logger.info("[测试] SAM 自动分割 (max_dim=512)...")
# 重置单例以避免缓存影响计时
import app.utils.sam_engine as se
se._sam_model = None
se._onnx_encoder_enabled = None
from app.utils.sam_engine import run_sam_automatic

t0 = time.perf_counter()
masks, region_map = run_sam_automatic(image, max_dim=512)
t_sam = time.perf_counter() - t0
logger.info(
    "[测试] SAM 完成 | {} masks | region_map shape={} | time={:.1f}s",
    len(masks), region_map.shape, t_sam,
)

# ── 摘要 ───────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("全流程摘要")
logger.info("=" * 60)
logger.info("  图像:           {} ({}×{})", Path(img_path).name, w, h)
logger.info("  深度估计:        {:.0f}ms", t_depth)
logger.info("  SAM 分割:        {:.1f}s ({} masks)", t_sam, len(masks))
logger.info("  总计:            {:.1f}s", t_depth / 1000 + t_sam)
logger.info("")

# 验证 ONNX DML 是否被使用
from app.utils.onnx_engine import is_dml_available
logger.info("  DML 可用:        {}", is_dml_available())
if is_dml_available():
    logger.info("  SAM encoder:     ONNX Runtime + DirectML")
    logger.info("  Depth engine:    ONNX Runtime + DirectML")
else:
    logger.info("  SAM encoder:     PyTorch CPU")
    logger.info("  Depth engine:    HuggingFace pipeline (CPU)")
