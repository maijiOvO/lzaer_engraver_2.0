#!/usr/bin/env python3
"""导出 Depth-Anything-V2-Small 到 ONNX 格式。"""

import sys
from pathlib import Path

# 路径注入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import torch
import numpy as np
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
OUTPUT_PATH = BACKEND_DIR / "models" / "depth_anything_v2_small.onnx"

print(f"[ONNX导出] 加载模型: {MODEL_ID}")
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
model.eval()

# DPv2-Small 原生分辨率: 518×518
h, w = 518, 518
dummy = torch.randn(1, 3, h, w)

print(f"[ONNX导出] 导出 ONNX (input shape: 1×3×{h}×{w})")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    model,
    dummy,
    str(OUTPUT_PATH),
    input_names=["pixel_values"],
    output_names=["predicted_depth"],
    dynamic_axes={
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "predicted_depth": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=17,
    do_constant_folding=True,
)

print(f"[ONNX导出] 完成 → {OUTPUT_PATH}")

# ── 验证: ONNX Runtime CPU vs DML 推理耗时对比 ──
print("\n[验证] ONNX Runtime 推理对比...")

import onnxruntime as ort
import time

# 加载 ONNX 模型
sess_cpu = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
try:
    sess_dml = ort.InferenceSession(str(OUTPUT_PATH), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    dml_available = True
except Exception as e:
    print(f"[验证] DML 创建失败: {e}")
    dml_available = False

# 准备测试数据 (模拟 518×518 输入)
test_input = np.random.randn(1, 3, 518, 518).astype(np.float32)

# Warmup
print("[验证] Warmup...")
sess_cpu.run(None, {"pixel_values": test_input})
if dml_available:
    sess_dml.run(None, {"pixel_values": test_input})

# Benchmark CPU
print("[验证] CPU benchmark...")
t0 = time.perf_counter()
for _ in range(5):
    sess_cpu.run(None, {"pixel_values": test_input})
cpu_time = (time.perf_counter() - t0) / 5 * 1000
result_cpu = sess_cpu.run(None, {"pixel_values": test_input})[0]

# Benchmark DML
if dml_available:
    print("[验证] DML benchmark...")
    t0 = time.perf_counter()
    for _ in range(5):
        sess_dml.run(None, {"pixel_values": test_input})
    dml_time = (time.perf_counter() - t0) / 5 * 1000
    result_dml = sess_dml.run(None, {"pixel_values": test_input})[0]

    diff = np.abs(result_cpu - result_dml).max()
    print(f"\n{'='*50}")
    print(f"  CPU:  {cpu_time:.0f}ms")
    print(f"  DML:  {dml_time:.0f}ms")
    print(f"  加速比: {cpu_time/dml_time:.1f}x")
    print(f"  最大差异: {diff:.6f}")
    print(f"  {'✅ 精度通过' if diff < 1e-3 else '⚠️ 精度差异较大'}")
    print(f"{'='*50}")
else:
    print(f"\n  CPU:  {cpu_time:.0f}ms (DML 不可用)")