#!/usr/bin/env python3
"""导出 MobileSAM image_encoder (ViT-Tiny) 到 ONNX 格式。"""

import sys
import os
from pathlib import Path

# 路径注入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "dev_tools" / "scripts"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# 复用 sam_engine 中的模型加载逻辑
os.environ.setdefault("MODEL_DIR", str(BACKEND_DIR / "models"))
os.environ.setdefault("OUTPUT_DIR", str(BACKEND_DIR / "outputs"))

import torch
import numpy as np

MODEL_PATH = BACKEND_DIR / "models" / "mobile_sam.pt"
OUTPUT_PATH = BACKEND_DIR / "models" / "mobile_sam_encoder.onnx"

print(f"[ONNX导出] 加载 MobileSAM 模型: {MODEL_PATH}")

from mobile_sam import sam_model_registry

# 加载完整模型
model = sam_model_registry["vit_t"](checkpoint=str(MODEL_PATH))
model.eval()

# 提取 image_encoder
encoder = model.image_encoder

# MobileSAM ViT-Tiny 输入: (1, 3, 1024, 1024)
h, w = 1024, 1024
dummy = torch.randn(1, 3, h, w)

print(f"[ONNX导出] 导出 image_encoder ONNX (input: 1×3×{h}×{w})")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 需要先跑一次 forward 让 jit 预热
with torch.inference_mode():
    _ = encoder(dummy)

torch.onnx.export(
    encoder,
    dummy,
    str(OUTPUT_PATH),
    input_names=["image"],
    output_names=["image_embeddings"],
    dynamic_axes={
        "image": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)

print(f"[ONNX导出] 完成 → {OUTPUT_PATH}")

# ── 验证: ONNX Runtime CPU vs DML ──
print("\n[验证] ONNX Runtime 推理对比...")

import onnxruntime as ort
import time

sess_cpu = ort.InferenceSession(str(OUTPUT_PATH), providers=["CPUExecutionProvider"])
try:
    sess_dml = ort.InferenceSession(str(OUTPUT_PATH), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    dml_available = True
except Exception as e:
    print(f"[验证] DML 创建失败: {e}")
    dml_available = False

test_input = np.random.randn(1, 3, 1024, 1024).astype(np.float32)

# Warmup
print("[验证] Warmup...")
sess_cpu.run(None, {"image": test_input})
if dml_available:
    sess_dml.run(None, {"image": test_input})

# Benchmark CPU
print("[验证] CPU benchmark...")
t0 = time.perf_counter()
for _ in range(3):
    sess_cpu.run(None, {"image": test_input})
cpu_time = (time.perf_counter() - t0) / 3 * 1000
result_cpu = sess_cpu.run(None, {"image": test_input})[0]

# Benchmark DML
if dml_available:
    print("[验证] DML benchmark...")
    t0 = time.perf_counter()
    for _ in range(3):
        sess_dml.run(None, {"image": test_input})
    dml_time = (time.perf_counter() - t0) / 3 * 1000
    result_dml = sess_dml.run(None, {"image": test_input})[0]

    diff = np.abs(result_cpu - result_dml).max()
    print(f"\n{'='*50}")
    print(f"  MobileSAM ViT-Tiny image_encoder (1024×1024)")
    print(f"  CPU:  {cpu_time:.0f}ms")
    print(f"  DML:  {dml_time:.0f}ms")
    print(f"  加速比: {cpu_time/dml_time:.1f}x")
    print(f"  输出 shape: {result_cpu.shape}")
    print(f"  最大差异: {diff:.6f}")
    print(f"  {'✅ 精度通过' if diff < 1e-2 else '⚠️ 精度差异较大'}")
    print(f"{'='*50}")
else:
    print(f"\n  CPU:  {cpu_time:.0f}ms (DML 不可用)")
    print(f"  输出 shape: {result_cpu.shape}")