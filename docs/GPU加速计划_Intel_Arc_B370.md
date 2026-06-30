# GPU 加速计划 — Intel Arc B370

> 版本: 1.0  
> 日期: 2026-06-19  
> 目标: 将 dev_tools/labeler 的 SAM + Depth 推理从 CPU 迁移到 Intel Arc B370 GPU

---

## 一、环境确认

| 项目 | 详情 |
|------|------|
| **操作系统** | Windows 11 原生 |
| **Python** | 3.13.9 (Anaconda) |
| **当前 PyTorch** | 2.8.0+cpu |
| **GPU** | Intel Arc B370 (Battlemage, Ultra 5 338H 核显) |
| **GPU 检测** | ✅ `onnxruntime-directml` 已安装，`DmlExecutionProvider` 可用 |
| **加速后端** | ONNX Runtime + DirectML Execution Provider |
| **标定器入口** | `python dev_tools/labeler/labeler_server.py` |
| **client_app 入口** | `uvicorn app.main:app` (本机) |

---

## 二、瓶颈分析

### 2.1 耗时最大的步骤

| 步骤 | 引擎 | 函数 | 运行设备 | 占总耗时 |
|------|------|------|----------|---------|
| **SAM 自动分割** | `sam_engine.py` | `run_sam_automatic()` | CPU | ~60% (~170s) |
| **深度估计** | `depth_engine.py` | `estimate_depth()` | CPU | ~30% (~85s) |
| 深度归属 + 连通修复 | `structural_segmentation.py` | `build_sam_driven_layers()` | numpy CPU | ~5% |
| 蒙版后处理 | `sam_engine.py` | 上采样 + 边缘精修 | numpy CPU | ~5% |

### 2.2 根因

1. `_resolve_device()` 只检测 `cuda → mps → cpu`，不支持 DirectML
2. PyTorch 安装的是 `2.8.0+cpu` 版本，完全无法使用 GPU
3. MobileSAM 的 ViT-Tiny image_encoder 和 mask_decoder 都在 CPU 上跑

---

## 三、加速方案：ONNX Runtime + DirectML

### 3.1 环境制约与方案选择

**实际环境**：Python 3.13.9 — `torch-directml` 暂无 Python 3.13 的 wheel。
**已验证可用**：`onnxruntime-directml==1.24.4`（`DmlExecutionProvider` 已检测到 Intel Arc B370）

| 因素 | torch-directml | ONNX Runtime DirectML | Intel IPEX + XPU |
|------|---------------|-----------------------|-------------------|
| **Python 3.13 支持** | ❌ 不可用 | ✅ 已验证 | ⚠️ 未知 |
| **安装复杂度** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **代码改动量** | ~50 行 | ~300 行 (需 ONNX 导出) | ~150 行 |
| **模型需导出** | ❌ 不需要 | ✅ 需要 (一次性) | ❌ 不需要 |
| **预期加速比** | 3-4x | 3-5x | 4-6x |
| **当前可实施** | ❌ | ✅ | ⚠️ 待验证 |

**结论**：**ONNX Runtime DirectML** 是当前唯一已验证可行的方案。

### 3.2 备选：降级 Python 到 3.12

如果 ONNX Runtime 方案在 SAM mask decoder（含 prompt encoder）上遇到兼容性问题，
可考虑创建 Python 3.12 conda 环境以使用 `torch-directml`：

```bash
conda create -n lzaer312 python=3.12
conda activate lzaer312
pip install torch-directml
```

### 3.3 ONNX Runtime DirectML 工作原理

```
MobileSAM PyTorch 模型
    ↓ torch.onnx.export() (一次性导出)
ONNX 模型文件 (.onnx)
    ↓ onnxruntime.InferenceSession
onnxruntime-directml (DmlExecutionProvider)
    ↓
DirectML (Microsoft GPU 加速库)
    ↓
Direct3D 12 Driver
    ↓
Intel Arc B370 GPU
```

---

## 四、实施步骤

### Phase 0: 环境验证 (已完成 ✅)

```bash
pip install onnxruntime-directml  # ✅ 1.24.4, DmlExecutionProvider 可用
```

### Phase 1: 模型导出 (预计 2h)

**目标**: 将 MobileSAM 和 Depth-Anything-V2 导出为 ONNX 格式。

#### 1.1 导出 MobileSAM image_encoder (ViT-Tiny)

```python
# dev_tools/scripts/export_sam_onnx.py
import torch
from mobile_sam import sam_model_registry

model = sam_model_registry["vit_t"](checkpoint="client_app/backend/models/mobile_sam.pt")
model.eval()
encoder = model.image_encoder

# ViT 输入: (1, 3, 1024, 1024)
dummy = torch.randn(1, 3, 1024, 1024)

torch.onnx.export(
    encoder, dummy,
    "client_app/backend/models/mobile_sam_encoder.onnx",
    input_names=["image"],
    output_names=["image_embeddings"],
    dynamic_axes={"image": {0: "batch"}},
    opset_version=17,
)
```

#### 1.2 导出 Depth-Anything-V2-Small

```python
# dev_tools/scripts/export_depth_onnx.py
from transformers import AutoModelForDepthEstimation
import torch

model_id = "depth-anything/Depth-Anything-V2-Small-hf"
model = AutoModelForDepthEstimation.from_pretrained(model_id)
model.eval()

dummy = torch.randn(1, 3, 518, 518)

torch.onnx.export(
    model, dummy,
    "client_app/backend/models/depth_anything_v2_small.onnx",
    input_names=["pixel_values"],
    output_names=["predicted_depth"],
    opset_version=17,
    dynamic_axes={"pixel_values": {0: "batch"}},
)
```

### Phase 2: ONNX 推理引擎 (预计 3h)

**新建文件**: `client_app/backend/app/utils/onnx_engine.py`

提供 ORT + DML 的统一推理接口：

```python
"""ONNX Runtime + DirectML 推理引擎。"""
import onnxruntime as ort

def create_session(model_path: str, use_gpu: bool = True) -> ort.InferenceSession:
    """创建 ONNX Runtime 推理会话。"""
    providers = (
        ["DmlExecutionProvider", "CPUExecutionProvider"] if use_gpu
        else ["CPUExecutionProvider"]
    )
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

# 单例
_encoder_session = None
_depth_session = None

def get_sam_encoder_session() -> ort.InferenceSession:
    global _encoder_session
    if _encoder_session is None:
        _encoder_session = create_session("app/models/mobile_sam_encoder.onnx")
    return _encoder_session

def get_depth_session() -> ort.InferenceSession:
    global _depth_session
    if _depth_session is None:
        _depth_session = create_session("app/models/depth_anything_v2_small.onnx")
    return _depth_session
```

### Phase 3: 修改 sam_engine.py (预计 3h)

**核心改动**：`run_sam_automatic()` 中，image_encoder 部分改为 ONNX Runtime DirectML 推理。

```python
def run_sam_automatic(image, max_dim=None, cache_path=None):
    # ... 预处理不变 ...
    
    # ── 替换: generator.generate(sam_rgb) ──
    # 原 PyTorch 路径:
    #   with torch.inference_mode():
    #       masks = generator.generate(sam_rgb)
    
    # 新 ONNX + MobileSAM 混合路径:
    # 1. image_encoder → ONNX Runtime DML (GPU 加速)
    # 2. mask_decoder + prompt_encoder → 保持 PyTorch CPU
    #    (AMG 内部循环调用 mask_decoder，其中含复杂的 prompt encoding + 
    #     mask prediction + IoU prediction，导出 ONNX 难度大)
    
    from app.utils.onnx_engine import get_sam_encoder_session
    import torch
    
    # 预处理：BGR→RGB, resize, normalize, to tensor
    sam_rgb = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
    # MobileSAM 期望 1024×1024 输入
    input_tensor = preprocess_for_sam(sam_rgb)  # → (1, 3, 1024, 1024) float32
    
    # ONNX Runtime DML 推理 image_encoder
    encoder_session = get_sam_encoder_session()
    ort_inputs = {"image": input_tensor.numpy()}
    ort_outputs = encoder_session.run(None, ort_inputs)
    image_embeddings = torch.from_numpy(ort_outputs[0])
    
    # 将 image_embeddings 注入 MobileSAM 的 generator
    # SamAutomaticMaskGenerator 内部会调用 predictor.set_image()
    # 需要 hack: 预先设置 image_embeddings 然后跳过 set_image 中的 encoder 部分
    generator = _create_generator()
    generator.predictor.is_image_set = True
    generator.predictor.features = image_embeddings
    generator.predictor.original_size = sam_rgb.shape[:2]
    generator.predictor.input_size = (1024, 1024)
    
    with torch.inference_mode():
        masks = generator.generate(sam_rgb)  # image_encoder 被跳过
    
    # ... 后续处理不变 ...
```

**关键洞察**：SAM 推理耗时分布为：
- image_encoder (ViT-Tiny): ~60% 的推理时间 → **ONNX DML 加速**
- mask_decoder (轻量): ~40% 的推理时间 → 保持在 PyTorch CPU
- 这样改动最小，但能获得大部分加速收益

### Phase 4: 修改 depth_engine.py (预计 2h)

```python
def estimate_depth(image, max_dim=None):
    # ... 预处理不变 ...
    
    # ── 替换 HuggingFace pipeline ──
    from app.utils.onnx_engine import get_depth_session
    from transformers import AutoImageProcessor
    
    # 1. HuggingFace 预处理器 (仍在 CPU)
    processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
    inputs = processor(images=pil_image, return_tensors="pt")
    
    # 2. ONNX Runtime DML 推理
    session = get_depth_session()
    ort_inputs = {"pixel_values": inputs["pixel_values"].numpy()}
    ort_outputs = session.run(None, ort_inputs)
    depth_raw = ort_outputs[0][0, 0]  # (H, W) float32
    
    # 3. 后处理 (numpy, CPU)
    # ... 缩放、归一化不变 ...
```

### Phase 5: 修改 labeler_server.py (预计 30 min)

笔刷精修 API `api_brush_refine()`：
- SAM predictor 的 image_encoder 部分使用 ONNX DML
- SamPredictor.set_image() 需要 hack 跳过 encoder

### Phase 6: 更新 requirements.txt

```diff
+ # GPU 加速 — Intel Arc B370 via ONNX Runtime DirectML
+ onnxruntime-directml>=1.24.0
+ onnx>=1.16.0
```

### Phase 7: 测试验证 (预计 1h)

```bash
# 1. 验证 ONNX Runtime DML
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# → ['DmlExecutionProvider', 'CPUExecutionProvider']

# 2. 导出模型
python dev_tools/scripts/export_sam_onnx.py
python dev_tools/scripts/export_depth_onnx.py

# 3. 端到端测试
python dev_tools/labeler/labeler_server.py
# → 访问 http://localhost:8090，对比 CPU vs DML 耗时
```

---

## 五、实测验证 (最小可行性验证 — 2026-06-19)

### 5.1 模型导出 + 推理对比

| 模型 | 输入尺寸 | CPU | DML | 加速比 | 精度差异 |
|------|---------|-----|-----|--------|---------|
| **Depth-Anything-V2-Small** | 1×3×518×518 | 198ms | 79ms | **2.5x** | 7e-6 ✅ |
| **MobileSAM ViT-Tiny (image_encoder)** | 1×3×1024×1024 | 310ms | 70ms | **4.4x** | 1e-6 ✅ |

### 5.2 端到端预期 (基于实测推算)

| 操作 | CPU (当前) | ONNX DML (实测推算) | 加速比 |
|------|-----------|---------------------|--------|
| SAM image_encoder (16MP → 1200px) | ~100s | ~23s | 4.4x |
| SAM mask_decoder (AMG loop) | ~70s | ~70s | 1x (保持 CPU) |
| 深度估计 (full pipeline) | ~8s | ~3s | 2.5x |
| **首次完整推理 (16MP)** | **~260s** | **~100s** | **2.6x** |

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| MobileSAM ViT-Tiny ONNX 导出失败 (动态 shape / 自定义 op) | 中 | 高 | 回退方案：用 Python 3.12 conda env + torch-directml |
| ONNX Runtime DML 与 ViT 的 LayerNorm/Attention 不兼容 | 低 | 中 | 使用 opset_version=17，DML 已支持标准 Transformer ops |
| SamPredictor 注入 image_embeddings 后 `generate()` 行为异常 | 中 | 高 | 手写 AMG 循环替代 `SamAutomaticMaskGenerator` |
| Depth-Anything-V2 ONNX 导出失败 | 低 | 中 | DPv2 结构简单 (ViT encoder)，ONNX 导出通常顺利 |
| mask_decoder 仍在 CPU 上，端到端加速有限 | 低 | 低 | image_encoder 占 60% 推理时间，加速它已收益显著 |
| 核显 VRAM 不足 (共享系统内存) | 低 | 低 | 单 batch 推理，显存使用 < 2GB |

---

## 七、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `client_app/backend/app/utils/device_manager.py` | **新建** | 统一设备管理器 |
| `client_app/backend/app/utils/sam_engine.py` | 修改 | 替换 `_resolve_device()` → `get_device()` |
| `client_app/backend/app/utils/depth_engine.py` | 修改 | 同上，适配 DML |
| `dev_tools/labeler/labeler_server.py` | 修改 | 笔刷 API 使用统一设备 |
| `client_app/backend/requirements.txt` | 修改 | 添加 `torch-directml` |
| `docs/开发者优化SAM处理脚本计划.md` | 修改 | 更新开发环境章节 ✅ |
| `docs/GPU加速计划_Intel_Arc_B370.md` | **新建** | 本文档 ✅ |