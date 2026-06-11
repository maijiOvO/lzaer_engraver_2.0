# 激光雕刻管线 API 契约 (API Contract)

**【全局指令】本文件定义了 `client_app` 中前后端交互的唯一标准。**
AI 代理在开发 FastAPI 路由（`backend/app/api/`）、Pydantic 模型（`models/requests.py`, `models/responses.py`）以及前端 Axios 调用接口时，**必须绝对遵守本契约，严禁擅自修改字段名或新增未授权接口。**

---

## 1. 全局规约 (Global Rules)

- **运行环境**: 本项目完全基于 Linux 环境，不涉及 Windows 系统。
- **Base URL**: `http://localhost:8080/api`
- **请求格式**: 除文件上传（`multipart/form-data`）外，所有 POST 请求均使用 `application/json`。
- **数据流转**: 后端处理完的图像均保存在容器的 `/app/outputs/` 目录下，并向前端返回可访问的静态 URL（如 `/outputs/img_123_lineart.png`）。前端通过该 URL 渲染画布。

### 1.1 标准错误响应 (Standard Error Response)
配合全局 Exception Handler，所有 4xx/500 报错必须返回以下格式，前端 Axios 拦截器据此解析：
```json
{
  "detail": "错误类型简述",
  "error_msg": "完整的报错信息或 Traceback"
}
```

---

## 2. 核心业务接口契约 (Pipeline Endpoints)

管线执行必须严格遵循以下顺序：Upload -> Segment (可选) -> Lineart -> Denoise -> Connectivity -> SVG。

### 步骤 0: 连通性测试
- **接口**: `GET /health`
- **说明**: 用于检测 FastAPI 服务是否就绪。
- **响应**: 
```json
{ 
  "status": "ok", 
  "version": "2.0" 
}
```

---

### 步骤 1: 图像上传
- **接口**: `POST /upload`
- **说明**: 用户选择并上传原始图片。
- **请求体**: `multipart/form-data` (字段名: `file`)
- **响应 (UploadResponse)**:
```json
{
  "image_id": "uuid-string",
  "width": 1920,
  "height": 1080,
  "original_url": "/outputs/uuid-string_original.jpg"
}
```

---

### 步骤 2 & 3: SAM 图层分割 (可选)
- **接口**: `POST /pipeline/segment`
- **说明**: 仅当用户选择“多层模式”时调用。根据预测或手动参数生成多图层掩码。单层模式直接跳过此接口。
- **请求 (SegmentParams)**:
```json
{
  "image_id": "uuid-string",
  "n_layers": 3,
  "depth_mode": "balanced",
  "merge_sensitivity": 1.0,
  "min_layer_area_pct": 5
}
```
- **响应 (SegmentResponse)**:
```json
{
  "overlay_url": "/outputs/uuid-string_segmented.png",
  "layers": [
    {"layer_index": 0, "mask_url": "/outputs/..._mask_0.png"},
    {"layer_index": 1, "mask_url": "/outputs/..._mask_1.png"}
  ]
}
```

---

### 步骤 4: 线稿提取 (Canny LineArt)
- **接口**: `POST /pipeline/lineart`
- **说明**: 使用 `canny_lineart` 引擎（CLAHE 局部对比度增强 + Canny 边缘检测）提取黑白线稿骨架。纯 CPU 运算，无需 GPU/模型下载。
- **请求 (LineArtParams)**:
```json
{
  "image_id": "uuid-string",
  "layer_index": null,  
  "low": 50,
  "high": 150,
  "smooth_level": 0
}
```
- **响应 (PipelineStepResponse)**:
```json
{
  "result_url": "/outputs/uuid-string_lineart.png",
  "processing_time_ms": 1250
}
```

---

### 步骤 5: 物理降噪 (Noise Reduction)
- **接口**: `POST /pipeline/denoise`
- **说明**: 必须在连通性检查之前执行。过滤线稿中的孤立噪点和小连通域。
- **请求 (DenoiseParams)**:
```json
{
  "image_id": "uuid-string",
  "min_component_area": 4  
}
```
- **响应 (PipelineStepResponse)**:
```json
{
  "result_url": "/outputs/uuid-string_denoised.png",
  "processing_time_ms": 150
}
```

---

### 步骤 6: 连通性检查与修复 (Connectivity)
- **接口**: `POST /pipeline/connectivity`
- **说明**: 使用 Union-Find 和 Bresenham 算法弥合主体线段断裂点。
- **请求 (ConnectivityParams)**:
```json
{
  "image_id": "uuid-string",
  "gap_tolerance": 5  
}
```
- **响应 (ConnectivityResponse)**:
```json
{
  "result_url": "/outputs/uuid-string_connected.png",
  "bridges_built": 12,  
  "processing_time_ms": 320
}
```

---

### 步骤 7: SVG 生成
- **接口**: `POST /pipeline/svg`
- **说明**: 将最终二值化骨架图转换为 SVG 矢量图路径。
- **请求 (SvgParams)**:
```json
{
  "image_id": "uuid-string",
  "simplify_tolerance": 1.0  
}
```
- **响应 (SvgResponse)**:
```json
{
  "svg_url": "/outputs/uuid-string_final.svg",
  "total_paths": 45,
  "total_points": 1024,
  "processing_time_ms": 500
}
```