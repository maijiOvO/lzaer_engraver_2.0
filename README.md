# 激光雕刻图像处理工具 (Laser Engraver V2)

本项目是一个高阶的"图像转 SVG 激光雕刻/纸雕"工具。核心目标是将任意风景/建筑照片转化为严格 1px 线宽、无断裂、支持物理分层的 SVG 矢量图。

> **运行环境**：本项目完全基于 Linux 环境运行（Docker 容器 + 宿主机 Linux），不涉及 Windows 系统。

---

## 🏗️ 顶层架构设计 (一国两制)

为了彻底解决端口冲突和前后端污染问题，本项目在物理与逻辑上被严格切分为两个互不干扰的战区：

### 1. `client_app/` (面向用户的客户端)
- **技术栈**：React 19 + Vite (前端) / FastAPI + OpenCV (后端)
- **运行方式**：**必须通过 Docker Compose 运行**。完全隔离在容器网络中。
- **职责**：处理用户的图片上传，执行 7 步核心图像管线，返回 SVG。

### 2. `dev_tools/` (面向开发者的训练与调试工具)
- **技术栈**：纯 Python + OpenCV HighGUI + scikit-learn
- **运行方式**：**宿主机本地直接运行**，零 HTTP，无 Web 服务。
- **职责**：使用 OpenCV 窗口进行参数标注、训练机器学习模型预测参数、算法纯函数沙盒测试。通过修改 `sys.path` 单向调用 `client_app` 内的核心算法。

---

## ⚙️ 客户端核心处理管线 (Client Pipeline)

用户端的图像处理严格遵循以下单向流程：

1. **图像上传**：用户选择需要处理的图片。
2. **模式选择**：用户选择「单层模式」或「多层模式」（针对立体纸雕等需求）。
3. **深度引导结构分层 (Depth-Anything-V2 + SAM)**：
   - 若为多层模式，使用 Depth-Anything-V2 单目深度估计 → 等距量化为 N 层 → 连通性校验 + 桥接修复(策略C) → 可选 SAM 逐层边界精修。
   - 三大核心约束：语义完整性（深度估计保证同一物体不拆散）、物理可支撑性（外框+桥接，保证不掉落）、仅处理结构边界（层内纹理由 Canny 负责）。
   - 首次运行约 5-10s（深度估计 + 可选 SAM 精修），深度图缓存命中 <0.1s。
   - 支持 `sam_quality` 三档：draft（跳过 SAM 精修）、standard（SAM 精修）、fine（增强边缘）。
   - 单层模式跳过此步。
4. **线稿提取 (`canny_lineart`)**：使用 CLAHE 局部对比度增强 + Canny 边缘检测提取黑白线稿。纯 CPU 运算、零模型依赖，即开即用。
5. **物理降噪**：对提取出的线稿进行连通域面积过滤，剔除孤立噪点和无效短枝，保证画面纯净。
6. **连通性检查与修复**：使用 Union-Find 算法与 Bresenham 画线算法，自动寻找并弥合主体线段上的断裂点，保证激光雕刻不断线。
7. **SVG 路径生成**：将处理完毕的 1px 二值骨架图，转换为平滑的单层贝塞尔曲线 SVG 并输出。

*(注：步骤 5 必须在步骤 6 之前，以防止连通性算法将噪点误认为线段端点导致“蜘蛛网”现象。)*

---

## 🛠️ 开发者工具与测试 (Dev Tools)

`dev_tools/` 目录存放纯本地脚本，用于算法独立验证和效果对比，无需启动 Docker 容器。

### 算法测试脚本
| 脚本 | 用途 |
|------|------|
| `test_canny.py` | Canny 线稿提取单步验证 |
| `test_denoise.py` | 降噪效果验证 |
| `test_pipeline.py` | 全管线端到端测试 |
| `test_sam_segment.py` | SAM 分割 / 深度分层效果验证 |
| `algorithm_sandbox.py` | 新算法快速原型沙盒 |

### 参考学习资料
- `references/single/` — 6 张单层剪纸成品 PNG，作为降噪/连通性修复的品质参考
- `references/multiple/` — 2 张多层纸雕 SVG（CorelDRAW 原稿），作为 SVG 生成的格式参考

### 基准测试
- `benchmarks/lineart_compare/` — 117 图 canny_lineart 算法对比，含独立 HTML 可视化

> **旧架构遗留**：`layer_labeler.py` 和 `train_layer_params.py` 为旧版 MobileSAM + K-means 参数标注工具。自 2026-06-12 架构升级为深度引导后不再使用，保留于仓库中以备参考。

---

## 🚀 快速启动指南

### 启动用户客户端 (Web 应用)
```bash
cd client_app
docker-compose up -d