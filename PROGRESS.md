# 项目进度 (Progress Log)

> 最后更新：2026-07-01（导向滤波边缘吸附引擎重写 + 形态学伪影过滤 + 自适应blur + 笔刷精修集成）
> Git HEAD：cec7fe0

## 整体架构状态

| 层级 | 进度 | 说明 |
|------|------|------|
| 项目治理 | ✅ | Git 初始化、.gitignore、AI_RULES.md 10 大准则、API_CONTRACT.md |
| client_app/backend | 🟡 | 7 步管线全部有路由+服务+算法；SAM 分层架构 sam_driven 统一接管 2-10 层 |
| client_app/frontend | 🟡 | 组件骨架完成；参数面板已同步新架构 |
| dev_tools | 🟡 | canny/denoise 测试脚本就绪；**标定器全链路修复完成**（2026-06-17 晚） |
| Docker 环境 | ✅ | docker-compose.yml 保留；当前开发以本机运行为主（uvicorn + npm run dev），Docker 作为可选部署方案 |

## 管线 7 步实现状态

| 步骤 | API 路由 | Service | Utils（纯算法） | 前端组件 | dev_tools 测试 |
|------|----------|---------|----------------|----------|---------------|
| 0. Health | main.py ✅ | — | — | — | — |
| 1. Upload | upload.py ✅ | — | — | ImageUploader.tsx ✅ | — |
| 2-3. 深度分层 | segment.py ✅ | segmentation_service.py ✅ | depth_engine.py ✅ structural_segmentation.py ✅ sam_engine.py ✅ layer_frame.py ✅ | ControlPanel.tsx ✅ | **test_sam_segment.py ✅ (四模式)** |
| 4. Canny | canny.py ✅ | canny_service.py ✅ | canny_lineart.py ✅ | ControlPanel.tsx ✅ | test_canny.py ✅ |
| 5. Denoise | denoise.py ✅ | denoise_service.py ✅ | denoise.py ✅ | ControlPanel.tsx ✅ | test_denoise.py ✅ |
| 6. Connectivity | connectivity.py ✅ | connectivity_service.py ✅ | connectivity.py ✅ | — | test_pipeline.py（部分） |
| 7. SVG | svg.py ✅ | svg_service.py ✅ | svg_generator.py ✅ | — | test_pipeline.py（部分） |

### 联合调试结果（2026-06-12 全功能测试）

| 测试项 | 结果 |
|--------|------|
| 单层全管线 (upload→canny→denoise→connectivity→svg) | ✅ 通过 |
| 多层全管线 (segment→逐层canny/denoise/connectivity→multi-svg) | ✅ 通过 |
| 参数变化验证 (MD5不同/cache-busting/tolerance影响) | ✅ 通过 |
| 输入鲁棒性 (7项非法参数) | ✅ 零500 |
| 崩溃恢复 (stop→refused→start→正常) | ✅ 2s恢复 |
| 深度缓存 | ✅ 54s→5s |

**新发现**: 3 个问题写入 UNSOLVED_ISSUES.md（案例 16-18）

## 架构变更记录

### 2026-06-12 — SAM 分层架构升级：K-means → 深度引导
- **旧方案**: MobileSAM AutoMask 无差别碎片化 → K-means(位置+面积+亮度) 聚类
- **新方案**: Depth-Anything-V2 单目深度估计 → 等距量化 N 层 → 连通性校验+桥接修复(策略C) → 可选 SAM 逐层边界精修
- **语义完整性**: 深度估计保证同一物体不被拆散（建筑=建筑、树=树）
- **物理可支撑性**: 每层生成固定宽度外框，内容通过 Bresenham 桥接保证不掉落
- **新增文件**: `depth_engine.py`, `structural_segmentation.py`
- **修改文件**: `sam_engine.py`(+refine_mask), `segmentation_service.py`(重写), `requests.py`(SegmentParams), `ControlPanel.tsx`, `App.tsx`, `index.ts`
- **SegmentParams 变更**: 移除 `depth_mode`/`merge_sensitivity`/`min_layer_area_pct`，新增 `frame_width`/`min_island_area`
- **依赖新增**: `transformers` + `depth-anything/Depth-Anything-V2-Small-hf`

### 2026-07-02 — test_sam_segment.py 升级为三合一开发者效率工具
- **旧方案**: test_sam_segment.py（364 行）— 手动 CLI 传参，单图处理，无数据集管理
- **新方案**: test_sam_segment.py（1162 行）— scan / label / train / search 四模式
  - **scan 模式**（默认）: SHA256 注册表扫描 → 新图自动深度估计+特征提取+参数预测+分割 → 注册
  - **label 模式**: 交互式标定（Y/n/e/q），改参循环，深度缓存命中 <1s 重跑
  - **train 模式**: RandomForest 预测器训练（≥10 样本），OOB score + 特征重要性
  - **search 模式**: 接口就位（NotImplementedError + Google API 配置指南）
- **新增类/函数**: `ImageRegistry`（SHA256 去重+状态管理+JSON 容错备份）、`extract_features()`（12 维特征）、`train_predictor()` / `predict_params()`（RandomForest + 马氏距离异常检测）
- **新增文件**: `dev_tools/data/labeled.json`（注册表）、`dev_tools/data/layer_predictor.pkl`（训练产物，.gitignore）
- **不动**: client_app/、Docker、docker-compose
- **详见**: docs/开发者优化SAM处理脚本计划.md

### 2026-07-01 — 导向滤波(Guided Filter)边缘吸附引擎重写，替代 GrabCut
- **旧方案**: `_snap_mask_to_edges` 使用 GrabCut + trimap + ROI裁剪
- **新方案**: `cv2.ximgproc.createGuidedFilter` — O(N) 全图运行，像素级边界咬合
- **关键参数**: `radius=edge_band, eps=1e-5`，死死咬住埃菲尔铁塔级高频边缘
- **自适应 blur_size**: `cv2.distanceTransform` 检测最细结构厚度，clamp 模糊核防溶解
- **依赖变更**: `opencv-python-headless` → `opencv-contrib-python`
- **测试验证**: 巴黎铁塔层0 22,949→339px (-98.5%)，布达佩斯 erased 1→22
- **详见**: sam_engine.py `_snap_mask_to_edges`

### 2026-07-01 — 笔刷精修集成导向滤波 + 全图/局部统一管线
- **修复**: `labeler_server.py` `/api/brush-refine` 在 SAM predict 后插入 `_snap_mask_to_edges`
- **结果**: 全图分割 + 标定工具 + 笔刷精修三路统一使用同一导向滤波函数
- **详见**: labeler_server.py L687-691

### 2026-07-01 — 形态学腐蚀伪影过滤器替代 aspect_ratio 过滤
- **旧方案**: `aspect_ratio > 15 && min_dim <= 12` 仅过滤细长矩形
- **新方案**: 5×5 连通分量腐蚀测试 — 腐蚀后完全消失即抹除
- **效果**: 弧形光晕、曲线接缝全部捕获；主体对象 (>10000px²) 直接跳过
- **详见**: structural_segmentation.py Step 5

### 2026-06-12 — Bug 修复：callSvg useCallback 闭包过期（案例 13）

- **症状**: connectivity 重跑后 SVG 不会标记为 stale，stale 检测中 connectivity 维度失效
- **根因**: `callSvg` useCallback deps=`[]` 但直接读取 `connectivityResult` state → 闭包过期，永远为初始值 `null`
- **修复**: 新增 `connectivityResultRef`（第 132 行），第 413 行改为 `connectivityResultRef.current ? connectivityGenRef.current : null`
- **详见**: PAST_ISSUES.md 案例 13

### 2026-06-12 — Bug 修复：DenoiseParams / ConnectivityParams / SvgParams 缺失 layer_index
- **症状**: 三个端点在单层/多层模式下均抛出 AttributeError 500
- **根因**: Service 层已添加多层模式分支代码，对应的 Pydantic 请求模型漏加 `layer_index` 字段
- **修复**: `requests.py` 中三个模型各加 `layer_index: Optional[int] = Field(None, ...)`
- **测试**: Phase 1 (Pydantic 6 项) + Phase 2 (单层冒烟 4 项) + Phase 3 (多层逐层 4 项) 全部通过
- **详见**: PAST_ISSUES.md 案例 8

## 前端组件状态

| 组件 | 状态 | 功能 |
|------|------|------|
| App.tsx | ✅ | 管线编排、状态管理、cache-busting version 计数器（案例 13 闭包已修复） |
| ImageUploader.tsx | ✅ | 拖拽/点击上传、预览、重置管线 |
| ControlPanel.tsx | 🟡 | 模式选择、SAM/Canny/Denoise 参数滑块、500ms 防抖 |
| Canvas.tsx | 🟡 | 原图+叠加层渲染、缩放拖拽、cache-busting |
| MultiLayerCanvas.tsx | 🟡 | 多层模式图层切换、逐层处理路径渲染 |
| ProgressBar.tsx | ✅ | 步骤进度指示 |
| useZoomPan.ts | ✅ | 画布缩放/平移 hook |
| types/index.ts | ✅ | TypeScript 接口定义 |
| client.ts | ✅ | Axios 实例 + 全局错误拦截器 |

## 参考数据

| 目录 | 内容 | 用途 |
|------|------|------|
| references/single/ | 6 张剪纸 PNG | 降噪/连通性修复的品质参考 |
| references/multiple/ | 2 张纸雕 SVG (CorelDRAW) | SVG 生成的格式参考 |
| test_imgs/ | 100+ 张测试图 | 各阶段测试输入 |
| benchmarks/lineart_compare/ | 117 图对比结果 | canny_lineart 算法验证 |
| outputs/sam/n3_f50_i100_gf/ | 巴黎+迪拜+布达佩斯 导向滤波测试结果 | 边缘吸附效果对比 |

## 下一步

1. ~~集成测试：全管线联调（upload → SVG）~~ ✅ — 2026-06-12
2. ~~修复 UNSOLVED_ISSUES.md 案例 16（n_layers 上限校验）~~ ✅
3. ~~修复 UNSOLVED_ISSUES.md 案例 17（多层 SVG total_points）~~ ✅
4. ~~SAM 测试脚本升级为四模式效率工具~~ ✅ — 2026-07-02
5. ~~标定器全链路修复~~ ✅ — 2026-06-17 晚（案例 25-31）
6. ~~GrabCut → 导向滤波迁移~~ ✅ — 2026-07-01（全图+笔刷精修全套）
7. ~~形态学腐蚀伪影过滤 + 自适应 blur_size~~ ✅ — 2026-07-01
8. 标定 test_imgs/ 中 10+ 张图 → 训练预测器 → 验证预测精度
9. 前端多步骤预览（connectivity/svg 步骤的 Canvas 叠加）
10. 连通性修复的 dev_tools 独立测试脚本

## 架构变更时间线

| 日期 | 变更 | 详见 |
|------|------|------|
| 2026-06-17 晚 | boundary 模式彻底删除，sam_driven 统一接管 2-10 层 | PAST_ISSUES 案例 25-26 |
| 2026-06-17 晚 | suggest_n_layers 自动推断从通用函数移除 | PAST_ISSUES 案例 27 |
| 2026-06-17 晚 | 谷底检测改进（σ+浅谷过滤+硬兜底） | PAST_ISSUES 案例 28,31 |
| 2026-06-17 晚 | 外框向外延伸（不遮挡内容）+ 逐层视图对齐 | PAST_ISSUES 案例 29-30 |
| 2026-06-30 | 环境从 Linux 迁移至 Windows 11 原生；文档全面同步 | README, AI_RULES, API_CONTRACT |
| 2026-06-30 | 笔刷引擎 5 Bug 修复（坐标、事件泄漏、单点、SAM logits、BBox） | 本文 |
| 2026-06-30 | SAM crop_n_layers=0 消除网格裁切伪影 + 细碎分量后处理过滤 | sam_engine.py, structural_segmentation.py |
| 2026-06-30 | repair_layer_mask 桥接逻辑移除；build_sam_driven_layers skip_connectivity_repair | structural_segmentation.py |
| 2026-06-30 | fw 偏移复发问题文档化（PAST_ISSUES 新增复发型问题警告） | PAST_ISSUES.md |
| 2026-07-01 | 导向滤波(Guided Filter)重写边缘吸附引擎 | sam_engine.py, requirements.txt |
| 2026-07-01 | 笔刷精修集成导向滤波，全图+局部统一管线 | labeler_server.py |
| 2026-07-01 | 形态学腐蚀伪影过滤器替代 aspect_ratio 过滤 | structural_segmentation.py |
| 2026-07-01 | 自适应 distanceTransform blur_size 防细线溶解 | sam_engine.py |
| 2026-07-01 | 巴黎+迪拜+布达佩斯导向滤波测试完成 | dev_tools/outputs/sam/n3_f50_i100_gf/ |