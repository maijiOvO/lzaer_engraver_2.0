# 标定工具全面重构计划

> 版本: 2.0  
> 日期: 2026-06-17  
> 关联: `dev_tools/labeler/`（标定器全栈）、`dev_tools/scripts/test_sam_segment.py`（CLI 工具）

---

## 零、背景与问题诊断

### 0.1 业务场景

2.5D 纸雕灯：纸张层叠雕刻，亚克力仅作外边框。核心需求：

1. **物理可支撑** — 每层必须与外框连通（纸张虽轻但不能掉落）
2. **视觉层次正确** — 分层需符合人对场景深度的认知
3. **边缘干净可切割** — 切割线应沿建筑/物体轮廓，不应穿过对象中部

### 0.2 当前算法诊断（2026-06-17 测试结论）

在上海.jpg（5472×3078）上的测试结果：

| 算法 | 层分布 | 宏观结构 | 细节边缘 | 结论 |
|------|--------|---------|---------|------|
| 客户端等距量化 (raw) | 62.6/25.9/11.5% | ✅ 正确 | ❌ 粗锯齿 | **当前最佳基础** |
| 客户端等距量化 (repaired) | 含外框 | ✅ | ❌ + 外框空白区 | 连通性修复，非质量问题 |
| SLIC + 深度投票 | 59.5/2.4/42.9% | ❌ 层1几乎空 | ❌ 毛刺多 | 分位数对左偏分布失效 |

**根因分析**：

- 深度模型（Depth-Anything-V2）对宏观空间结构的判断**正确**——能区分建筑群和天空
- 深度图的空间分辨率（518px 高）决定了它**无法提供精确的对象边界**
- 深度值分布严重左偏（中位数=0.0000），导致 SLIC 的分位数分层失效
- 关键矛盾：深度模型负责"哪片区域属于第几层"（正确），但当前管线让深度模型同时负责"边界画在哪"（错误）

### 0.3 新方向

```
深度模型 → "这片属于第 N 层"（层级归属）  ← 深度模型做这个，正确
    +
SAM 模型 → "边界在这条线上"（精确边缘）    ← SAM 做这个，原图精度
    ↓
正确的分层蒙版
```

---

## 一、目标

将标定工具从"自动化分割 + 被动确认"升级为"智能分割 + 人工精修 + 机器学习闭环"的完整开发者工具链。

**三层递进目标**：

1. **分割质量提升** — 深度等距量化提供宏观分层，SAM 在原图精度上精修层间边界
2. **快速人工干预** — 笔刷式局部修正，像 PS 一样涂抹选中的/排除的区域
3. **ML 闭环** — 从人工修正中学习：笔刷信号 → 局部特征 → 失败模式聚类 → 精修预测器

---

## 二、架构变更总览

### 2.1 从脚本到全栈标定平台

```
旧架构：
  test_sam_segment.py (CLI)
    └─ label 模式: 交互式参数调优

新架构：
  dev_tools/labeler/
  ├── labeler_server.py       ← Web 标定器（FastAPI，端口 8090）
  │     ├── run_segmentation()    ← 改造：支持精修模式
  │     ├── api_brush_refine()    ← 新增：笔刷式 SAM 局部精修
  │     ├── api_save()            ← 改造：记录笔刷事件
  │     └── api_brush_events()    ← 新增：笔刷事件查询/导出
  │
  ├── boundary_refine.py       ← 新增：边界带提取 + SAM 精修引擎
  ├── brush_recorder.py        ← 新增：笔刷事件记录模块
  ├── slic_segmentation.py     ← 保留：SLIC 算法（备用对比）
  ├── score_engine.py          ← 保留：5 维自动评分
  │
  ├── static/                  ← 前端（纯 HTML/CSS/JS，零框架）
  │     ├── index.html          ← 改造：新增笔刷工具栏 + 逐层审核模式
  │     ├── brush_tool.js       ← 新增：Canvas 笔刷引擎
  │     └── ...
  │
  └── dev_tools/scripts/
      └── test_sam_segment.py   ← 保留 CLI，对接新引擎
```

### 2.2 管线变更

```
旧管线（等距量化 + 连通修复 + 可选 SAM 逐层精修）：
  深度图 → 等距量化 → repair_layer_mask → SAM 逐层精修（事后补救）
                                                     ↑ 已经切坏了才修

新管线（深度分层 + 层间边界带 SAM 精修）：
  深度图 → 等距量化 raw（不做连通修复）→ 提取边界带 → SAM 框选精修 → 回填
                 ↑ 宏观分层          ↑ 只在这 SAM 运行
```

---

## 三、新增模块详细设计

### 3.1 `boundary_refine.py` — 边界精修引擎

**输入**：
- `image` (np.ndarray): BGR 原图
- `layer_masks_raw` (list[np.ndarray]): 等距量化的 N 层原始蒙版（不含外框、不含连通修复）
- `depth_map` (np.ndarray): 深度图

**流程**：

```
Step 1: 提取层间边界带
  对相邻层 i 和 i+1：
    zone = dilate(layer_i, 5px) ∩ dilate(layer_{i+1}, 5px)
    ← 这是深度图认为"过渡区域"的地带

Step 2: 在边界带内找到连通分量
  cv2.connectedComponentsWithStats(zone)
  每个分量 = 一个独立的"需要 SAM 判断归属"的局部区域

Step 3: SAM 框选精修
  对每个连通分量：
    bbox = 分量外接矩形 + 10px padding
    image_crop = 原图[bbox]
    SAM predictor.set_image(image_crop)
    使用 box prompt（分量轮廓的 bbox 在原图坐标）→ SAM 输出精确蒙版
  
Step 4: 深度投票决定归属
  对 SAM 输出的每个精修片段：
    取片段在原图中的对应区域
    计算深度中位数
    归属到最近的等距量化层级

Step 5: 回填到层蒙版
  将片段分配回对应层
  未覆盖的像素保持等距量化的结果
```

**输出**：
- `refined_masks`: N 个精修后的层蒙版（不含外框）
- `frame_mask`: 外框蒙版
- `stats`: 每层统计 + 精修片段数

**关键参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `band_width` | 5 | 边界带膨胀宽度（像素） |
| `min_component_area` | 100 | 跳过过小的边界带分量 |
| `sam_box_padding` | 10 | SAM 框选 padding |
| `enable` | True | 是否启用精修（draft 模式时 False） |

### 3.2 `brush_recorder.py` — 笔刷事件记录模块

**数据结构**：

```python
@dataclass
class BrushEvent:
    """一次笔刷修正事件"""
    image_hash: str           # 图片 SHA256
    image_name: str           # 文件名
    layer_index: int          # 当前编辑的层
    brush_type: str           # "include" (纳入) | "exclude" (排除)
    stroke_points: list       # [(x, y), ...] 笔刷轨迹坐标
    bbox: tuple               # (x1, y1, x2, y2) 笔刷覆盖区域
    timestamp: str
    
    # 该区域的局部特征（实时计算）
    local_features: dict = None  # {depth_gradient, rgb_edge, texture, ...}
```

**存储**：`dev_tools/data/brush_events/{image_hash}_{timestamp}.json`

**功能**：

| 函数 | 作用 |
|------|------|
| `record_event(event: BrushEvent)` | 记录一次笔刷事件 |
| `get_events(image_hash: str)` | 获取某图所有笔刷事件 |
| `export_events()` | 导出为训练格式 `[{features, label}, ...]` |
| `cluster_failure_modes()` | 对积累的事件做失败模式聚类 |

**局部特征计算**（对笔刷覆盖区域实时计算）：

| 特征 | 含义 | ML 用途 |
|------|------|--------|
| `depth_gradient_mean` | 区域内深度梯度均值 | 判断"深度模型是否漏了边界" |
| `rgb_edge_mean` | 区域内 Canny 边缘强度均值 | 判断"RGB 边缘是否明显但未被使用" |
| `texture_complexity` | 局部纹理方差 | 判断"是否复杂纹理干扰了分割" |
| `layer_adjacency` | 该区域邻接几个层 | 判断"跨层冲突严重程度" |
| `component_size` | 区域面积 | 判断"是碎片问题还是大边界问题" |
| `depth_median` | 区域内深度中位数 | 判断"深度归属是否错误" |

---

## 四、前端交互设计

### 4.1 新增「逐层审核 + 笔刷精修」模式

```
┌──────────────────────────────────────────────────────────┐
│  [原图] [彩色叠加] [逐层审核 🔍]                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────────────────────────────────┐              │
│   │                                      │              │
│   │    原图 + 当前层蒙版 (半透明叠加)      │              │
│   │    Canvas 支持缩放/拖拽/双击复位       │              │
│   │                                      │              │
│   └──────────────────────────────────────┘              │
│                                                          │
│  图层选择: [层0 ●] [层1 ○] [层2 ○]                        │
│                                                          │
│  工具栏:                                                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 🟢 纳入  │  🔴 排除  │  笔刷大小 [────○────] 20px   │  │
│  │ [应用SAM] │  [撤销]   │  [重置本层]                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  快捷键: Ctrl+Z 撤销  |  [ / ] 调笔刷大小                  │
│                                                          │
│  状态栏: 层0 修正 3 处 | 上次 SAM 精修 +12px²              │
└──────────────────────────────────────────────────────────┘
```

### 4.2 笔刷操作流程

```
1. 开发者选择「逐层审核」，选中层 1
2. 发现某建筑边缘被错误归属 → 切换到 🟢 纳入笔刷
3. 在错误区域涂抹 → 笔画以绿色半透明覆盖显示
4. 对照检查：发现另一处天空碎片被错误纳入 → 切换到 🔴 排除
5. 涂抹排除区域 → 笔画以红色半透明覆盖
6. 点击「应用SAM」：
   a. 🟢 笔画区域 → 作为 SAM 正提示点 → 在当前层扩展
   b. 🔴 笔画区域 → 作为 SAM 负提示点 → 从当前层移除
   c. SAM 只在笔画覆盖的局部 bbox 内运行 → < 1s
7. 开发者验证结果 → 满意/继续修正
8. 确认标定
```

### 4.3 笔刷引擎技术要求

- Canvas 画布分层：原图层 / 蒙版叠加层（半透明彩色）/ 笔画层（SVG 路径）
- 笔画存储为坐标点序列，非光栅图（支持缩放后重新渲染）
- 笔刷大小范围 5-100px，默认 20px
- 支持圆形笔刷（基础）/ 可扩展为软边笔刷
- 撤销栈保留最近 20 次操作

---

## 五、标定数据格式升级

### 5.1 新增字段

```json
{
  "images": {
    "a1b2c3...": {
      "filename": "上海.jpg",
      "width": 5472,
      "height": 3078,
      "megapixels": 16.8,
      "features": { ... },
      "params": {
        "n_layers": 3,
        "frame_width": 50,
        "min_island_area": 100,
        "quality": "standard",
        "refine_mode": "boundary_sam"     ← 新增
      },
      "brush_events": [                    ← 新增
        {
          "layer": 1,
          "type": "include",
          "bbox": [1200, 800, 1350, 950],
          "point_count": 43,
          "timestamp": "2026-06-17T17:30:00"
        }
      ],
      "brush_event_file": "brush_events/a1b2c3_20260617_173000.json",  ← 新增
      "scores": { ... },
      "labeled_at": "2026-06-17T17:35:00"
    }
  }
}
```

### 5.2 数据质量标准

**有效标定条件**：
- 开发者通过「确认标定」按钮显式确认（不仅是自动分割后直接存）
- 或：CLI label 模式下按 Y 确认
- 或：笔刷修正后点「确认标定」

**无效标定**（不入训练集）：
- 仅自动分割、未经过开发者确认
- 「跳过」按钮处理
- 评分 `combined_score < 0.5` 且未修正

---

## 六、ML 闭环设计

### 6.1 数据积累路径

```
自动分割（新管线）
    ↓
开发者逐层审核 + 笔刷修正
    ↓ 每次笔刷操作自动记录
brush_events/{hash}_{ts}.json
    ↓ 确认标定
labeled.json（含 brush_events 引用 + 精修后蒙版）
    ↓ 积累 ≥50 张有效标定
训练数据集
```

### 6.2 两层 ML 目标

**第一层：参数预测（延续当前 RF，改进数据质量）**

| 项目 | 旧 | 新 |
|------|-----|-----|
| 训练数据 | 自动标记（无效） | 开发者确认 + 笔刷修正后 |
| 特征 | 11 维全局特征 | 11 维全局 + 笔刷事件统计特征 |
| 模型 | RandomForest | RandomForest（不变） |
| 最低样本 | 10 | 50（质量优先） |

**第二层：精修触发预测器（新增）**

```
输入：深度等距量化的边界 + 该边界的局部特征
输出：{segment_id: 是否需要 SAM 精修}  (二分类)
训练数据：来自笔刷事件
  开发者涂抹了某段边界 → label = "需要精修"
  开发者未触碰的边界段 → label = "不需要精修"
```

使用时：
1. 深度等距量化 → 提取所有层间边界段
2. 精修预测器判断每段是否需要 SAM
3. SAM 只在预测为"需要"的段上运行
4. 跳过"不需要"的段（如地面线、简单直线）

### 6.3 训练触发条件

| 条件 | 第一层（参数预测） | 第二层（精修触发） |
|------|------------------|-------------------|
| 最少样本 | 50 张有效标定 | 50 张有笔刷事件的标定 |
| 每样本至少 | 开发者确认 | ≥3 次笔刷事件 |
| 验证方式 | OOB score | 留一交叉验证 |

### 6.4 特征集扩展

在现有 11 维全局特征基础上，新增笔刷事件派生的统计特征：

| 特征 | 来源 | 影响 |
|------|------|------|
| `brush_event_count` | 笔刷事件总数 | 判断图是否难分割 |
| `brush_include_ratio` | 纳入/排除笔刷比 | 漏割 vs 过割倾向 |
| `brush_coverage_pct` | 笔刷覆盖面积占比 | 自动分割错误率 |
| `brush_layer_distribution` | 各层笔刷事件分布 | 哪层最难分割 |
| `avg_brush_component_size` | 笔刷区域平均大小 | 错误粒度 |

---

## 七、实施阶段

### 阶段 1：边界精修引擎

**目标**：实现 `boundary_refine.py`，在标定器中跑通新管线。

| 文件 | 操作 | 内容 |
|------|------|------|
| `dev_tools/labeler/boundary_refine.py` | 新建 | 边界带提取 + SAM 框选精修 + 回填 |
| `dev_tools/labeler/labeler_server.py` | 修改 | `run_segmentation()` 增加 `refine_mode` 参数，支持新旧管线切换 |

**验证**：在上海.jpg 上跑新管线，视觉对比新旧结果。

### 阶段 2：笔刷工具（前端）

**目标**：实现 Canvas 笔刷引擎 + SAM 局部再精修。

| 文件 | 操作 | 内容 |
|------|------|------|
| `dev_tools/labeler/static/index.html` | 修改 | 新增「逐层审核」模式 + 工具栏 |
| `dev_tools/labeler/static/brush_tool.js` | 新建 | Canvas 笔刷引擎（纳入/排除/撤销/笔刷大小） |
| `dev_tools/labeler/labeler_server.py` | 修改 | 新增 `POST /api/brush-refine`（SAM 局部精修 API） |

**交互**：
1. 分割完成后 → 切换到「逐层审核」
2. 笔刷涂抹 → 前端收集笔画坐标
3. 点击「应用SAM」→ `POST /api/brush-refine {layer, brush_strokes}`
4. 后端：笔画 → SAM point prompts → 局部 bbox → SAM → 返回精修后蒙版
5. 前端更新蒙版显示

### 阶段 3：笔刷事件记录

**目标**：每次笔刷操作自动记录，为 ML 积累数据。

| 文件 | 操作 | 内容 |
|------|------|------|
| `dev_tools/labeler/brush_recorder.py` | 新建 | 事件记录 + 局部特征计算 |
| `dev_tools/labeler/labeler_server.py` | 修改 | 笔刷 API 调用 recorder，确认标定时写入 labeled.json |

**记录时机**：每次「应用SAM」操作自动触发，无需开发者额外操作。

### 阶段 4：标定数据格式升级

**目标**：labeled.json 新增 brush_events 字段，loading 逻辑兼容新旧格式。

| 文件 | 操作 | 内容 |
|------|------|------|
| `dev_tools/scripts/test_sam_segment.py` | 修改 | `ImageRegistry` 兼容新字段，`get_labeled_dataset()` 筛选有效数据 |
| `dev_tools/data/labeled.json` | 重建 | 清除旧数据后的新格式（v2） |

### 阶段 5：CLI 工具同步

**目标**：`test_sam_segment.py` 的 label/scan/train 模式对接新引擎。

| 文件 | 操作 | 内容 |
|------|------|------|
| `dev_tools/scripts/test_sam_segment.py` | 修改 | label 模式使用 boundary_refine；scan 模式支持参数 --refine |

### 阶段 6：首批有效数据积累 + ML 训练

**目标**：用新工具标定 ≥10 张图，跑通第一轮训练。

| 活动 | 产出 |
|------|------|
| 标定上海.jpg | 验证全链路 |
| 标定 10 张城市天际线 | 首批有效训练集 |
| `--mode train` | 第一版 RF 预测器 + 特征重要性分析 |

---

## 八、效率设计原则

### 8.1 标定吞吐量目标

| 操作 | 耗时目标 | 说明 |
|------|---------|------|
| 深度估计（首次） | < 20s | 必须等待，一次性 |
| 深度估计（缓存命中） | < 0.1s | 复用缓存 |
| 等距量化 + 边界带提取 | < 0.5s | 纯 numpy |
| SAM 边界带精修（全图） | < 15s | 只在边界带内运行，非全图 |
| SAM 笔刷局部精修 | < 1s | 只跑笔画 bbox |
| 开发者审核 + 笔刷 | 1-3min | 人类时间 |
| **整体：新图从零到确认标定** | **< 3min** | |

### 8.2 防重复设计

- 深度缓存：图片哈希 → 一次推理永久复用
- SAM 笔刷精修结果缓存：同一笔画区域不重复跑 SAM
- 标定历史：同一图片同一参数组合不重复处理

---

## 九、文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `dev_tools/labeler/boundary_refine.py` | 新建 | 边界带 + SAM 精修引擎 |
| `dev_tools/labeler/brush_recorder.py` | 新建 | 笔刷事件记录模块 |
| `dev_tools/labeler/static/brush_tool.js` | 新建 | 前端 Canvas 笔刷引擎 |
| `dev_tools/labeler/static/index.html` | 修改 | 新增逐层审核模式 + 工具栏 |
| `dev_tools/labeler/labeler_server.py` | 修改 | 新增精修模式、笔刷 API、事件记录 |
| `dev_tools/scripts/test_sam_segment.py` | 修改 | label/scan/train 对接新引擎，兼容新 labeled.json |
| `dev_tools/data/labeled.json` | 重建 | v2 格式，含 brush_events 字段 |
| `dev_tools/data/brush_events/` | 新建 | 笔刷事件存储目录 |
| `dev_tools/data/layer_predictor.pkl` | 延迟 | 首批 ≥50 张有效标定后训练 |
| `docs/开发者优化SAM处理脚本计划.md` | 重写 | 本文件 |

**不动任何 `dev_tools/` 以外的文件。**

---

## 十、Gap Analysis + 容错

| 盲区 | 对策 |
|------|------|
| 边界带过宽（包含过多区域） | `band_width=5`，后续根据验证调整；过大 → SAM 运行量激增 |
| 边界带遗漏（深度梯度为零处） | 补 RGB Canny 边缘作为备用触发器；对深度中位数=0 的极端图片特殊处理 |
| SAM 精修把建筑切成碎片 | 精修后验证连通分量数，异常增多时退回原始等距量化 |
| 笔刷事件文件数量爆炸 | 每次「确认标定」后合并同一图片的事件文件为单文件 |
| labeled.json 格式升级兼容 | `version` 字段 → `2`，ImageRegistry 加载时自动兼容 v1 |
| 笔画在缩放/拖拽后坐标偏移 | 所有笔画坐标存储为原图像素坐标（非 viewport 坐标），渲染时动态转换 |
| 深度缓存与注册表不同步 | 哈希去重不受文件名影响，深度缓存 stem 与注册表 filename 对齐 |
| SAM mobile_sam.pt 缺失 | draft 模式跳过所有 SAM 步骤；精修模式缺模型时降级为纯等距量化 + 警告 |
| CPU 环境 SAM 推理耗时 | 笔刷 SAM 精修只跑局部 bbox（< 1s）；边界带精修可选 draft 跳过 |
