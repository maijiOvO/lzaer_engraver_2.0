# 标定工具全面重构计划

> 版本: 2.3  
> 日期: 2026-06-24（笔刷工具根因诊断 + 优先级重新评估）  
> 关联: `dev_tools/labeler/`（标定器全栈）、`dev_tools/scripts/test_sam_segment.py`（CLI 工具）

---

## -1. 开发环境

| 项目 | 详情 |
|------|------|
| **操作系统** | Windows 11 (原生，非 WSL) |
| **Python 环境** | Windows 原生 Python 3.12 |
| **GPU** | Intel Arc B370 (Battlemage, Ultra 5 338H 核显) |
| **GPU 加速后端** | torch-directml (Microsoft DirectML for PyTorch) |
| **标定器运行方式** | `python dev_tools/labeler/labeler_server.py` (本机) |
| **client_app 运行方式** | `uvicorn app.main:app` (本机，非 Docker) |

> **注意**：`client_app/docker-compose.yml` 和 `DOCKER_INFRA_GUIDE.md` 是历史遗留的 WSL2 Docker 环境配置，
> 当前开发已完全迁移到 Windows 原生环境。后续 GPU 加速计划也基于此原生环境设计。

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
| 客户端等距量化 (raw) | 62.6/25.9/11.5% | ✅ 正确 | ❌ 粗锯齿 | 深度图边界模糊 |
| 客户端等距量化 (repaired) | 含外框 | ✅ | ❌ + 外框空白区 | 连通性修复，非质量问题 |
| SLIC + 深度投票 | 59.5/2.4/42.9% | ❌ 层1几乎空 | ❌ 毛刺多 | 分位数对左偏分布失效 |
| **SAM驱动分层 (sam_driven)** | **14.8/31.2/63.4%** | ✅ **无拦腰斩断** | ✅ **SAM原图边缘** | **当前默认引擎** |

**根因分析**：

- 深度模型（Depth-Anything-V2）对宏观空间结构的判断**正确**——能区分建筑群和天空
- 深度图的空间分辨率（518px 高）决定了它**无法提供精确的对象边界**
- 深度值分布严重左偏（中位数=0.0000），导致 SLIC 的分位数分层失效
- 关键矛盾：深度模型负责"哪片区域属于第几层"（正确），但旧管线让深度模型同时负责"边界画在哪"（错误）

### 0.3 核心思路（已落地）

```
SAM 模型 → "边界在这条线上"（精确边缘，原图精度）  ← SAM 决定对象形状
    +
深度模型 → "这片属于第 N 层"（层级归属）            ← 深度只管 Z 轴排序
    ↓
正确的分层蒙版（整栋建筑不会被拦腰切断）
```

### 0.4 管线演进历史

| 阶段 | 管线 | 默认引擎 | 状态 |
|------|------|---------|------|
| 阶段 0 | 等距量化 + 连通修复 | `none` | 保留（`--refine none`） |
| 阶段 1 | 等距量化 + 边界带 SAM 精修 | `boundary` | ⚠️ 已废弃（拦腰斩断） |
| 阶段 2 | **SAM 自动分割 → 深度归属** | **`sam_driven`** | ✅ **当前默认** |

`boundary` 废弃原因：先做 `quantize_depth` 等距切分 → 高楼被水平切断 → 边界带 BBox 是狭长条带，
无法包裹整栋建筑 → SAM 只能在局部补修 → 依然是断裂的半截楼。

### 0.5 当前核心阻塞点（2026-06-24 诊断）

**SAM 切割在实验室条件下能产出正确结果，但人工反馈链路完全断裂。** 

开发者无法用标定工具产出一张"满意的、可作为参考标准"的分割结果，因为：

1. **笔刷精修工具存在3个确定bug**，导致"逐层审核 + 笔刷修正"闭环完全不可用
2. **无任何有效标注数据**，ML训练无从谈起

因此当前优先级是：**修复笔刷工具 → 走通人工修正闭环 → 积累标注数据 → 再评估ML**。

---

## 一、目标

将标定工具从"自动化分割 + 被动确认"升级为"智能分割 + 人工精修 + 机器学习闭环"的完整开发者工具链。

**当前阶段目标**（修订于2026-06-24）：
1. **修复笔刷精修工具** — 解决坐标偏移、mask_key断裂、SAM调用方式错误三个bug
2. **走通端到端人工修正闭环** — 对一张图完成：分割→笔刷→精修→满意→确认标定
3. **积累首批有效标注** — 对10张城市图逐一审核确认，产出一批"开发者认可"的训练数据
4. **评估ML方向** — 基于有效数据跑 train 模式，观察RF预测器实际表现

---

## 二、笔刷精修工具 — 需求澄清

### 2.1 笔刷工具的真正意图

用户用笔刷进行**粗糙区域标记**，算法在标记区域内做**像素级精确切割**。

```
第一步（人）：用户粗略涂抹 → "大概是这片区域需要调整"
第二步（算法）：在标记区域内，自动找到精确的物体边界
第三步（合并）：纳入 = 旧蒙版 + 新精确区域，排除 = 旧蒙版 - 新精确区域
```

### 2.2 纳入/排除模式的约束

| 模式 | 允许的操作 | 禁止的操作 |
|------|-----------|-----------|
| 🟢 纳入 (include) | 往当前层**添加**新像素 | **绝对不能删除**当前层已有像素 |
| 🔴 排除 (exclude) | 从当前层**移除**已有像素 | **绝对不能添加**新像素到当前层 |

多层之间允许共享同一片区域（一个像素可以同时出现在多层）。

### 2.3 算法选择：笔刷精修阶段使用 SAM mask_input

笔刷精修的本质是用户提供了**粗略掩码提示**，让算法在提示区域内做像素级精确分割。SAM 的 `mask_input` 参数正好匹配这个需求：

| 比较维度 | 初次大范围分割 | 笔刷精修 |
|---------|-------------|---------|
| 输入 | 全图 | 笔刷涂抹区域 |
| SAM 调用 | `run_sam_automatic()`（全图自动分割） | `predict(mask_input=...)`（局部掩码精修） |
| 耗时 | ~290s（首次）/ ~8s（缓存） | < 3s |
| 精度 | SAM 自身精度上限（1200px） | 同左，但限制在笔刷区域内 |

**选型理由**：
- SAM 的 `mask_input`（`torch.Tensor`，shape `[1,1,256,256]`）专门用于"输入粗掩码，输出精修掩码"
- 它自动将粗糙的人类笔刷收缩/扩展到原图中物体的真实边界
- 不需要复杂的 point prompt 策略、不需要 box padding
- 已有的 `_upscale_mask_smooth()`（三次立方插值 + 高斯模糊 + Sobel 边缘吸附）保障上采样回原图分辨率时的精度

---

## 三、笔刷精修链路设计（修订版）

### 3.1 完整链路（含分辨率标注）

```
① 用户笔刷涂抹 → 坐标记录（原图分辨率）
   精度：无损。前端存储为原图像素坐标，不经过缩放。

② 后端光栅化 → 笔刷粗掩码（原图分辨率，uint8）
   精度：无损。在全分辨率画布上画圆填充笔画。

③ 缩小到 256×256 → torch.Tensor → SAM mask_input
   精度损失：刻意为之。SAM mask_input 固定 256×256。
   缩小采用 cv2.resize(INTER_AREA)，区域插值不丢结构。

④ SAM 推理 → 精修掩码（SAM 处理分辨率，max_dim ≤ 1200px）
   精度：SAM 自身精度上限。

⑤ 精修掩码 → 上采样回原图分辨率（如 5472×3078）
   精度保障：_upscale_mask_smooth()
     - cv2.INTER_CUBIC 三次立方插值
     - GaussianBlur(sigma=0.8) 抗锯齿
     - threshold(0.5) 二值化恢复
     - _snap_mask_to_edges(edge_band=3) Sobel 梯度吸附到原图边缘
   复用已有实现，无需重写。

⑥ 布尔运算：
   纳入: 新蒙版 = 旧蒙版 | 精修掩码
   排除: 新蒙版 = 旧蒙版 & ~精修掩码
   精度：逐像素位运算，全分辨率。

⑦ 写回文件 + 前端刷新显示
```

### 3.2 分辨率精度保障结论

整条链路中，精度损失仅发生在步骤③（256×256 mask_input 是SAM的固定要求）。步骤⑤使用**三次立方 + 高斯 + Sobel吸附**上采样回原图分辨率，已有代码经过验证，精度可靠。

---

## 四、当前Bug清单（P0 — 阻塞人工修正闭环）

### Bug 1：逐层审核模式下笔刷坐标偏移（⚠️ 核心问题）

**症状**：用户在逐层审核模式下涂抹的区域，与SAM精修后实际生效的区域存在偏移。

**根因**：
- `renderLayersView()`（index.html 第444行）为逐层Canvas设置 `style="left:-${fw}px; top:-${fw}px"` 偏移（默认 fw=50）
- `BrushTool.init()`（brush_tool.js 第43行）创建的笔刷Canvas位于 `top:0; left:0`
- 两者都在 `#world` 容器内，但笔刷Canvas没有做对应的 `-fw` 偏移补偿
- `vp2img()` 坐标转换未感知此偏移

**影响**：用户看到的蒙版叠加位置与笔刷实际记录的位置相差 `frame_width` 像素（默认50px）。

**修复位置**：`brush_tool.js` — `enable()` 方法增加 `frameWidth` 参数，笔刷Canvas偏移 `-fw`。

---

### Bug 2：currentMaskKey 前端拼接断裂

**症状**：点击「应用SAM」后，后端 `/api/brush-refine` 返回404 "蒙版不存在"。

**根因**：
- 前端 `applyBrushSam()` 对 overlay URL 做正则截断：`ovStem.replace(/_n\d+.*/, '')` → 得到 `"上海"`
- 后端拼接文件路径：`{mask_key}_mask_{layer_index}.png` → 查找 `上海_mask_0.png`
- 实际文件名为：`上海_n3_f50_i100_std_mask_0.png`（带完整suffix）

**修复方案**：
- `run_segmentation()` 返回值中新增 `mask_key` 字段（如 `"上海_n3_f50_i100_std"`），让数据生产方明确告诉消费方
- 前端 `applyBrushSam()` 直接读取 `segResult.mask_key`，不再截断拼接

**修复位置**：`labeler_server.py`（返回值加字段）+ `index.html`（前端读取新字段）。

---

### Bug 3：SAM 调用方式错误 — include/exclude 意图被丢弃

**症状**：不管用户使用纳入笔刷还是排除笔刷，SAM精修的结果都一样——整张蒙版被SAM输出覆盖。

**根因**：
- 后端收集了 `include_points` 和 `exclude_points`（第560-576行）
- 但 `predict()` 调用时传了 `point_coords=None, point_labels=None`（第616-618行），注释说"MobileSAM 维度冲突"
- SAM 实际只收到 `box` 参数（笔刷包围盒），**完全不知道用户想纳入还是排除**
- 第636行 `cv2.imwrite(str(mask_path), refined_u8)` 直接覆盖旧蒙版文件，**旧蒙版内容全部丢失**
- 没有做布尔运算（纳入=OR，排除=AND NOT）

**修复方案**：重写 `api_brush_refine()` 的核心逻辑：
1. 将笔刷笔画光栅化为粗掩码（原图分辨率 uint8）
2. 缩放到 256×256 作为 `mask_input`
3. 调用 `predict(mask_input=...)` 获取精修掩码
4. 上采样回原图分辨率（`_upscale_mask_smooth` + `_snap_mask_to_edges`）
5. 纳入：`new_mask = old_mask | refined_mask`
6. 排除：`new_mask = old_mask & ~refined_mask`

**修复位置**：`labeler_server.py` — `api_brush_refine()` 函数。

---

### Bug 4（次要）：精修后frame边框硬编码 `fw=50`

**症状**：`api_brush_refine()` 第642行 `fw = 50` 硬编码，不跟随用户分割时使用的 `frame_width`。

**影响**：如果用户使用 `frame_width=100` 分割，精修后frame裁剪仍按50px → 边框残留/内容误删。

**修复位置**：`labeler_server.py` 第642行。

---

## 五、实施计划（修订版）

### ✅ 阶段 0：SAM驱动分层引擎（已完成 — 2026-06-17）

- `build_sam_driven_layers()` 作为默认引擎
- SAM缓存 `{stem}_sam_region.npz`，首次290s/缓存8s
- 上海.jpg验证通过，无拦腰斩断

### ✅ 阶段 1：标定器基础设施（代码已完成，未验证）

| 模块 | 文件 | 状态 |
|------|------|------|
| Web后端 | `labeler_server.py` (765行) | ✅ 代码完成 |
| 前端 | `index.html` (780行) | ✅ 代码完成 |
| 笔刷引擎 | `brush_tool.js` (271行) | ✅ 代码完成，含3个bug |
| 笔刷API | `api_brush-refine` | ✅ 代码完成，调用逻辑错误 |
| 事件记录器 | `brush_recorder.py` (279行) | ✅ 代码完成，含局部特征提取 |
| CLI工具 | `test_sam_segment.py` (四模式) | ✅ 代码完成 |

### 🔴 阶段 2：笔刷精修Bug修复（当前 — 最高优先级）

**目标**：修复 Bug 1-3，走通"分割→笔刷→SAM精修→确认标定"完整闭环。

| 文件 | 改动 | Bug |
|------|------|-----|
| `brush_tool.js` | `enable()` 加 `frameWidth` 参数，笔刷Canvas偏移补偿 | Bug 1 |
| `index.html` | `setMode('review')` 传 `frameWidth`；`applyBrushSam()` 改用 `segResult.mask_key` | Bug 1,2 |
| `labeler_server.py` | `run_segmentation()` 返回值加 `mask_key` 字段 | Bug 2 |
| `labeler_server.py` | 重写 `api_brush_refine()`：笔刷→粗掩码→SAM mask_input→精修掩码+布尔运算 | Bug 3 |
| `labeler_server.py` | 硬编码 `fw=50` 改为从请求参数读取（可选，低优先级） | Bug 4 |

**不动**：`brush_recorder.py`、`client_app/` 下任何文件、`test_sam_segment.py`。

**验证标准**：用一张测试图（如上海.jpg）走通：
1. 选择图片 → 开始分割 → 逐层审核
2. 🟢纳入笔刷涂抹 → 应用SAM → 看到该层蒙版新增了像素（旧像素保留）
3. 🔴排除笔刷涂抹 → 应用SAM → 看到该层蒙版移除了像素（其他像素保留）
4. 确认标定 → `labeled.json` 写入正确

### 阶段 3：积累首批有效标注

**目标**：对 `test_imgs/train/` 下的10张城市图逐一审核确认。

| 活动 | 产出 |
|------|------|
| 每张图跑分割（`/api/auto-segment`） | 自动分割结果 |
| 逐层审核，必要时笔刷修正 | 人工校准 |
| 确认标定或跳过 | `labeled.json` 中写入有效标注 |

**质量标准**：开发者通过「确认标定」按钮显式确认（不仅是自动分割后直接存）。

### 阶段 4：ML 重新评估

**前置条件**：≥10张有效标定。

**评估项**：
- 跑 `test_sam_segment.py --mode train`，观察 RF 预测器的 OOB R² 和特征重要性
- 如果 RF 参数预测精度足够（OOB R² > 0.5），第二层 ML（精修触发预测器）可能不需要
- 如果精度不足，考虑：
  - 增加标注数量（>50张）
  - 引入笔刷事件特征（`brush_recorder.py` 积累的数据）
  - 训练"精修触发预测器"判断哪些边界段需要 SAM 精修

**第二层 ML（精修触发预测器）仅在以下条件满足时启动**：
- ≥50 张有笔刷事件的标定
- 每样本 ≥3 次笔刷事件
- 验证方式：留一交叉验证

---

## 六、ML 闭环设计（远期目标，当前不做）

### 6.1 数据积累路径

```
自动分割（sam_driven）
    ↓
开发者逐层审核 + 笔刷修正
    ↓ 每次笔刷操作自动记录（brush_recorder.py）
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
| 最低样本 | 10（自动） | 50（质量优先） |

**第二层：精修触发预测器（新增，远期）**

```
输入：深度等距量化的边界 + 该边界的局部特征
输出：{segment_id: 是否需要 SAM 精修} (二分类)
训练数据：来自笔刷事件
  开发者涂抹了某段边界 → label = "需要精修"
  开发者未触碰的边界段 → label = "不需要精修"
```

### 6.3 训练触发条件

| 条件 | 第一层（参数预测） | 第二层（精修触发） |
|------|------------------|-------------------|
| 最少样本 | 50 张有效标定 | 50 张有笔刷事件的标定 |
| 每样本至少 | 开发者确认 | ≥3 次笔刷事件 |
| 验证方式 | OOB score | 留一交叉验证 |

---

## 七、笔刷事件记录器（已实现，当前不阻塞）

### 7.1 `brush_recorder.py` 状态

代码已完成（279行），数据结构如下：

```python
@dataclass
class BrushEvent:
    image_hash: str
    image_name: str
    layer_index: int
    brush_type: str           # "include" | "exclude"
    point_count: int
    bbox: tuple[int, int, int, int]
    timestamp: str
    sam_score: float
    fg_pct_before: float | None
    fg_pct_after: float | None
    local_features: dict[str, float]
```

局部特征维度（6维）：
- `rgb_edge_mean` — Canny 边缘强度
- `texture_complexity` — 局部纹理方差
- `depth_gradient_mean` — 深度梯度均值
- `depth_median` — 深度中位数
- `depth_std` — 深度标准差
- `component_size` — 区域面积

### 7.2 当前不阻塞原因

记录器在 `/api/brush-refine` 成功返回后自动触发（第665-677行），无需开发者额外操作。Bug 1-3 修复后，记录器自然随之工作。

---

## 八、效率设计原则

### 8.1 标定吞吐量目标

| 操作 | 耗时目标 | 说明 |
|------|---------|------|
| 深度估计（首次） | < 20s | 必须等待，一次性 |
| 深度估计（缓存命中） | < 0.1s | 复用缓存 |
| SAM 自动分割（首次） | ~290s | 一次性，写缓存 |
| SAM 自动分割（缓存命中） | ~8s | 读缓存 |
| SAM 笔刷精修（mask_input） | < 3s | 局部推理 |
| 开发者审核 + 笔刷 | 1-3min | 人类时间 |
| **整体：新图从零到确认标定** | **< 5min** | |

### 8.2 防重复设计

- 深度缓存：`{stem}_depth.npy`，图片哈希 → 一次推理永久复用
- SAM分割缓存：`{stem}_sam_region.npz`，首次290s/缓存8s
- 标定历史：`ImageRegistry` SHA256 去重，同一图片不会重复处理

---

## 九、文件清单

| 文件 | 当前状态 | 需要改动 |
|------|---------|---------|
| `dev_tools/labeler/labeler_server.py` | ✅ 代码完成 | 🔧 加 `mask_key` 字段 + 重写 `api_brush_refine()` |
| `dev_tools/labeler/static/index.html` | ✅ 代码完成 | 🔧 修复 `mask_key` 读取 + `setMode` 传参 |
| `dev_tools/labeler/static/brush_tool.js` | ✅ 代码完成 | 🔧 修复坐标偏移 |
| `dev_tools/labeler/brush_recorder.py` | ✅ 代码完成 | 不动 |
| `dev_tools/scripts/test_sam_segment.py` | ✅ 代码完成 | 不动 |
| `dev_tools/data/labeled.json` | 已有（自动标注） | 修复后重新积累 |
| `dev_tools/data/brush_events/` | 已有目录 | 修复后自动积累 |
| `docs/开发者优化SAM处理脚本计划.md` | — | ✅ 本文档（v2.3） |

**不动任何 `dev_tools/` 以外的文件。**

---

## 十、Gap Analysis + 容错

| 盲区 | 对策 |
|------|------|
| SAM mask_input 与 MobileSAM 兼容性 | 先用单张图验证 `predict(mask_input=...)` 是否在 MobileSAM 上正常返回；如有维度问题则降级为 "笔刷区域 = 膨胀后直接做洪水填充 + Canny 边界" |
| 笔刷在缩放/拖拽后坐标偏移 | 所有笔画坐标存储为原图像素坐标（非 viewport 坐标），渲染时动态转换（已实现 `vp2img`） |
| 深度缓存与注册表不同步 | 哈希去重不受文件名影响 |
| SAM mobile_sam.pt 缺失 | draft 模式跳过所有 SAM 步骤；精修模式缺模型时返回明确错误信息 |
| 笔刷事件文件数量爆炸 | 每次「确认标定」后可合并同一图片的事件文件 |
| labeled.json 格式升级兼容 | `version` 字段 → `2`，ImageRegistry 加载时自动兼容 v1 |