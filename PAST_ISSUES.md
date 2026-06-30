# 已解决问题记录 (Past Resolved Issues)

> 记录项目中遇到的每个问题、根因、以及最终修复方式。
> 
> **最新记录**：案例 31（2026-06-17 晚，谷底阈值保底）。自 2026-06-30 起，开发环境已从 Linux (WSL2) 完全迁移至 Windows 11 原生。详见 `README.md` 和 `AI_RULES.md` 最新版本。
> 每个案例包含：症状 → 诊断 → 根因 → 修复 → 验证。

---

## 案例 1：项目无版本控制 —— ✅ 已修复

- **日期**：2026-06-11
- **症状**：项目根目录有 `.gitignore` 但没有 `.git/` 目录
- **根因**：项目从未初始化 git
- **修复**：`git init` → 分支改名 `main` → 首次提交（1fef368）
- **验证**：`.git/` 目录已存在，git log 有提交历史
- **影响**：无回滚能力、无变更追溯、无法 push
- **关联规则**：AI_RULES.md 未涉及（已通过口头约定补充：push 前必须开发者确认测试通过）

---

## 案例 2：全局异常处理器只返回 `str(exc)` 而非完整 Traceback —— ✅ 已修复

- **日期**：2026-06-11
- **症状**：前端 Axios 拦截器捕获到的 `error_msg` 只有一行异常消息，无法定位问题
- **根因**：`main.py:53` 写的是 `"error_msg": str(exc)`，而 `tb = traceback.format_exc()` 已经捕获了完整堆栈但没返回
- **修复**：`"error_msg": str(exc)` → `"error_msg": tb`（02c982c）
- **验证**：`main.py` 第53行确认为 `"error_msg": tb`；API_CONTRACT.md §1.1 要求合规

---

## 案例 3：线稿引擎从 `lineart_anime` 切换到 `canny_lineart` —— ✅ 已修复

- **日期**：2026-06-11
- **原因**：用户决策 — canny_lineart 纯 CPU、零模型依赖、117 图验证
- **变更范围**：
  - `api/lineart.py` → `api/canny.py`
  - `services/lineart_service.py` → `services/canny_service.py`
  - `utils/lineart_anime.py` 删除，`utils/canny_lineart.py` 新增
  - `models/requests.py` 参数从 `detect_resolution/line_strength/thin` → `low/high/smooth_level`
  - API_CONTRACT.md § Step 4 参数描述同步更新
- **提交**：775cc3f

---

## 案例 4：`test_canny.py` 输出到错误目录 —— ✅ 已修复

- **日期**：2026-06-11
- **症状**：`test_canny.py` 将测试结果写入 `dev_tools/data/`（模型/JSON 目录）
- **根因**：第 25 行硬编码 `OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "data"`
- **修复**：改为 `OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "canny"`
- **附带修复**：建立 `dev_tools/outputs/<stage>/` 完整目录树（sandbox/sam/canny/denoise/connectivity/svg），编写 `outputs/README.md` 硬性规则

---

## 案例 5：前端线稿预览不更新（浏览器缓存问题） —— ✅ 已修复

- **日期**：2026-06-12
- **症状**：拖动 Canny 参数滑块后右侧预览图不变化
- **诊断**：
  1. curl 验证后端产出不同 MD5（13% 像素差异）→ 排除后端
  2. 发现 `result_url` 始终是 `/outputs/{id}_canny.png` → 固定 URL
  3. 浏览器 DevTools Network 面板确认无新 GET 请求 → 浏览器缓存
- **根因**：Canvas.tsx 的 `<img>` 标签没有 `key` 属性，URL 不带版本参数。浏览器看到同一个 URL 直接返回缓存，React 也没重建 DOM
- **修复**：
  - App.tsx 加 `cannyVersion` 状态，每次拿到新结果后递增
  - Canvas.tsx 的 `<img>` 加 `key={version}` 和 `?v={version}`
  - 扩展到多步骤：denoise/connectivity/svg 各自维护独立 version 计数器
- **经验固化**：`docs/agent-debug-sop.md` — 分层诊断树 + Cache-Busting 标准修复模式

---

## 案例 6：AI_RULES.md 缺失输出目录规则 —— ✅ 已修复

- **日期**：2026-06-12
- **症状**：AI_RULES.md §2 定义了 `data/` 和 `benchmarks/`，但没有 `outputs/` 的硬性规则
- **修复**：`dev_tools/outputs/README.md` 作为补充规则文档，三条硬性规则写入；AI_RULES.md §3 已明确引用
- **状态**：✅ 已解决 — outputs/ 规则由 dev_tools/outputs/README.md 定义，AI_RULES.md §3 已有引用路径

---

## 案例 7：SAM 依赖安装问题（Docker 环境） —— ✅ 已修复并验证

- **日期**：2026-06-12
- **症状**：`segment-anything` 在 Docker 构建时安装失败
- **诊断**：Dockerfile 中 `pip install segment-anything` 需先安装 `torch`
- **修复**：调整 Dockerfile.dev 依赖安装顺序，PyTorch 先于 segment-anything
- **状态**：Dockerfile.dev 已调整 PyTorch→MobileSAM 安装顺序，容器正常运行中，已验证通过

---

## 案例 8：DenoiseParams / ConnectivityParams / SvgParams 缺失 layer_index 字段 —— ✅ 已修复

- **日期**：2026-06-12
- **症状**：Denoise、Connectivity、SVG 三个端点在任何请求下都会抛出 `AttributeError: 'XxxParams' object has no attribute 'layer_index'`，返回 500
- **诊断**：
  1. 读取 `requests.py`，对比 `CannyParams`（有 `layer_index`）与另外三个模型（没有）
  2. 检查对应 service 文件 — `denoise_service.py:81`、`connectivity_service.py:73`、`svg_service.py:88` 均无条件访问 `params.layer_index`
  3. 追溯提交历史：架构升级时 service 层同步添加了多层模式分支，但 Request 模型忘记同步更新
  4. Pydantic v2 默认 `extra='ignore'`，前端即使传了 `layer_index` 也被静默丢弃，导致属性不存在
- **根因**：Service 层与 Model 层不同步 — 三个 Pydantic 请求模型漏加了 `layer_index` 字段
- **修复**：`requests.py` 中 `DenoiseParams`、`ConnectivityParams`、`SvgParams` 各加一行 `layer_index: Optional[int] = Field(None, description="Layer index (null for single-layer mode)")`
- **验证**：
  1. Python 语法检查通过
  2. Phase 1：6 个 Pydantic 测试全部通过（传/不传 layer_index）
  3. Phase 2：单层模式 denoise/connectivity/svg 全部 200
  4. Phase 3：多层模式 layer_index=0 全线通过，产出正确文件名

---

## 案例 9：sam_engine.py 使用 print() 违反 AI_RULES.md §5 —— ✅ 已修复

- **日期**：2026-06-12
- **症状**：SAM 模型加载/下载的状态信息直接输出到 stdout，不经过 loguru 日志管道（无时间戳、无级别、不入 `/tmp/backend.log`）
- **诊断**：`grep print sam_engine.py` 发现 9 处 `print()` 调用，全部在 `_download_model()` 和 `_get_sam_model()` 函数中
- **根因**：`sam_engine.py` 模块顶层未导入 loguru；`run_sam_automatic()` 和 `refine_mask()` 函数内部有局部 `from loguru import logger`，但模块级函数（`_download_model`、`_get_sam_model`）没有
- **修复**：
  1. 模块顶层加 `from loguru import logger`
  2. 9 处 `print()` → `logger.info()`（正常流程）/ `logger.warning()`（异常回退）
  3. 日志消息统一为中文，与项目其他模块风格一致
- **验证**：`grep print sam_engine.py` 返回 0 结果

---

## 案例 10：死代码清理 — 空文件 + 未使用 import —— ✅ 已修复

- **日期**：2026-06-12
- **症状**：`image_io.py`（0 字节）和 `anime_edge.py`（0 字节）作为空文件残留；`svg_service.py` 导入 `render_layer_frame` 但从未调用
- **诊断**：
  1. `search_files` 确认两个文件均 0 字节且无任何模块 import
  2. `anime_edge.py` 本应在案例 3 中删除，但仅被清空未移除
  3. `grep render_layer_frame\(` 确认只在定义处出现，无调用点
- **根因**：重构后遗留垃圾文件；import 语句未随代码演进清理
- **修复**：删除 `image_io.py`、`anime_edge.py`；`svg_service.py` import 中移除 `render_layer_frame`
- **验证**：全项目搜索 `image_io` / `anime_edge` / `render_layer_frame(` 无残留引用

---

## 案例 11：全项目 "lineart" 命名残留 —— ✅ 已修复

- **日期**：2026-06-12
- **修复（方案 A，3 批执行）**：
  - 第 1 批（后端）：`LineArtParams` → `CannyParams`；路由 `/pipeline/lineart` → `/pipeline/canny`；输出文件名 `_lineart_L{N}.png` → `_canny_L{N}.png`；denoise/connectivity/svg 输入查找同步更新
  - 第 2 批（前端）：PipelineStep `'lineart'` → `'canny'`；所有 `lineart*` 状态/ref/函数 → `canny*`；ControlPanel props/labels；MultiLayerCanvas case；types/index.ts `LineArtParams`/`lineartUrl` → `CannyParams`/`cannyUrl`
  - 第 3 批（文档）：API_CONTRACT.md 路由+模型名；AI_RULES.md；PROGRESS.md；agent-debug-sop.md curl 示例+变量名
- **保留不动**：`canny_lineart` 引擎函数名（正确语义）、`dev_tools/lineart_engine_export/` 目录名（可移植引擎包）、benchmarks 历史数据
- **验证**：前端零 `LineArt`/`lineart` 命中；后端全部 `_canny_` 模式；全项目 `grep -rn 'lineart\|LineArt'`（排除 PAST_ISSUES.md/canny_lineart/benchmarks历史）零命中

---

## 案例 12：connectivity 和 svg 步骤缺少独立 version 计数器 —— ✅ 已修复

- **日期**：2026-06-12
- **修复**：App.tsx 新增 `connectivityVersion`/`svgVersion` useState；`callConnectivity()`/`callSvg()` 中递增；version 三元表达式补全 connectivity/svg 分支；`handleUploaded` 重置两个计数器
- **验证**：App.tsx 第68、91行确认两个计数器存在；第700行 version 三元表达式覆盖所有步骤

---

## 案例 13：callSvg useCallback 误依赖 connectivityResult —— ✅ 已修复

- **日期**：2026-06-12
- **修复**：从 `callSvg` 的 `useCallback` 依赖数组中删除 `connectivityResult`，改为 `[]`
- ⚠️ **已知遗留问题（已修复 2026-06-12）**：修复引入了新问题 — `callSvg` 第 412 行直接访问 `connectivityResult` 状态值（`connectivityResult ? connectivityGenRef.current : null`），但依赖数组为空 → 闭包过期。`connectivityResult` 将永远为初始值 `null`，导致 `setSvgConnectivitySnap(null)` 始终被调用，SVG 的 stale 检测失效（connectivity 重跑后 SVG 不会标记为 stale）。
- **最终修复（2026-06-12）**：新增 `connectivityResultRef`（第 132 行），遵循项目中已有的 ref 桥接模式；第 413 行改为 `connectivityResultRef.current ? connectivityGenRef.current : null`
- **验证**：`grep connectivityResultRef App.tsx` 命中第 132、413 行；`grep "connectivityResult ?" App.tsx` 零命中（不再直接访问 state）

---

## 案例 14：AI_RULES.md §3 目录树残留已删除文件引用 —— ✅ 已修复

- **日期**：2026-06-12
- **修复**：AI_RULES.md §3 utils/ 目录树已重构：删除 `image_io.py`/`anime_edge.py`，替换为实际存在的 8 个算法文件
- **验证**：`grep "anime_edge\|image_io" AI_RULES.md` 零命中

---

## 案例 15：API_CONTRACT.md 中 SVG 文件名与实际产出不一致 —— ✅ 已修复

- **日期**：2026-06-12
- **修复**：API_CONTRACT.md § Step 7 `uuid-string_final.svg` → `uuid-string.svg`（与 `svg_service.py` 实际产出一致）；同时修正 § Step 4 `uuid_lineart.png` → `uuid_canny.png`
- **验证**：`grep "final.svg\|_lineart.png" API_CONTRACT.md`（Step 描述部分）零命中

---

## 案例 16：n_layers=100 静默返回空（误报 — Pydantic 校验已生效）

- **日期**：2026-06-12
- **症状**：测试中 `n_layers=100` 返回 `len(layers)=0`，一度被判定为 bug
- **诊断**：`requests.py:33` 中 `SegmentParams.n_layers` 已有 `Field(3, ge=2, le=5)`。测试脚本用 `d.get('layers', [])` 读取 422 校验错误响应（不含 `layers` key），`[]` 导致 `len=0` 被误读为"返回 0 层"
- **根因**：测试脚本误判 — Pydantic `le=5` 正常工作，`n_layers=100` 正确返回 422：`"Input should be less than or equal to 5"`
- **修复**：无需改代码。测试脚本已修正解析逻辑
- **验证**：重新 curl 验证确认 422 + 完整校验消息

---

## 案例 19：多层模式切换排列方式后手势卡死

- **日期**：2026-06-12
- **状态**：✅ 已修复（待用户浏览器验证）
- **症状**：从叠加模式切换到纵向/横向排列后，滚轮缩放和拖拽平移无响应。切回叠加模式仍卡死
- **诊断**：`MultiLayerCanvas.tsx` 使用三元表达式 `viewMode === 'overlay' ? (controls+viewport+sliders) : (viewport)`。两个分支第一个子元素不同，React reconciliation 销毁并重建 `.canvas-viewport` DOM 节点。`useZoomPan` 的 `useEffect` 依赖项 `[containerRef, ...]` 全部稳定（ref 对象不变），新 DOM 节点创建后不会重新绑定 wheel/mousedown 事件
- **根因**：三元分支的兄弟元素结构不同 → React 重建 viewport DOM → 事件监听器丢失且不会重新绑定
- **修复**：两步修复 — (1) 将 `canvas-viewport` 从三元分支提取到外层，始终渲染同一个 DOM 节点。controls 和 sliders 用 `{viewMode === 'overlay' && (...)}` 条件渲染在 viewport 外部；(2) 给 `useZoomPan` 新增 `reconnectKey` 参数，`MultiLayerCanvas` 传入 `reconnectKey: viewMode`，切换模式时强制 detach + re-attach 事件监听器
- **验证**：Docker 内 `npx tsc --noEmit` 零错误；用户浏览器验证通过 — 三种排列模式下手势均正常，切换不丢失

---

## 案例 20：多层模式原图按钮无法隐藏原图

- **日期**：2026-06-12
- **状态**：✅ 已修复（待用户浏览器验证）
- **症状**：多层叠加模式下「原图」按钮点击无反应，始终显示且无法隐藏
- **诊断**：`MultiLayerCanvas.tsx` 第 92 行原图按钮 `disabled={false}` 无 `onClick` handler；第 120 行原图 `<img>` 固定 `opacity: 1` 无条件渲染
- **根因**：原图按钮只是装饰性 UI 元素，未绑定 visibility toggle 逻辑
- **修复**：新增 `const [origVisible, setOrigVisible] = useState(true)` 本地状态；按钮加 `onClick={() => setOrigVisible(v => !v)}` + className 动态 active；原图 `<div>` 包裹在 `{origVisible && (...)}` 内；纵向/横向排列模式下原图作为第一个 tile 显示，附带独立 toggle 按钮
- **验证**：Docker 内 `npx tsc --noEmit` 零错误；用户浏览器验证通过 — 叠加模式下可切换原图显隐，纵向/横向排列中 toggle 正常工作

---

## 案例 21：预览图拖拽时被浏览器意外选中/拖拽

- **日期**：2026-06-12
- **状态**：✅ 已修复（用户验证通过）
- **症状**：拖拽平移预览图时，(1) 图片被浏览器选中变蓝，(2) 松开鼠标后图片仍被选中并跟随光标移动，干扰手势操作
- **诊断**：
  - `.canvas-viewport` — 缺失 `user-select: none`
  - `.layer-tile-img` — 缺失 `user-select: none` + `-webkit-user-drag: none`
  - `.canvas-world-overlay` / `.layer-tiles` — 缺失 `user-select: none`
  - 对比 `.canvas-world-img` 已有完整保护
- **根因**：CSS 防选中规则未覆盖全部图片元素和容器；tiled 模式图片 `.layer-tile-img` 完全没有防 drag 保护
- **修复**：
  - CSS 5 处：`.canvas-viewport`、`.canvas-world-overlay`、`.layer-tiles` 加 `user-select: none`；`.layer-tile-img` 加 `user-select: none` + `-webkit-user-drag: none`
  - React 2 处：`Canvas.tsx` 和 `MultiLayerCanvas.tsx` 的 viewport div 各加 `onDragStart={e => e.preventDefault()}`
- **验证**：Docker 内 `npx tsc --noEmit` 零错误；用户浏览器验证通过 — 单层+多层所有模式下拖拽不再出现蓝色选中，图片不跟随光标

---

*记录格式遵循：症状（用户看到了什么）→ 诊断（如何定位的）→ 根因（真正的问题）→ 修复（做了什么）→ 验证（如何确认修好了）。*

---

## 案例 17：多层 SVG total_points 始终为 0

- **日期**：2026-06-12
- **状态**：✅ 已修复
- **症状**：`POST /pipeline/svg/multi-layer` 响应中 `total_points=0`，而单层 SVG 端点正确返回点数
- **诊断**：`svg_service.py:240` 硬编码 `total_points=0`，注释 `# not counted per-layer during merge; set 0`
- **根因**：多层 SVG 生成循环中 `total_paths, _ = _count_svg_stats(svg_str)` 丢弃了每层的 `total_points`
- **修复**：
  - 新增 `per_layer_points: list[int] = []`（第 181 行）
  - 循环内改为 `total_paths, total_points = _count_svg_stats(svg_str)` + `per_layer_points.append(total_points)`（第 204-206 行）
  - 响应中 `total_points=sum(per_layer_points)`（第 240 行）
  - logger 格式串补上 `total_points={}` 占位符（第 234 行）
- **验证**：curl 多层 SVG 端点，`total_points: 653905` > 0

---

## 案例 18：测试方案与代码实现的小偏差

- **日期**：2026-06-12
- **状态**：✅ 已修复
- **症状**：
  1. 测试方案中健康端点为 `/health`，实际为 `/api/health`
  2. 测试方案期望 `gap_tolerance=0` 返回 `bridges_built=0`，实际 Pydantic 校验 `ge=1` 返回 422
- **诊断**：方案编写时未对照实际 Pydantic 模型字段约束和 FastAPI 路由前缀
- **根因**：文档与代码不同步
- **修复**：`docs/联合调试_全功能测试.md` 中 4 处 `/health` → `/api/health`，2 处 `gap_tolerance=0` → `gap_tolerance=1`
- **验证**：`grep '/health[^/]' docs/联合调试_全功能测试.md` 零命中；`grep 'gap_tolerance.*0' docs/联合调试_全功能测试.md` 零命中

---

## 案例 22：SAM refine_mask 在 CPU+jit.trace 路径下崩溃 → 修复后低置信度返回空蒙版

- **日期**：2026-06-17
- **状态**：✅ 已修复
- **症状**：
  1. Labeler standard/fine 模式下每层报 `'numpy.ndarray' object has no attribute 'cpu'`，SAM 精修全部降级
  2. 修复崩溃后 SAM 真正运行，但置信度 0.26-0.39、返回 refined_fg=0%（空蒙版），导致全部分层内容丢失
- **诊断**：
  1. 日志 `层0 SAM精修失败（使用原始蒙版）| 'numpy.ndarray' object has no attribute 'cpu'` → 定位 `sam_engine.py:491` `masks[0].cpu().numpy()`
  2. 确认 `_device == "cpu"` 且模型被 `torch.jit.trace` 追踪 → MobileSAM 的 `SamPredictor.predict()` 在 CPU+jit 路径下返回 numpy array
  3. 修复崩溃后日志 `[SAM精修] 完成 | refined_fg=0.0%` + 置信度 0.3-0.4 → SAM 对深度引导蒙版不认可
- **根因**：
  - 崩溃：`sam_engine.py:491` 无条件调用 `.cpu()`，numpy array 无此方法
  - 空蒙版：MobileSAM 在 CPU+jit.trace 路径对深度引导蒙版置信度极低，返回全空结果（以前被 crash 掩盖）
- **修复**：
  1. `sam_engine.py:491` → `np.asarray(masks[0])`（兼容 torch tensor 和 numpy array）
  2. `sam_engine.py:508` 后新增质量门：`score < 0.5 || refined_fg_pct < rough_fg_pct * 0.1` → 退回原始蒙版
  3. `labeler_server.py:183` 后新增 uint8 规范化：SAM 精修返回 bool 数组，`cv2.imwrite` 不接受 → 统一 `mask.astype(np.uint8) * 255`
- **验证**：
  - 上海/北京/伦敦/伊斯坦布尔 4 张图 auto-segment 全部 200，SAM 日志显示「不可信 → 使用原始蒙版」
  - 蒙版文件 fg_px 非零（伦敦层0: 16946735430），max=255 确认 uint8 格式
- **影响文件**：`client_app/backend/app/utils/sam_engine.py`（2 处）、`dev_tools/labeler/labeler_server.py`（1 处）

---

## 案例 23：Labeler「重新分割」忽略用户滑块值 / 模式栏不显示

- **日期**：2026-06-17
- **状态**：✅ 已修复
- **症状**：
  1. 开发者设 n_layers=4 后点「🔄 重新分割」，返回仍是 3 层
  2. 分割完成后无法切换到「逐层」模式——模式栏（原图/彩色叠加/逐层）不显示
- **诊断**：
  1. 前端 `runSegManual()` 第 262 行始终调用 `/api/auto-segment`，无视 `segResult` 是否已存在
  2. 第 269 行将 ML 预测值覆盖回滑块 `document.getElementById('nLayers').value = pp.n_layers || 3`
  3. 第 275 行调 `setMode('overlay')` 但从未调 `showAllControls()`——modeBar/layerBar/opacityBar 保持 `display:none`
- **根因**：`runSegManual()` 未区分首次/重跑——首次和重跑都走 ML 预测路径，用户滑块调整被覆盖；分割成功后未恢复模式栏显示
- **修复**：`index.html` `runSegManual()` 重写：
  - 新增 `const isFirstRun = !segResult` 判断
  - 首次 → `/api/auto-segment` + 回填 ML 预测参数
  - 重跑 → `/api/segment` + 传当前滑块值（n_layers/frame_width/min_island_area/quality）
  - 分割成功后调用 `showAllControls()`
- **验证**：curl 模拟完整用户流程——首次 auto-segment 3 层，重跑 `/api/segment` n_layers=4 返回 4 层
- **影响文件**：`dev_tools/labeler/static/index.html`（1 处函数重写）

---

## 案例 24：逐层模式 CSS mask-image 不渲染 → Canvas 替代

- **日期**：2026-06-17
- **状态**：✅ 已修复
- **症状**：逐层模式下显示完整的长方形色块，而非蒙版切割后的形状
- **诊断**：
  1. 蒙版 PNG 文件验证：2D 灰度 PNG，0/255 值正确（层0: 54.2% white, 45.8% black）
  2. 创建独立 mask_test.html 隔离测试 CSS mask-image 方案
  3. 确认 CSS `mask-image` + `background-color` 在当前浏览器环境不工作
- **根因**：CSS `mask-image` 对灰度 PNG（无 alpha 通道）的 luminance masking 在不同浏览器/渲染引擎中表现不一致，该环境下完全不生效
- **修复**：`index.html` `renderLayersView()` 重写——CSS mask-image → Canvas 像素渲染：
  - 每个图层创建 `<canvas>` 元素
  - 加载 frame PNG → `ctx.drawImage()` → `getImageData()`
  - 遍历像素：白色（r>128）→ 替换为层颜色 + alpha=255；黑色 → alpha=0
  - `putImageData()` 写回 → 显示彩色蒙版形状
  - 逐层显隐/透明度通过 `.world-pane` 的 `opacity` CSS 控制（与旧方案兼容）
- **验证**：蒙版文件内容正确，Canvas 渲染逻辑通过代码审查（`getImageData` → 像素遍历 → `putImageData`）
- **影响文件**：`dev_tools/labeler/static/index.html`（1 处函数重写）

---

## 案例 25：boundary 模式精度差 → 彻底删除

- **日期**：2026-06-17 晚
- **状态**：✅ 已修复
- **症状**：用户选 4 层后用 boundary 模式，切割完全偏离深度热力图。天安门屋檐与天空的明显交界线未被切到
- **诊断**：boundary 管线先深度等距量化再边界带 SAM 精修。深度图仅 518px，等距量化将建筑拦腰斩断，SAM box prompt 的 BBox 无法包裹整栋建筑
- **根因**：boundary_refine 的设计缺陷——"先切后修"无法修复已被深度阈值切碎的建筑物。深度模型的 518px 空间分辨率是硬瓶颈
- **修复**：彻底删除 boundary 模式：
  - 删除 `boundary_refine.py`（240行）
  - 删除 `labeler_server.py` 中 boundary 分支（14行）
  - 清理 `SegmentRequest.refine_mode` pattern、`save_label` 默认值
  - 清理 `__pycache__` 残留
- **验证**：`grep -r "boundary" dev_tools/labeler/` 零匹配
- **影响文件**：`boundary_refine.py`（删除）、`labeler_server.py`、`index.html`

---

## 案例 26：sam_driven 3 层上限误区 → 移除限制，统一接管 2-10 层

- **日期**：2026-06-17 晚
- **状态**：✅ 已修复
- **症状**：用户请求 4 层，后端静默 clamp 到 3，前端预览不更新；切换到 boundary 后精度差
- **诊断链**：
  1. §4.24 理论担忧 `int(median_depth * n_layers)` 的 N>3 时区块被切散
  2. 实施了 3 层硬上限 + auto-fallback 到 boundary
  3. 实测推翻理论：上海.jpg 5472×3078，SAM 125 区块，3/4 层下分配率完全一致（97/125）
- **根因**：理论假设未经验证就设了限制。3 层上限的担忧在实践中不成立——SAM 125+ 精细区块的深度中位数已有足够分布密度
- **修复**：
  - `labeler_server.py` L74: `le=6` → `le=10`
  - `labeler_server.py` L195-200: 删除 auto-fallback 到 boundary
  - `index.html` L142: 滑块 `max="6"` → `max="10"`
  - `index.html` L297-315: 删除 autoMode 切换，统一走 sam_driven
- **验证**：上海.jpg/北京.jpg 2-8 层全部 sam_driven，请求N→返回N（9/9 通过）
- **影响文件**：`labeler_server.py`、`index.html`
- **通用原则**：理论担忧需要实测验证，不要基于假设设限制

---

## 案例 27：suggest_n_layers 自动推断覆盖用户 3 层选择

- **日期**：2026-06-17 晚
- **状态**：✅ 已修复
- **症状**：北京.jpg（suggest=2），用户操作序列 2→4→3，第 3 步请求 3 层却返回 2 层
- **诊断**：`labeler_server.py` L180-181 的 `if n_layers == 3 and suggested_n != 3: n_layers = suggested_n` 在通用分割函数中执行
- **根因**：自动推断逻辑最初为首次 auto-segment 设计（默认值 3→自动推断），但代码放在 `run_segmentation()` 中，导致手动重跑时也触发
- **修复**：从 `run_segmentation()` 删除自动推断。`suggested_n_layers` 仍返回给前端作为建议，前端用其填充初始滑块值
- **验证**：北京.jpg 2→2 ✅, 4→4 ✅, 3→3 ✅
- **影响文件**：`labeler_server.py` L175-181
- **通用原则**：自动推断应放在入口层（`api_auto_segment`/前端），不应在通用函数中越权

---

## 案例 28：谷底检测灵敏度过高 → 过滤浅谷

- **日期**：2026-06-17 晚
- **状态**：✅ 已修复
- **症状**：北京.jpg n=3 找到谷底 [0.430, 0.758]，0.430 是建筑内部浅谷（密度仅降 ~5%），切穿了建筑
- **诊断**：`find_valley_thresholds` 使用 `smooth_sigma=1.5` + 4 邻域比较 + 无谷深过滤
- **根因**：平滑不足使建筑深度范围的微小波动被误判为有效谷底
- **修复**：
  - `smooth_sigma`: 1.5 → 3.0
  - `n_bins`: 64 → 80
  - 新增 `min_dip_ratio=0.15`
  - 搜索半径 `max(2, int(smooth_sigma))`
  - 按 `dip_ratio` 降序排列
- **验证**：北京.jpg 谷底 [0.563, 0.860]，建筑-天空边界 0.86 成为主切点 ✅
- **影响文件**：`structural_segmentation.py` `find_valley_thresholds()`

---

## 案例 29：边框向内收缩遮挡图像内容 → 改为向外延伸

- **日期**：2026-06-17 晚
- **状态**：✅ 已修复
- **症状**：外框作为内部边框覆盖图像四边内容，遮挡雕刻细节
- **诊断**：`generate_frame_mask(h, w, fw)` 返回 (h, w) 尺寸，frame 在四边 fw 像素处设为 255 覆盖内容
- **根因**：边框方向错误——应该向外延伸作为物理支撑，不应该向内收缩覆盖内容
- **修复**：
  - `generate_frame_mask`: 返回 (h+2*fw, w+2*fw)，frame 在外围
  - `build_sam_driven_layers`: 层蒙版 `cv2.copyMakeBorder` pad 到扩展尺寸
  - `labeler_server.py`: overlay 原图同步 pad 白色边框
- **验证**：北京.jpg 6100×3431 → mask 6200×3531（+100px=2×50）✅
- **影响文件**：`structural_segmentation.py`、`labeler_server.py`

---

## 案例 30：逐层视图图层与原图偏移（外扩边框适配）

- **日期**：2026-06-17 晚
- **状态**：✅ 已修复
- **症状**：逐层模式下图层 canvas 与原图出现偏移，内容区不对齐
- **诊断**：层蒙版 6200×3531 vs 原图 6100×3431，CSS 原点相同（0,0），内容差 50px
- **根因**：外扩边框后前端未适配——layer pane 需负偏移 fw px 对齐内容区
- **修复**：`renderLayersView()` 中层 pane 使用 `left:-${fw}px; top:-${fw}px` 绝对定位
- **验证**：frontend 源码含 `left:-${fw}px; top:-${fw}px` ✅
- **影响文件**：`index.html` L444

---

## ⚠️ 复发型问题：外框 (fw) 偏移导致的图层错位

> **性质**：复发型配置问题（多次因移除 fw 偏移导致图层错位）
> **最近出现**：2026-06-30
> **关联文件**：`index.html:445`, `brush_tool.js:64-66`
>
> ### 根因
> 后端 `build_sam_driven_layers()` 生成的 mask 尺寸为 `(orig_h + 2*fw, orig_w + 2*fw)`。
> mask 有效内容从 `(fw, fw)` 像素才开始（外围 fw px 为白色边框）。
> 前端原图 `<img>` 尺寸为 `(orig_h, orig_w)`，原点在 `(0,0)`。
> 
> 要使图层 canvas 的内容区与原图对齐，**必须**在 world-pane 和 brushCanvas 上
> 使用 `left:-fw; top:-fw` 负偏移。移除这个偏移会导致图层向右下角偏移 fw px。
>
> ### 必须保持的三点一致性
> 1. `index.html` `renderLayersView()` — world-pane 的 `left:-${fw}px; top:-${fw}px`
> 2. `brush_tool.js` `enable()` — brushCanvas 的 `left: -fw; top: -fw`
> 3. 以上两项的 fw 值必须与后端 `frame_width` 参数一致
>
> ### 防止复发
> - 修改前端图层布局时，务必确认 mask 实际尺寸
> - 如果未来后端改为输出原图尺寸的 mask（去掉外框 padding），这三处必须同步移除 fw 偏移

---

## 案例 31：n=7 谷底阈值不够 → valley_quantize_depth IndexError

- **日期**：2026-06-17 晚
- **状态**：✅ 已修复
- **症状**：北京.jpg n=7 返回 HTTP 500 `list index out of range`
- **诊断**：`find_valley_thresholds` 分位数补足时接近限制太严，6 个阈值只生成 5 个；`valley_quantize_depth` 无兜底
- **根因**：二重——阈值生成不足 + 消费方无保护
- **修复**：分位数补足动态放宽接近限制 + 等距阈值硬兜底，确保 `taken[:needed]` 恒有 needed 个元素
- **验证**：北京.jpg n=7 正常返回 7 层 ✅
- **影响文件**：`structural_segmentation.py` `find_valley_thresholds()`
- **通用原则**：依赖外部生成 N 个值的函数必须有硬兜底
