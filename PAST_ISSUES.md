# 已解决问题记录 (Past Resolved Issues)

> 记录项目中遇到的每个问题、根因、以及最终修复方式。
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
