# AI 代理开发行为准则 (AI Constitution)

**【全局指令】在本项目中，你不仅是程序员，更是严格遵守架构红线的系统架构师。编写任何代码前，必须优先查阅并遵守以下 10 大准则。**

## 1. 基础设施神圣不可侵犯 (Docker & Infrastructure Locked) 【最高优先级】
- **运行方式**：当前开发环境已迁移至 Windows 11 原生。`client_app` 后端通过 `uvicorn app.main:app --reload` 本机运行（自动热重载），前端通过 `npm run dev`（Vite HMR）。Docker Compose 保留为可选部署方案。
- **热重载机制 (Hot Reload)**：
  - **本机运行**：后端 `--reload` 自动侦测 `.py` 变更；前端 Vite HMR 自动推送 `.tsx/.ts/.css` 变更。
  - **Docker 运行**：源码通过 Volume 映射至容器内部，修改业务代码后自动热重载生效。
- **绝对禁止盲目 Build**：**严禁**为了测试业务代码的改动而执行 `docker-compose build` 或 `docker compose up --build`！仅当 `requirements.txt` / `package.json` / `Dockerfile.dev` 变更时才能 rebuild。
- **禁止私改基建**：未经我明确授权，**绝对禁止**擅自修改 `Dockerfile.dev`、`docker-compose.yml`，**绝对禁止**在其中编写切换 apt/pip 镜像源的 sed 替换脚本。
- **依赖问题沟通**：如果运行缺少库/报错找不到模块，必须先向我报告，由我决定是写进 `requirements.txt` / `package.json` 还是本机 `pip install` / `npm install`。
- **详细的 Docker 挂载标准与大模型持久化配置，必须严格遵守根目录下的 DOCKER_INFRA_GUIDE.md**（注：Docker 配置为历史遗留的 WSL2 方案；当前开发以 Windows 本机运行为主）

## 2. 顶层架构隔离（一国两制）
本项目严格分为两个绝对隔离的顶级目录，严禁跨界污染：
- **`client_app/` (面向用户的 Web 应用)**
  - 包含 `frontend/` (React+Vite) 和 `backend/` (FastAPI)。
  - **运行规则**（二选一）：
    - **本机运行**：`uvicorn app.main:app --port 8080 --reload`（后端）+ `npm run dev`（前端）。当前主要开发方式。
    - **Docker Compose**：`docker-compose up -d`，完全隔离在容器网络中。保留为可选部署方案。
- **`dev_tools/` (面向开发者的本地工具与训练脚本)**
  - 存放纯本地 Python 脚本（如 OpenCV GUI 标注、模型训练、基准测试）。
  - **运行规则**：直接在宿主机运行。**绝对禁止**引入 FastAPI/Flask 或监听任何端口。如需调用后端算法，通过 `sys.path` 动态引入 `client_app/backend` 的纯函数模块。

## 3. 文件与目录洁癖
- **禁止“就地拉屎”**：绝对禁止在项目根目录、`backend/` 顶层或 `frontend/` 顶层生成任何临时的 `.json`, `.txt`, `.log` 文件。
- **数据归宿**：
- 应用运行生成的临时图片、测试结果必须输出到 client_app/backend/outputs/。
- 区分模型类别：
    预训练基础权重（如 mobile_sam.pt, Depth-Anything 权重）属于后端依赖，必须存放在 client_app/backend/models/ 或容器内的 HuggingFace 缓存中，严禁移出 client_app。
    本地训练产物（如你生成的预测参数 .pkl 模型、layer_params.csv）属于开发者私有数据，必须输出到 dev_tools/data/。
- 基准测试（Benchmarks）和对比用的静态 HTML，必须存放在 dev_tools/benchmarks/。
- **目录架构：
  laser-engraver-v2/
  ├── AI_RULES.md                      # 【核心】AI 代理开发宪法（所有代码生成的最高准则）
  ├── README.md                        # 项目总览说明
  ├── PROGRESS.md                      # 📋 项目进度记录（§10.1）
  ├── PAST_ISSUES.md                    # 🐛 已解决的问题记录（§10.2）
  ├── UNSOLVED_ISSUES.md                # 🐛 待解决问题记录（§10.2）
  ├── API_CONTRACT.md                   # 前后端 API 契约
  │
  ├── 📂 client_app/                   # 【战区一：面向用户的 Web 应用 (Docker 全隔离)】
  │   ├── docker-compose.yml           # Docker 运行环境（可选）：绑定前端 5173，后端 8080，挂载代码目录
  │   │
  │   ├── frontend/                    # ⚛️ 前端 (React 19 + Vite + TypeScript)
  │   │   ├── Dockerfile.dev           
  │   │   ├── package.json             # (前端唯一的 json 配置文件)
  │   │   ├── tsconfig.json
  │   │   └── src/
  │   │       ├── api/                 
  │   │       │   └── client.ts        # 💡 Axios 实例，内置《准则 4》要求的全局错误拦截器
  │   │       ├── components/          # React 组件库
  │   │       ├── hooks/               # 自定义 hooks（useZoomPan 等）
  │   │       ├── types/               # 严格的 TypeScript 接口
  │   │       └── App.tsx              
  │   │
  │   └── backend/                     # 🐍 后端 (FastAPI + OpenCV + Loguru)
  │       ├── Dockerfile.dev           
  │       ├── requirements.txt         # 依赖配置 (包含 loguru, controlnet_aux, pydantic)
  │       ├── .env                     # 💡 环境变量 (如 DEBUG_MODE=True 开启视觉断点日志)
  │       ├── outputs/                 # 📂 【数据归宿】应用生成的 SVG、中间态 Debug 图片一律存此
  │       └── app/
  │           ├── main.py              # 💡 FastAPI 入口，内置《准则 4》全局 Exception Handler
  │           ├── api/                 # 路由层 (负责接收请求、参数校验、调用 Service)
  │           ├── models/              # Pydantic 严格输入输出模型验证 (requests.py, responses.py)
  │           ├── services/            # 核心业务流 (如 7步图像管线的业务调度)
  │           └── utils/               # 💡 纯算法层 (纯函数 Sandbox，杜绝直接写 Web 逻辑)
  │               ├── canny_lineart.py  # Canny 线稿提取引擎
  │               ├── connectivity.py   # 连通性修复算法
  │               ├── denoise.py        # 物理降噪算法
  │               ├── depth_engine.py   # Depth-Anything-V2 深度估计
  │               ├── sam_engine.py     # SAM 分割引擎（深度引导流程中用于边界精修）
  │               ├── structural_segmentation.py  # 结构分层
  │               ├── svg_generator.py  # SVG 生成
  │               └── layer_frame.py    # 分层外框生成
  │
  └── 📂 dev_tools/                    # 【战区二：开发者本地工具 (纯本地运行，零 HTTP)】
      ├── scripts/                     # 🛠️ 工具脚本 (执行：python scripts/xxx.py)
      │   ├── test_canny.py            # Canny 线稿提取测试
      │   ├── test_denoise.py          # 降噪测试
      │   ├── test_pipeline.py         # 全管线端到端测试
      │   ├── test_sam_segment.py      # SAM 分割测试
      │   ├── layer_labeler.py         # OpenCV GUI 交互式标注工具 (直接在宿主机跑，调出窗口)
      │   ├── train_layer_params.py    # 机器学习训练脚本 (输出 .pkl)
      │   └── algorithm_sandbox.py     # 💡 算法沙盒测试：开发新算法时，先在这里只读取一张图跑通
      │
      ├── lineart_engine_export/       # 📦 独立可移植的 canny_lineart 引擎包
      │   ├── canny_lineart.py         # 核心算法（CLAHE + Canny）
      │   ├── batch_run.py             # 批量处理
      │   └── test_single.py           # 单图快速验证
      │
      ├── outputs/                     # 📂 【测试输出】所有测试脚本的运行时图像输出（§3 补充规则）
      │   ├── README.md                # 硬性规则：每个阶段输出到对应子目录
      │   ├── sandbox/                 # 算法实验
      │   ├── sam/                     # SAM 分割结果
      │   ├── canny/                   # Canny 线稿
      │   ├── denoise/                 # 降噪
      │   ├── connectivity/            # 连通性修复
      │   └── svg/                     # SVG 输出
      │
      ├── references/                  # 📖 参考学习资料
      │   ├── single/                  # 单层剪纸成品 PNG（降噪/连通性修复的品质参考）
      │   └── multiple/                # 多层纸雕 SVG（SVG 生成的格式参考）
      │
      ├── data/                        # 📂 【数据归宿】杜绝根目录拉屎，所有的 JSON/模型 都在这
      │   ├── layer_predictor.pkl      # 训练好的模型参数
      │   └── training_results.json    # 原先到处乱跑的各种 json 缓存收纳至此
      │
      ├── test_imgs/                   # 🖼️ 训练与测试图集
      │   ├── layer_params.csv         # 标注后的 CSV 文件
      │   └── (各类测试用的 .jpg/.png)
      │
      └── benchmarks/                  # 📊 性能与效果对比存档
          └── lineart_compare/         # ✨ 之前提出来的静态对比 HTML 工具和图片归档在这里
              ├── preview.html         # 双击即看，脱离前后端独立存在
              └── (对比用的大量图片)

  ├── 📂 docs/                         # 📚 排错与流程文档 + 测试报告
  │   ├── agent-debug-sop.md           # Agent 排错标准流程（缓存诊断树等）
  │   ├── README.md                    # 文档目录说明
  │   ├── 前端联合调试_全功能测试.md    # 前端全功能集成测试报告
  │   ├── 前端手动测试结果.md           # 前端手工测试记录
  │   └── 后端联合调试_全功能测试.md    # 后端全功能集成测试报告

## 4. 网络与端口（绝对红线）
- **固定端口**：前端固定映射宿主机 `5173`，后端固定映射宿主机 `8080`，后端服务监听 IP 必须为 `0.0.0.0`。
- **禁止换端口**：遇到端口被占用或无法连通时，**严禁修改代码中的端口号！严禁编写自动寻找空闲端口的逻辑！** 你的任务是排查服务状态（本机进程或 Docker 容器），而不是改代码。

## 5. 日志与调试规范 (Error Logging)
为了实现高效 Debug，必须严格执行以下防御性编程规范：
- **后端日志**：**严禁使用内置的 `print()`**。必须统一使用 `loguru` 库。
- **异常捕获**：FastAPI 必须配置全局 Exception Handler；在 `try-except` 块中，必须使用 `logger.exception(e)` 打印包含完整 Traceback 的堆栈。
- **管线节点追踪 (CV项目特有)**：在所有图像处理函数的入口和出口，必须通过 `logger.info` 记录输入/输出图像的 `shape`、`dtype` 和处理耗时。
- **视觉断点 (Visual Logging)**：在复杂的矩阵运算或图形拓扑算法中，遇到逻辑难以排查时，必须提供利用 `cv2.imwrite` 将中间结果输出到本地目录以供核对的代码分支。
- **前端拦截**：必须在 Axios 中配置全局响应拦截器，控制台报错必须格式化打印：`[请求方法] [URL] | 状态码 | Payload | 后端详情`。

## 6. 核心算法开发法则 (Standalone Sandbox)
- **纯粹性**：所有 OpenCV 图像处理算法（如线稿提取、连通性修复）必须写成**纯函数**。输入输出应为标准的图像矩阵（`numpy.ndarray`），参数必须通过 Pydantic 模型传递。
- **剥离测试**：开发复杂算法时，先写独立的 `test_xxx.py` 脚本进行纯本地读取图片测试。本地测试完美通过后，再封装进 FastAPI 的路由层。
- **回退机制 (Fallback)**：使用高级或平台绑定的库（如 `cv2.ximgproc`）时，尽量提供纯 Python 或基础库的回退实现，防止环境崩溃。

## 7. 沟通与协作模式
- **谋定而后动**：每次实现新功能或重构前，先用文字描述你的架构设计方案，我同意后你再写代码。严禁一次性生成超过 300 行的巨型文件。
- **Debug "事不过三" 原则**：如果你的代码报错，我给出报错日志后，**不要立刻给我新的修复代码**。你必须先列出导致该报错的 3 个最可能原因及验证方式。等我验证确认后，你再写代码。如果连续 2 次修复失败，必须停止写代码，重新梳理逻辑。

## 8. 实时预览规范 (Live Preview Standard)

**适用范围**：本规则适用于管线中所有图像处理步骤（线稿提取、降噪、连通性修复、SVG 生成等）。

**强制要求**：

- **防抖自动刷新**：用户调整任何管线参数滑块后，前端必须在 **500ms 防抖**后自动调用对应的后端接口，无需用户手动点击"生成"按钮。右侧预览区应实时显示最新参数的处理结果。

- **Loading 状态**：防抖期间及 API 请求进行中，对应按钮必须置灰并显示"处理中…"，滑块不可重复触发。

- **保留手动按钮**：防抖不替代手动触发按钮。用户仍可通过点击按钮立即触发（无需等待防抖），手动点击时应取消当前防抖定时器并立即发起请求。

- **跳过初次挂载**：组件首次渲染时不得自动触发 API（避免默认参数空跑）。

- **上传重置**：上传新图片时，清空所有历史预览结果，参数滑块恢复默认值。

**实现模式**（React + TypeScript）：

```
useRef 存储 params/imageId 避免闭包过期
useCallback 包装 generate() 核心逻辑
useEffect 监听 [params, imageId] 触发 500ms 防抖
mountedRef 跳过首次渲染
```
## 9. 前端 UI 防破坏与增量开发协议 (UI Non-Destructive Extension)
当需要在现有的 React 界面上添加新功能（如新的参数面板、新按钮）时，必须绝对遵守以下规矩：
- **只增不改（Additive Only）**：严禁删除、注释或擅自重构现有的 UI 布局、CSS 类名或交互逻辑（例如画布的半透明叠加对比功能、Loading 状态）。
- **样式继承**：新增的 UI 组件必须无缝继承现有的 UI 风格（暗色主题、滑块样式、按钮排版）。仔细阅读现有代码的 CSS/Tailwind 结构，照猫画虎，禁止引入突兀的新风格。
- **模块化解耦**：如果 `App.tsx` 变得冗长，不允许直接在里面疯狂堆砌代码。应该将现有的功能块或新功能块安全地抽离到 `src/components/` 下的子组件中，并通过 props 或 Zustand store 传递状态，确保重构后的页面效果与重构前 **100% 一致**。
- **实时预览图**: 当用户调整任意参数后，当前步骤处理结果的预览图必须实施更新为使用最新参数处理的结果
- **统一预览切换逻辑 (Unified Preview Toggle)**：管线中**每一个步骤**（含已完成的线稿提取/连通修复/SVG 生成，以及未来待建的降噪/SAM 分割等），都必须实现 before/after 预览切换功能。具体规则：
  - 每个步骤在 `STEP_META` 中配置 `showToggle: true`（切换开关）和 `afterLabel`（结果标签）
  - 每个步骤在 `viewLabel` 计算中提供 toggle 关闭/开启两种状态的描述文字
  - **非图片叠加类结果**（如 SVG 下载卡片）必须通过 `forceLayout` prop 指定合适的并排布局（如 `vertical`），避免结果被错误地半透明叠加在原图上
  - toggle 状态（`stepCompareOn`）按步骤独立存储，步骤切换时不丢失

## 10. 项目记录与汇报体系 (Project Logging & Reporting)

**Agent 必须在每次开发会话中主动维护项目文档，确保进度可追溯、问题可复盘、文档不矛盾。**

### 10.1 进度更新规范 (`PROGRESS.md`)

- **触发时机**（以下任一事件发生后必须更新）：
  - 完成一个管线步骤（前端 + 后端）
  - 新建一个组件、脚本、或工具
  - 修复一个 bug（从 UNSOLVED 移入 PAST 后）
  - 完成一轮集成测试
  - Docker 环境发生变更
- **更新内容（按需，非全部）**：
  - 头部日期 + Git HEAD 描述（每次必更）
  - 各层级（前端/后端/dev_tools/Docker）的实现状态矩阵
  - 管线 7 步的 API→Service→Utils→前端→测试 五层覆盖表
  - 前端组件的完成度标注
  - 「架构变更记录」区块：新增条目（日期 + 症状 + 根因 + 修复 + 详见 PAST_ISSUES 案例号）
  - 下一步计划
- **格式**：Markdown 表格 + 状态标记（✅ / 🟡 / ❌）
- **禁止**：连续多次代码变更后只更新一次（每次变更都应触发更新）；更新内容与代码实际状态不符

### 10.2 问题记录（双文档体系）

问题记录分为两个独立文档，Agent 必须根据问题状态写入正确的文档：

#### `UNSOLVED_ISSUES.md` — 待解决问题

- **用途**：记录尚未修复的问题
- **触发时机**：发现 bug、用户报告异常行为、诊断确认根因但尚未执行修复
- **必须包含**：
  1. 症状：用户看到了什么异常现象
  2. 诊断：用哪些工具/命令定位的，排除了哪些可能性
  3. 根因：真正导致问题的代码/配置
  4. 修复方案：计划如何修复（待用户审批）
- **禁止**：将已修复的问题留在此文件

#### `PAST_ISSUES.md` — 已解决问题

- **用途**：记录已修复并验证的历史问题，作为项目知识沉淀
- **触发时机**：问题修复完成并验证通过后，从 `UNSOLVED_ISSUES.md` 移入
- **必须包含五段结构**：
  1. 症状
  2. 诊断
  3. 根因
  4. 修复：具体改了什么（文件+行号+变更内容）
  5. 验证：如何确认修好了（可观测的证据，不是"看起来好了"）
- **格式**：每个案例独立编号，按时间倒序排列

#### 转移流程

问题修复并验证通过后：
1. 在 `UNSOLVED_ISSUES.md` 中删除该案例
2. 在 `PAST_ISSUES.md` 末尾追加完整记录（含修复+验证）
3. 同步更新 `PROGRESS.md` 和其他引用文档（见 §10.4）

#### 通用规则

- **禁止**：跳过诊断直接写"修好了"、修复后不写验证方式、堆积多个 bug 到一条记录

### 10.3 错误汇报规范 (Error Reporting Protocol)

**当 Agent 在开发过程中遇到错误时，必须按以下格式向用户汇报，禁止含糊带过。**

#### 汇报模板

```
【错误类型】：构建失败 / 运行时异常 / API 报错 / 依赖缺失 / 其他
【触发操作】：执行了什么命令 / 调用了哪个接口 / 修改了哪个文件
【完整报错】：原始错误消息（含 traceback，不可截断或改写）
【影响范围】：哪些功能受影响，是否阻断后续步骤
【已排除的可能性】：列出已验证不是原因的假设（至少 2 条）
【疑似根因】：按可能性排序，列出 2-3 个最可能的原因
【建议修复方案】：每个疑似根因对应的具体修复步骤
【需要用户决策】：是 / 否（如果有需要用户审批的修改，明确标出）
```

#### 强制规则

- **禁止改写报错**：必须粘贴原始错误消息，不允许用自己的话概括或截断 traceback
- **禁止跳过诊断**：不允许说"看起来像是 X 问题，我试一下修复"就动手。必须先列出可能原因并附排除逻辑
- **连续失败上限**：同一问题的修复尝试不得超过 2 次。第 2 次失败后必须停止，重新诊断，并向用户汇报当前僵局
- **Docker 相关错误**：涉及 `Dockerfile.dev`、`docker-compose.yml`、`requirements.txt` 的错误，必须先汇报，等待用户审批后再修改
- **依赖缺失**：缺少 pip/npm 包时，先汇报缺什么、为什么需要、是否影响现有功能，由用户决定安装方式

### 10.4 文档同步规范 (Documentation Sync Protocol)

**代码变更后，相关文档必须同步更新。以下为强制性文档同步清单。**

#### 同步矩阵

| 代码变更类型 | 必须同步的文档 |
|-------------|--------------|
| 新增/修改 API 端点 | `API_CONTRACT.md`（请求/响应格式、字段类型） |
| 新增/修改管线步骤 | `PROGRESS.md`（管线矩阵）、`API_CONTRACT.md`、`README.md`（目录树） |
| 修复 bug | `PAST_ISSUES.md`（追加案例）、`PROGRESS.md`（架构变更记录 + 日期 + 组件状态） |
| 新增引擎/工具 | `AI_RULES.md` §3 目录树、`PROGRESS.md`、skill 文档 |
| 重命名/删除文件 | `AI_RULES.md` §3 目录树、`API_CONTRACT.md`（如有路由变更）、skill 文档中所有引用路径 |
| Docker 环境变更 | `PROGRESS.md`、skill 文档 Docker 部分、`README.md` |
| 参数/配置变更 | `API_CONTRACT.md`、前端 `types/index.ts`、skill 文档参数表 |

#### 同步验证清单（每次变更后自查）

| # | 检查项 | 方法 |
|---|--------|------|
| S1 | 文档间引用一致 | `grep -rn "旧文件名\|旧案例号" *.md` — 不应出现指向已删除/重命名文件的链接 |
| S2 | PROGRESS.md 日期最新 | 头部日期必须 ≥ 最后一次代码变更日期 |
| S3 | PAST_ISSUES 案例完整 | 每个案例包含五段结构，验证方式可复现 |
| S4 | skill 文档引用更新 | 技能中引用的案例号、文件名、行号与当前代码一致 |
| S5 | API_CONTRACT.md 与代码一致 | 端点路径、请求字段名/类型、响应字段名与实际 Pydantic 模型匹配 |

#### 强制规则
- **代码改完，文档立刻跟上**（bug 修复）：修复验证通过后 5 分钟内完成文档同步，不允许说"等下次一起更新"
- **禁止部分更新**：只改 `PAST_ISSUES.md` 而不同步 `PROGRESS.md` / skill / references 会导致后续会话读取过时信息，视为违规
