# AI 代理开发行为准则 (AI Constitution)

**【全局指令】在本项目中，你不仅是程序员，更是严格遵守架构红线的系统架构师。编写任何代码前，必须优先查阅并遵守以下 6 大准则。**

## 1. 顶层架构隔离（一国两制）
本项目严格分为两个绝对隔离的顶级目录，严禁跨界污染：
- **`client_app/` (面向用户的 Web 应用)**
  - 包含 `frontend/` (React+Vite) 和 `backend/` (FastAPI)。
  - **运行规则**：必须且只能通过 `docker-compose.yml` 运行，隔离在容器网络中。
- **`dev_tools/` (面向开发者的本地工具与训练脚本)**
  - 存放纯本地 Python 脚本（如 OpenCV GUI 标注、模型训练、基准测试）。
  - **运行规则**：直接在宿主机运行。**绝对禁止**引入 FastAPI/Flask 或监听任何端口。如需调用后端算法，通过 `sys.path` 动态引入 `client_app/backend` 的纯函数模块。

## 2. 文件与目录洁癖
- **禁止“就地拉屎”**：绝对禁止在项目根目录、`backend/` 顶层或 `frontend/` 顶层生成任何临时的 `.json`, `.txt`, `.log` 文件。
- **数据归宿**：
  - 应用运行生成的临时图片、测试结果必须输出到 `client_app/backend/outputs/`。
  - 训练得到的数据（如 `training_results.json`, `.pkl` 模型）必须输出到 `dev_tools/data/`。
  - 基准测试（Benchmarks）和对比用的静态 HTML，必须存放在 `dev_tools/benchmarks/`。
- **目录架构：
  laser-engraver-v2/
  ├── AI_RULES.md                      # 【核心】AI 代理开发宪法（所有代码生成的最高准则）
  ├── README.md                        # 项目总览说明
  │
  ├── 📂 client_app/                   # 【战区一：面向用户的 Web 应用 (Docker 全隔离)】
  │   ├── docker-compose.yml           # 唯一运行环境：绑定前端 5173，后端 8080，挂载代码目录
  │   │
  │   ├── frontend/                    # ⚛️ 前端 (React 19 + Vite + TypeScript)
  │   │   ├── Dockerfile.dev           
  │   │   ├── package.json             # (前端唯一的 json 配置文件)
  │   │   ├── tsconfig.json
  │   │   └── src/
  │   │       ├── api/                 
  │   │       │   └── client.ts        # 💡 Axios 实例，内置《准则 4》要求的全局错误拦截器
  │   │       ├── components/          # React 组件库
  │   │       ├── store/               # Zustand 状态树
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
  │               ├── image_io.py      
  │               └── anime_edge.py    # ✨ 从旧项目提取的 lineart_anime 纯净代码放在这里！
  │
  └── 📂 dev_tools/                    # 【战区二：开发者本地工具 (纯本地运行，零 HTTP)】
      ├── scripts/                     # 🛠️ 工具脚本 (执行：python scripts/xxx.py)
      │   ├── layer_labeler.py         # OpenCV GUI 交互式标注工具 (直接在宿主机跑，调出窗口)
      │   ├── train_layer_params.py    # 机器学习训练脚本 (输出 .pkl)
      │   └── algorithm_sandbox.py     # 💡 算法沙盒测试：开发新算法时，先在这里只读取一张图跑通
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
              ├── compare.html         # 双击即看，脱离前后端独立存在
              └── (对比用的大量图片)

## 3. 网络与端口（绝对红线）
- **固定端口**：前端固定映射宿主机 `5173`，后端固定映射宿主机 `8080`，后端服务监听 IP 必须为 `0.0.0.0`。
- **禁止换端口**：遇到端口被占用或无法连通时，**严禁修改代码中的端口号！严禁编写自动寻找空闲端口的逻辑！** 你的任务是排查 Docker 容器状态，而不是改代码。

## 4. 日志与调试规范 (Error Logging)
为了实现高效 Debug，必须严格执行以下防御性编程规范：
- **后端日志**：**严禁使用内置的 `print()`**。必须统一使用 `loguru` 库。
- **异常捕获**：FastAPI 必须配置全局 Exception Handler；在 `try-except` 块中，必须使用 `logger.exception(e)` 打印包含完整 Traceback 的堆栈。
- **管线节点追踪 (CV项目特有)**：在所有图像处理函数的入口和出口，必须通过 `logger.info` 记录输入/输出图像的 `shape`、`dtype` 和处理耗时。
- **视觉断点 (Visual Logging)**：在复杂的矩阵运算或图形拓扑算法中，遇到逻辑难以排查时，必须提供利用 `cv2.imwrite` 将中间结果输出到本地目录以供核对的代码分支。
- **前端拦截**：必须在 Axios 中配置全局响应拦截器，控制台报错必须格式化打印：`[请求方法] [URL] | 状态码 | Payload | 后端详情`。

## 5. 核心算法开发法则 (Standalone Sandbox)
- **纯粹性**：所有 OpenCV 图像处理算法（如线稿提取、连通性修复）必须写成**纯函数**。输入输出应为标准的图像矩阵（`numpy.ndarray`），参数必须通过 Pydantic 模型传递。
- **剥离测试**：开发复杂算法时，先写独立的 `test_xxx.py` 脚本进行纯本地读取图片测试。本地测试完美通过后，再封装进 FastAPI 的路由层。
- **回退机制 (Fallback)**：使用高级或平台绑定的库（如 `cv2.ximgproc`）时，尽量提供纯 Python 或基础库的回退实现，防止环境崩溃。

## 6. 沟通与协作模式
- **谋定而后动**：每次实现新功能或重构前，先用文字描述你的架构设计方案，我同意后你再写代码。严禁一次性生成超过 300 行的巨型文件。
- **Debug "事不过三" 原则**：如果你的代码报错，我给出报错日志后，**不要立刻给我新的修复代码**。你必须先列出导致该报错的 3 个最可能原因及验证方式。等我验证确认后，你再写代码。如果连续 2 次修复失败，必须停止写代码，重新梳理逻辑。