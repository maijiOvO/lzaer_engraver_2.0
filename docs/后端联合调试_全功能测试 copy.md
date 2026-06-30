# 联合调试与全功能测试方案

> **版本:** 1.1 | **日期:** 2026-06-12 | **状态:** 已执行 | **更新:** 2026-06-30
>
> ⚠️ **环境说明**：本文档编写于 Linux (WSL2) Docker 环境时期。当前开发环境已迁移至 Windows 11 原生，但测试方案中的 API 端点、Pydantic 参数约束、超时策略、防死循环规则等核心测试逻辑**仍然完全适用**。仅需将 `docker compose` / `docker exec` 命令替换为 `uvicorn` / `npm run dev` 的本机启动方式，并将 `/tmp/` 临时目录改为 `%TEMP%`（Windows）。
>
> 本文档定义 `lzaer_engraver_2.0` 项目的完整测试方案，覆盖 Docker 环境验证、7 步管线端到端测试、前端用户行为模拟、鲁棒性压力测试、dev_tools 脚本补全、参考图对标。

**目标:** Docker 环境下的 7 步管线全链路端到端验证，覆盖单层/多层两种模式，模拟真实用户从上传到 SVG 下载的完整操作流程。当前亦支持本机 uvicorn + Vite 运行方式执行。

**架构:** Docker Compose 双容器（laser-frontend:5173 + laser-backend:8080）或本机 uvicorn + Vite，前端通过 Vite proxy 转发 `/api` 到后端。测试分 3 层：纯 Python 引擎验证 → 服务冒烟测试 → 浏览器全行为模拟。

**参考:** AI_RULES.md §1-10, DOCKER_INFRA_GUIDE.md, API_CONTRACT.md, PAST_ISSUES.md 案例5/8/11/12/13

**测试素材:** `dev_tools/test_imgs/` (97张), `dev_tools/references/single/` (6张剪纸参考), `dev_tools/references/multiple/` (2张纸雕 SVG 参考)

---

## 超时与防卡死策略 (Timeout & Anti-Stall Protocol)

所有可能卡死的操作必须带超时保护。

### 1.1 HTTP API 调用超时

每个 `curl` 命令强制加 `--max-time`，按步骤类型分级：

| 步骤 | 最大耗时 | curl 参数 |
|------|----------|-----------|
| upload / health / denoise / connectivity | 10s | `--max-time 10` |
| canny（1024px 内） | 15s | `--max-time 15` |
| canny（大图 >2000px） | 60s | `--max-time 60` |
| segment（首次深度估计+下载模型） | 300s | `--max-time 300` |
| segment（缓存命中） | 30s | `--max-time 30` |
| svg（单层） | 30s | `--max-time 30` |
| svg（多层） | 60s | `--max-time 60` |

超时后行为: 
- **不重试** — 记录到测试日志，标记 `TIMEOUT`，继续下一个测试
- 如果同一端点连续超时 2 次 → 判定该端点异常，跳过后续依赖该端点的测试，**不进入死循环**

### 1.2 纯 Python 脚本超时

```bash
# dev_tools 测试脚本一律带 timeout 命令
timeout 120 python3 dev_tools/scripts/test_canny.py
```
- canny 批量: 300s (97 张图)
- connectivity/SVG 单图: 30s
- segment 纯引擎: 120s

### 1.3 Docker 操作超时

```bash
# 启动超时
timeout 60 docker compose up -d

# 如果 60s 未启动 → 查看日志定位
docker compose logs --tail=50
# 常见卡死原因: 模型下载中(正常等待)、端口冲突(UAC弹窗)、依赖安装失败
```

### 1.4 防死循环规则

```
LEVEL 0: 单次超时 → 记录 → 跳过 → 下一个测试
LEVEL 1: 连续 2 次同端点超时 → 标记端点异常 → 跳过整类测试
LEVEL 2: 连续 3 次不同错误 → 停止所有测试 → 汇报用户
LEVEL 3: 同一修复尝试 2 次失败 → 停止 → 重新诊断
```

禁止行为:
- ❌ 同一命令不加修改重试
- ❌ 逐渐增加超时时间（"timeout 30 不行，试试 60"）
- ❌ 用 `while true` 等待容器启动（用固定次数的 `sleep 5` + 健康检查循环）

---

## 测试结果与日志存放规范

严格遵守 AI_RULES.md §3 "禁止就地拉屎"原则。每轮测试创建独立的时间戳会话目录。

### 2.1 目录结构

```
dev_tools/outputs/test_runs/
└── 2026-06-12_230000/              # ← 本轮测试会话
    ├── README.md                    # 本会话摘要（时间、测试范围、通过/失败统计）
    ├── api_results/                 # curl 原始响应 JSON
    │   ├── 01_upload.json
    │   ├── 02_canny_default.json
    │   ├── 02_canny_high.json
    │   ├── 03_denoise.json
    │   ├── 04_connectivity_g5.json
    │   ├── 04_connectivity_g0.json
    │   ├── 05_svg_t1.json
    │   ├── ...
    │   └── multi_layer/
    │       ├── segment.json
    │       ├── L0_canny.json
    │       └── ...
    ├── api_outputs/                 # 后端产出的图片/SVG 下载副本
    │   ├── <uuid>_canny.png
    │   ├── <uuid>_denoised.png
    │   ├── <uuid>_connected.png
    │   └── <uuid>.svg
    ├── dev_tools_outputs/           # dev_tools 脚本产出（批量测试）
    │   ├── canny/
    │   ├── connectivity/
    │   └── svg/
    ├── test_report.md               # 最终测试报告（自动生成）
    └── console.log                  # 完整终端输出日志（script 命令录制）
```

### 2.2 日志录制

```bash
# 整个测试会话用 script 录制
script -q -c "bash run_all_tests.sh" dev_tools/outputs/test_runs/2026-06-12_230000/console.log
```

### 2.3 Docker 后端日志

```bash
# 每个 Phase 开始前清除旧日志
docker compose exec backend truncate -s 0 /tmp/backend.log

# 每个 Phase 结束后保存
docker compose exec backend cat /tmp/backend.log > dev_tools/outputs/test_runs/2026-06-12_230000/backend_phase1.log
```

### 2.4 测试报告格式 (test_report.md)

```markdown
# 测试报告 — 2026-06-12 23:00

**环境:** Docker (laser-frontend + laser-backend), WSL
**测试图片:** 天津.jpg (1024x768)
**覆盖步骤:** upload, canny, denoise, connectivity, svg (单层) + segment (多层)

| 步骤 | 状态 | 耗时 | 备注 |
|------|------|------|------|
| 1. Docker启动 | ✅ | 12s | 两容器正常 |
| 2. Health | ✅ | 45ms | 200 |
| 3. Upload | ✅ | 230ms | image_id=xxx |
| 4. Canny low=50 | ✅ | 850ms | fg=12345px |
| 5. Canny low=80 | ✅ | 820ms | fg=14567px, MD5≠step4 ✅ |
| 6. Denoise | ✅ | 150ms | fg: 12345→11890 |
| 7. Connectivity g=5 | ✅ | 320ms | bridges=12 |
| 8. Connectivity g=0 | ✅ | 180ms | bridges=0 ✅ |
| 9. SVG t=1.0 | ✅ | 500ms | 45 paths |
| 10. Segment n=3 | ⏳ | — | 见 separate report |
| ... | ... | ... | ... |

**通过率:** 9/10 (90%)
**失败/超时:** 无
**新发现 Bug:** 见 UNSOLVED_ISSUES.md 案例16
```

---

## Phase 1: Docker 环境就绪验证

### Task 1.1: Docker 容器启动与健康检查

**目标:** 确认 docker-compose 能正常启动并前后端通信

**步骤:**

**Step 1: 启动容器**
```bash
cd /home/myron/lzaer_engraver_2.0/client_app && docker compose up -d
```
预期: 两个容器 running (laser-frontend, laser-backend)

**Step 2: 后端健康检查**
```bash
curl -s http://localhost:8080/api/health | python3 -m json.tool
```
预期: `{"status": "ok", "version": "2.0"}`

**Step 3: 前端 Vite 就绪**
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:5173
```
预期: `200`

**Step 4: Vite proxy 转发正常**
```bash
curl -s http://localhost:5173/api/health | python3 -m json.tool
```
预期: 同 Step 2（Vite proxy 正确转发到 backend:8080）

**Step 5: 验证关键依赖加载**
```bash
docker exec laser-backend python3 -c "import cv2, torch; print(f'OpenCV {cv2.__version__}, Torch {torch.__version__}')"
docker exec laser-backend python3 -c "from app.utils.canny_lineart import canny_lineart; print('canny_lineart OK')"
docker exec laser-backend python3 -c "from app.utils.sam_engine import run_sam_automatic; print('sam_engine OK')"
docker exec laser-backend python3 -c "from app.utils.depth_engine import estimate_depth; print('depth_engine OK')"
```
预期: 全部 OK，无 ImportError

**验证:** 所有 curl 返回预期值，所有 import 成功

---

### Task 1.2: 日志管道验证

**目标:** 确认 loguru 日志正常输出（非 print）

**步骤:**

**Step 1: 触发一个简单请求产生日志**
```bash
curl -s http://localhost:8080/health
```

**Step 2: 检查容器日志包含 loguru 格式**
```bash
docker logs laser-backend 2>&1 | tail -5
```
预期: 日志行包含时间戳、级别（INFO/DEBUG）、模块名等 loguru 格式标记

**Step 3: 确认无 print() 裸输出混入**
```bash
docker logs laser-backend 2>&1 | grep -v "|" | grep -v "INFO\|DEBUG\|WARNING\|ERROR\|SUCCESS" | head -5
```
预期: 无裸 print 输出（可能有一些第三方库的日志，但不应有项目代码的 print）

**验证:** 日志为 loguru 格式，无项目裸 print

---

## Phase 2: 全管线端到端 — 单层模式

### Task 2.1: 上传图片

**目标:** 验证 upload API 正常工作

**步骤:**

**Step 1: 选择测试图片并上传**
```bash
IMAGE=/home/myron/lzaer_engraver_2.0/dev_tools/test_imgs/天津.jpg
curl -s -X POST http://localhost:8080/upload \
  -F "file=@${IMAGE};type=image/jpeg" | python3 -m json.tool
```
预期: 返回 `{"image_id": "uuid-string", "width": ..., "height": ..., "original_url": "/outputs/uuid_original.jpg"}`，HTTP 200

**Step 2: 验证原图可访问**
```bash
# 用返回的 original_url
curl -s -o /dev/null -w "%{http_code} %{size_download}" http://localhost:8080/outputs/UUID_original.jpg
```
预期: `200` + 合理的文件大小

**验证:** image_id 非空，原图 URL 可访问

---

### Task 2.2: Canny 线稿提取（Step 4）

**目标:** 验证 canny_lineart 引擎和 API 正常，含参数变更→结果不同

**步骤:**

**Step 1: 默认参数调用**
```bash
ID=<上一步的image_id>
curl -s -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"low\": 50, \"high\": 150, \"smooth_level\": 0}" | python3 -m json.tool
```
预期: HTTP 200，`result_url` 指向 `_canny.png`，`processing_time_ms` > 0

**Step 2: 不同参数调用（验证结果不同）**
```bash
curl -s -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"low\": 80, \"high\": 200, \"smooth_level\": 1}" | python3 -m json.tool
```

**Step 3: MD5 对比确认不同参数产出不同结果**
```bash
curl -s http://localhost:8080/outputs/${ID}_canny.png -o /tmp/canny_default.png
# 重新调一次不同参数后
curl -s http://localhost:8080/outputs/${ID}_canny.png -o /tmp/canny_high.png
md5sum /tmp/canny_default.png /tmp/canny_high.png
```
预期: 两个 MD5 不同（证明参数生效）

**验证:** 两次不同参数调用产出不同的 `_canny.png`，result_url 可访问

---

### Task 2.3: Denoise 降噪（Step 5）

**目标:** 验证降噪链路：自动查找 `_canny.png` → 降噪 → 产出 `_denoised.png`

**步骤:**

**Step 1: 不传 layer_index（单层模式）**
```bash
curl -s -X POST http://localhost:8080/pipeline/denoise \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"min_component_area\": 4}" | python3 -m json.tool
```
预期: HTTP 200，`result_url` 指向 `_denoised.png`

**Step 2: 验证降噪后像素减少**
```bash
# 用 Python 快速对比
python3 -c "
import cv2
canny = cv2.imread('/tmp/canny_default.png', 0)
denoised_path = '$(curl -s http://localhost:8080/outputs/${ID}_denoised.png -o /tmp/denoised.png && echo /tmp/denoised.png)'
denoised = cv2.imread(denoised_path, 0)
print(f'Canny fg: {(canny > 0).sum()}, Denoised fg: {(denoised > 0).sum()}')
"
```
预期: `Denoised fg` ≤ `Canny fg`（降噪不应增加像素）

**验证:** HTTP 200，降噪后前景像素不增加

---

### Task 2.4: Connectivity 连通性修复（Step 6）

**目标:** 验证连通性修复链路

**步骤:**

**Step 1: 默认参数**
```bash
curl -s -X POST http://localhost:8080/pipeline/connectivity \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"gap_tolerance\": 5}" | python3 -m json.tool
```
预期: HTTP 200，`result_url` 指向 `_connected.png`，含 `bridges_built` 字段

**Step 2: gap_tolerance=1 验证（最小桥接）**
```bash
curl -s -X POST http://localhost:8080/pipeline/connectivity \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"gap_tolerance\": 1}" | python3 -m json.tool
```
预期: `bridges_built: 0` 或很小的值

**验证:** gap_tolerance 变化影响 bridges_built 数量

---

### Task 2.5: SVG 生成（Step 7）

**目标:** 验证 SVG 生成链路和输出格式

**步骤:**

**Step 1: 生成 SVG**
```bash
curl -s -X POST http://localhost:8080/pipeline/svg \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"simplify_tolerance\": 1.0}" | python3 -m json.tool
```
预期: HTTP 200，`svg_url` 指向 `.svg`，含 `total_paths`、`total_points`

**Step 2: 下载 SVG 并验证格式**
```bash
curl -s http://localhost:8080${SVG_URL} -o /tmp/test_output.svg
head -5 /tmp/test_output.svg
```
预期: 以 `<?xml` 或 `<svg` 开头，包含 `<path d=` 标签

**Step 3: 验证 simplify_tolerance 影响**
```bash
# 低容差 → 更多路径
curl -s -X POST ... -d '{"simplify_tolerance": 0.1}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'paths={d[\"total_paths\"]}')"
# 高容差 → 更少路径
curl -s -X POST ... -d '{"simplify_tolerance": 5.0}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'paths={d[\"total_paths\"]}')"
```
预期: low tolerance → high paths, high tolerance → low paths

**验证:** SVG 格式正确，路径数随容差单调变化

---

## Phase 3: 全管线端到端 — 多层模式

### Task 3.1: 深度分层（Step 2-3）

**目标:** 验证 depth_engine → structural_segmentation → SAM refine 完整链路

**步骤:**

**Step 1: 执行分段（首次运行，无缓存）**
```bash
ID=<image_id>
curl -s --max-time 300 -X POST http://localhost:8080/pipeline/segment \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"n_layers\": 3, \"sam_quality\": \"standard\", \"frame_width\": 50, \"min_island_area\": 100}" | python3 -m json.tool
```
预期: HTTP 200，返回 `layers` 数组（length=3），每层含 `mask_url`、`frame_url`；首次调用 processing_time_ms 应 > 3000（深度估计）

**Step 2: 缓存命中验证（二次调用）**
```bash
curl -s --max-time 30 -X POST http://localhost:8080/pipeline/segment \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"n_layers\": 3, \"sam_quality\": \"standard\"}" | python3 -m json.tool
```
预期: 二次调用 processing_time_ms < 500（深度图缓存命中）

**Step 3: sam_quality=draft 验证（跳过 SAM 精修）**
```bash
curl -s --max-time 120 -X POST http://localhost:8080/pipeline/segment \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"n_layers\": 3, \"sam_quality\": \"draft\"}" | python3 -m json.tool
```
预期: HTTP 200，processing_time_ms 明显小于 standard 模式

**Step 4: 验证 frame 宽度**
```bash
# 下载 frame_0.png 检查边框宽度
python3 -c "
import cv2, numpy as np
frame = cv2.imread('/tmp/frame_0.png', 0)
# 边框应该是白色(255)，检查左侧50px是否全白
left_strip = frame[:, :50]
print(f'Frame shape: {frame.shape}, left strip white px: {(left_strip == 255).sum()}/{left_strip.size}')
"
```
预期: 左边框 50px 范围像素基本全白

**验证:** 分段返回正确层数，缓存命中加速，draft 模式无 SAM

---

### Task 3.2: 多层 Canny → Denoise → Connectivity → SVG

**目标:** 逐层验证多层模式下每个步骤

**步骤:**

**Step 1: 多层 Canny（layer_index=0,1,2）**
```bash
for L in 0 1 2; do
  echo "=== Layer $L ==="
  curl -s -X POST http://localhost:8080/pipeline/canny \
    -H "Content-Type: application/json" \
    -d "{\"image_id\": \"$ID\", \"layer_index\": $L, \"low\": 50, \"high\": 150}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result_url'], d['processing_time_ms'], 'ms')"
done
```
预期: 3 层全部返回 `_canny_L0.png` / `_canny_L1.png` / `_canny_L2.png`

**Step 2: 多层 Denoise**
```bash
for L in 0 1 2; do
  curl -s -X POST http://localhost:8080/pipeline/denoise \
    -H "Content-Type: application/json" \
    -d "{\"image_id\": \"$ID\", \"layer_index\": $L, \"min_component_area\": 4}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result_url'])"
done
```
预期: 3 层全部 200，产出 `_denoised_L{N}.png`

**Step 3: 多层 Connectivity**
```bash
for L in 0 1 2; do
  curl -s -X POST http://localhost:8080/pipeline/connectivity \
    -H "Content-Type: application/json" \
    -d "{\"image_id\": \"$ID\", \"layer_index\": $L, \"gap_tolerance\": 5}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result_url'], 'bridges:', d.get('bridges_built', 'N/A'))"
done
```
预期: 3 层全部 200，产出 `_connected_L{N}.png`

**Step 4: 多层 SVG**
```bash
curl -s -X POST http://localhost:8080/pipeline/svg/multi-layer \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"n_layers\": 3, \"simplify_tolerance\": 1.0}" | python3 -m json.tool
```
预期: HTTP 200，返回 `svg_url` 和每层统计

**验证:** 所有层级全部 200，文件名匹配 `_L{N}` 模式

---

## Phase 4: 前端全行为模拟测试

### 模拟用户行为的方法论

本测试方案运行于 CLI 环境，模拟用户行为采用三层递进方案：

**L1: API 级模拟（curl）** — 按用户操作序列发送 HTTP 请求，模拟"用户调整滑块→防抖→API调用"。Phase 2-3 已完成。

**L2: 页面状态采样（curl + 解析）** — 用 `curl http://localhost:5173` 获取前端 HTML/JS bundle，验证关键 DOM 元素存在（Uploader 组件、Canvas 容器、ControlPanel 面板）。这是对 L1 的补充——确认前端代码能正确渲染，不只是后端 API 能跑。

**L3: 用户操作序列重放（curl POST 模拟前端行为）** — 前端本质上就是发 HTTP 请求。模拟以下用户操作模式：

```
用户行为                          → 等价 HTTP 操作
─────────────────────────────────────────────────────────
拖入图片上传                      → curl POST /upload -F file=@test.jpg
调整 Canny low 滑块 50→80        → curl POST /pipeline/canny -d '{"low":80,...}'
   500ms 内再调 low 80→55         → 只发最后一次（防抖验证）
切换到多层模式                    → curl POST /pipeline/segment
点击"Layer 1"                     → curl POST /pipeline/canny -d '{"layer_index":1,...}'
点击"生成 SVG"                    → curl POST /pipeline/svg
上传新图（重置管线）              → curl POST /upload + 检查旧 state 是否清空
后端崩溃后恢复                    → docker stop → curl(预期失败) → docker start → curl(预期恢复)
```

**L3.5: 浏览器手动验证（关键节点）** — 对以下不可自动化的场景，由用户在浏览器中手动验证：
- Canvas 视觉效果（线稿叠加原图的半透明渲染）
- SVG 下载卡片是否正确显示
- Loading 动画和 ProgressBar 进度
- 暗色主题 / 响应式布局

### Task 4.1: 前端静态加载验证（L2 采样）

**目标:** 确认前端页面可正常加载，核心组件渲染无报错

**步骤:**

**Step 1: 打开前端页面**
```bash
# 获取页面标题确认加载
curl -s http://localhost:5173 | grep -o '<title>.*</title>'
```
预期: 页面标题非空

**Step 2: 检查浏览器 Console 无 JS 错误**
在浏览器打开 http://localhost:5173 → F12 DevTools → Console 面板
预期: 无红色报错（允许 React DevTools 警告）

**Step 3: 检查 Network 面板无 404/500**
DevTools → Network 面板，刷新页面
预期: 所有请求 200 或 304

**验证:** 页面渲染完整（Uploader + Canvas + ControlPanel 可见）

---

### Task 4.2: 上传→预览 全流程（单层模式）

**目标:** 模拟用户拖拽上传 → 参数调整 → 实时预览

**步骤:**

**Step 1: 上传图片**
- 点击 Uploader 区域或拖入 `dev_tools/test_imgs/天津.jpg`
- 预期: Canvas 显示原图，ControlPanel 激活

**Step 2: 调整 Canny 滑块**
- 拖动 `low` 滑块 50→80
- 预期: 500ms 防抖后自动调用 `/pipeline/canny`，Canvas 预览更新为新线稿

**Step 3: 验证 cache-busting**
- DevTools Network 面板 → 筛选 `_canny.png`
- 多次拖动滑块 → 每次应产生新的 GET 请求（带 `?v=N` 参数）
- 预期: 每次参数变化都有新请求，不是 304 缓存

**Step 4: 切换 "查看原图" toggle**
- 点击 "查看原图" 开关
- 预期: Canvas 在半透明叠加和原图之间切换

**Step 5: 走完剩余步骤**
- 依次调整 Denoise / Connectivity / SVG 参数
- 每步预期: 500ms 防抖自动触发，预览更新

**验证:** 7 步全部有预览响应，cache-busting 生效

---

### Task 4.3: 多层模式全流程

**目标:** 模拟用户从选择多层模式到 SVG 下载

**步骤:**

**Step 1: 切换到多层模式**
- ControlPanel 顶部切换到 "多层模式"
- 上传图片
- 预期: SAM 分段参数面板出现（n_layers, sam_quality 等）

**Step 2: 触发分段**
- 调整 n_layers=3，点击"生成分段"
- 预期: ProgressBar 显示分段进度，完成后 Canvas 显示分层叠加图

**Step 3: 逐层调整参数**
- 在 Layer 选择器中切换到 Layer 0
- 调整 Canny low/high → 预览更新（仅 Layer 0 的线稿）
- 切换到 Layer 1 → 重复
- 预期: 每层独立预览，切换不丢失其他层状态

**Step 4: 多层 SVG 下载**
- 点击 "生成 SVG"
- 预期: 弹出下载或显示 SVG 下载卡片

**验证:** 多层模式下每层独立参数 + 独立预览，SVG 正常生成

---

### Task 4.4: 边界情况与错误处理

**目标:** 验证异常场景下的前端行为

**步骤:**

**Step 1: 后端断连恢复**
- `docker stop laser-backend`
- 调整 Canny 滑块 → 预期: 前端显示错误提示（不崩溃，不白屏）
- `docker start laser-backend`
- 再次调整 → 预期: 恢复正常

**Step 2: 上传非图片文件**
- 尝试上传 `.txt` 文件
- 预期: Uploader 拒绝或后端返回 400 并显示友好提示

**Step 3: 快速连续调整参数**
- 快速拖动 Canny 滑块 3-5 次
- 预期: 防抖机制生效，仅最后一次触发 API（Network 面板验证）

**Step 4: 上传新图重置管线**
- 上传一张新图片
- 预期: 历史预览清空，参数滑块恢复默认值

**验证:** 异常场景不崩溃，防抖/重置行为正确

---

## Phase 4.5: 鲁棒性压力测试

### 设计原则

鲁棒性 = 程序在非理想条件下的存活能力。测试覆盖四个维度：

| 维度 | 含义 | 测试方法 |
|------|------|----------|
| 输入鲁棒性 | 非法/极端参数不崩溃 | 边界值、类型错误、空值 |
| 并发鲁棒性 | 多请求同时到达不乱序 | 并行 curl 竞态测试 |
| 恢复鲁棒性 | 故障后能自动恢复 | kill→restart→retry 循环 |
| 资源鲁棒性 | 大图/OOM 不崩溃 | 97 张 test_imgs 批量扫描 |

### Task 4.5.1: 输入鲁棒性 — 非法参数

**目标:** 验证所有端点对非法输入的响应是 422/400 而非 500 崩溃

**步骤:**

```bash
ID=<valid_id>
SESSION_DIR=dev_tools/outputs/test_runs/2026-06-12_230000

# 1. 缺失必填字段
curl -s --max-time 10 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"low": 50}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('missing image_id:', d.get('detail', 'NO DETAIL'))"

# 2. 字段类型错误（low 传字符串）
curl -s --max-time 10 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"image_id": "'$ID'", "low": "not_a_number"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('bad type:', d.get('detail', 'NO DETAIL'))"

# 3. 超出范围值
curl -s --max-time 10 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"image_id": "'$ID'", "low": -100, "high": 999}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('out of range:', d.get('detail', d.get('result_url', 'NO ERROR')))"

# 4. 不存在的 image_id
curl -s --max-time 10 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"image_id": "nonexistent-12345", "low": 50, "high": 150}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('bad id:', d.get('detail', 'NO DETAIL'))"

# 5. 空 JSON body
curl -s --max-time 10 -X POST http://localhost:8080/pipeline/denoise \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('empty body:', d.get('detail', 'NO DETAIL'))"

# 6. segment: n_layers 极端值
curl -s --max-time 30 -X POST http://localhost:8080/pipeline/segment \
  -H "Content-Type: application/json" \
  -d '{"image_id": "'$ID'", "n_layers": 100}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('n_layers=100:', d.get('detail', d.get('layers', 'NO ERROR')))"

# 7. svg: simplify_tolerance 负数
curl -s --max-time 30 -X POST http://localhost:8080/pipeline/svg \
  -H "Content-Type: application/json" \
  -d '{"image_id": "'$ID'", "simplify_tolerance": -5.0}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('neg tolerance:', d.get('detail', 'NO DETAIL'))"
```

**预期结果（鲁棒性标准）:**
- 缺失必填字段 → 422 (Pydantic validation error)，非 500
- 类型错误 → 422，非 500
- 范围超限 → 422 或 200（后端自行 clamp），非 500
- 不存在 ID → 404 或 400（"image not found"），非 500
- 空 body → 422，非 500
- 极端 n_layers → 422 或 clamp 到有效范围，非 500 或 OOM
- 负数容差 → 422，非 500

**核心断言:** **任何非法输入不得触发 500 Internal Server Error**。500 说明后端未做输入校验就传给了算法层。

### Task 4.5.2: 并发鲁棒性 — 竞态请求

**目标:** 验证快速连续请求不会导致状态错乱

**步骤:**

```bash
# 同时发起 3 个不同参数的 canny 请求
curl -s --max-time 15 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"image_id": "'$ID'", "low": 30}' > /tmp/race_1.json &
curl -s --max-time 15 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"image_id": "'$ID'", "low": 50}' > /tmp/race_2.json &
curl -s --max-time 15 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"image_id": "'$ID'", "low": 80}' > /tmp/race_3.json &
wait

# 验证三个请求都成功
for f in /tmp/race_1.json /tmp/race_2.json /tmp/race_3.json; do
  python3 -c "import json; d=json.load(open('$f')); print(d.get('result_url', d.get('detail', 'FAIL')))"
done
```
预期: 3 个请求全部 200（uvicorn 默认单 worker，串行处理是正常行为；关键是不崩溃不丢请求）

### Task 4.5.3: 恢复鲁棒性 — 故障恢复

**目标:** 验证后端崩溃后重启，前端能恢复通信

**步骤:**

```bash
# 1. 正常请求确认连通
curl -s --max-time 10 http://localhost:8080/api/health && echo "PRE-CRASH: OK"

# 2. 模拟崩溃
docker stop laser-backend
sleep 2

# 3. 崩溃后请求（预期失败）
curl -s --max-time 5 http://localhost:8080/api/health && echo "DURING-CRASH: unexpected OK" || echo "DURING-CRASH: connection refused (expected)"

# 4. 恢复
docker start laser-backend
echo "Waiting for recovery..."
for i in $(seq 1 20); do
  if curl -s --max-time 3 http://localhost:8080/api/health > /dev/null 2>&1; then
    echo "RECOVERED after ${i}s"
    break
  fi
  sleep 1
done

# 5. 恢复后功能验证
ID=<saved_id>
curl -s --max-time 15 -X POST http://localhost:8080/pipeline/canny \
  -H "Content-Type: application/json" \
  -d "{\"image_id\": \"$ID\", \"low\": 50, \"high\": 150}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('POST-CRASH canny:', d.get('result_url', 'FAIL'))"
```
预期: 崩溃后拒绝连接 → 重启后 20s 内恢复 → 恢复后功能正常

### Task 4.5.4: 资源鲁棒性 — 批量图片扫描

**目标:** 对 dev_tools/test_imgs/ 中所有 97 张图片（含大图 5472×3078）跑 Canny，验证无 OOM 无崩溃

**步骤:**

```bash
# 用 timeout 保护，预期完成时间 < 5min
timeout 300 python3 -c "
import sys, time, json, os
sys.path.insert(0, 'client_app/backend')
from pathlib import Path
import cv2
from app.utils.canny_lineart import canny_lineart

test_dir = Path('dev_tools/test_imgs')
results = []
for img_path in sorted(test_dir.glob('*.jpg')) + sorted(test_dir.glob('*.png')):
    t0 = time.perf_counter()
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            results.append({'file': img_path.name, 'status': 'SKIP_READ_ERROR'})
            continue
        h, w = img.shape[:2]
        # 大图缩放保护
        if max(h, w) > 1600:
            scale = 1600 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        result = canny_lineart(img, low=50, high=150, smooth_level=0)
        elapsed = int((time.perf_counter() - t0) * 1000)
        fg = (result > 0).sum()
        results.append({'file': img_path.name, 'status': 'OK', 'shape': f'{h}x{w}', 'elapsed_ms': elapsed, 'fg_px': int(fg)})
        print(f'OK  {img_path.name:40s}  {elapsed:5d}ms  fg={fg:7d}')
    except Exception as e:
        results.append({'file': img_path.name, 'status': f'ERROR: {e}'})
        print(f'ERR {img_path.name:40s}  {e}')

# Summary
ok = [r for r in results if r['status'] == 'OK']
err = [r for r in results if r['status'] != 'OK']
print(f'\nTotal: {len(results)}  OK: {len(ok)}  Errors: {len(err)}')
if err:
    print('Errors:')
    for e in err:
        print(f'  {e[\"file\"]}: {e[\"status\"]}')
print(f'\nAvg time: {sum(r[\"elapsed_ms\"] for r in ok)//len(ok)}ms  Max time: {max(r[\"elapsed_ms\"] for r in ok)}ms')
# Save report
os.makedirs('dev_tools/outputs/test_runs/2026-06-12_230000/dev_tools_outputs', exist_ok=True)
with open('dev_tools/outputs/test_runs/2026-06-12_230000/dev_tools_outputs/batch_canny_report.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

**预期（鲁棒性标准）:**
- 通过率 ≥ 95%（允许少数图片因格式问题跳过，但不允许 OOM 崩溃）
- 大图（>4000px）自动缩放 ≤ 1600px，不触发 MemoryError
- 每张图 < 5s，总计 < 5min

---

## Phase 5: dev_tools 测试脚本补全

### Task 5.1: 补写 test_connectivity.py

**目标:** 独立的连通性修复纯 Python 测试脚本

**文件:** `dev_tools/scripts/test_connectivity.py`

**功能:**
- 读取 `dev_tools/outputs/denoise/` 中的 `_denoised.png`（或 `canny/` 回退）
- 调用 `repair_connectivity()` 以多种 `gap_tolerance`（1, 3, 5, 10）
- 对比桥接像素数，保存到 `dev_tools/outputs/connectivity/`
- 输出每组参数的 bridges_built 统计

**验证命令:**
```bash
cd /home/myron/lzaer_engraver_2.0 && timeout 30 python3 dev_tools/scripts/test_connectivity.py
```
预期: 4 组参数产出 4 张 `_connected_g{N}.png`，bridges 随 gap 增大而增多

---

### Task 5.2: 补写 test_svg.py

**目标:** 独立的 SVG 生成纯 Python 测试脚本

**文件:** `dev_tools/scripts/test_svg.py`

**功能:**
- 读取 `dev_tools/outputs/connectivity/` 中的 `_connected.png`（或 `denoise/` 回退）
- 调用 `generate_svg()` 以多种 `simplify_tolerance`（0.1, 0.5, 1.0, 3.0）
- 保存 SVG 到 `dev_tools/outputs/svg/`
- 输出每组容差的 path 数、points 数、文件大小

**验证命令:**
```bash
cd /home/myron/lzaer_engraver_2.0 && timeout 30 python3 dev_tools/scripts/test_svg.py
```
预期: 4 组容差产出 4 个 SVG，paths 随容差增大而减少

---

### Task 5.3: 批量管线测试

**目标:** 对 3-5 张代表性图片运行完整 canny→connectivity→svg 管线并汇总统计

**验证命令:**
```bash
python3 dev_tools/scripts/test_pipeline.py  # 现有功能应继续工作
```

---

## Phase 6: 参考图对标验证

### Task 6.1: 单层剪纸参考图对标

**目标:** 用 `dev_tools/references/single/` 中的 6 张参考剪纸，对比本项目 denoise+connectivity 产出的品质

**步骤:**

**Step 1: 选择一张参考图对应的原始照片（如有）或最接近的 test_img**

**Step 2: 运行完整单层管线**

**Step 3: 将 denoised/connected 输出与参考 PNG 做视觉对比（叠加或并排）**

**Step 4: 记录连通性修复的完整度（桥接是否过度/不足）**

**验证:** 产出 `_connected.png` 在视觉上与参考剪纸具有可比的结构完整性

---

### Task 6.2: 多层纸雕 SVG 参考图对标

**目标:** 用 `dev_tools/references/multiple/` 中的 2 张 CorelDRAW SVG（天津之眼2748 paths, 世纪钟1913 paths），对比本项目 SVG 产出的路径组织方式

**步骤:**

**Step 1: 读取参考 SVG 的路径结构（颜色编码: RED=结构 frame, BLACK=纹理 detail）**

**Step 2: 本项目输出的多层 SVG 与参考对比**
- 路径数量级是否在同一范围
- 结构层（外框）vs 纹理层的分离是否合理
- 贝塞尔曲线平滑度

**Step 3: 记录差异和可改进点**

**验证:** 本项目 SVG 的路径组织方式与参考 CorelDRAW SVG 可比

---

## Phase 7: 文档同步

### Task 7.1: PROGRESS.md 更新

**内容:**
- 日期更新为测试日期
- 管线矩阵中的 🟡 → ✅（通过测试的步骤）
- 前端组件状态更新（Canvas ControlPanel）
- 架构变更记录追加（如有新发现的问题）

### Task 7.2: UNSOLVED_ISSUES.md → PAST_ISSUES.md

**规则:** 测试中发现的任何新 bug → UNSOLVED_ISSUES.md；修复后 → PAST_ISSUES.md

---

## 执行顺序与依赖

```
Phase 1 (Docker) ──→ Phase 2 (单层API) ──→ Phase 3 (多层API)
                        │                       │
                        └──→ Phase 4 (前端UI) ←──┘
                                 │
Phase 4.5 (鲁棒性) ←─────────────┘
                                 │
Phase 5 (dev_tools脚本) ←───────┘
                                 │
Phase 6 (参考图对标) ←───────────┘
                                 │
Phase 7 (文档同步) ←─────────────┘
```

- Phase 2-4 可按顺序推进，也可先全走 API 再走 UI
- Phase 5 可在 Phase 2 验证通过后并行开发
- Phase 6 依赖 Phase 2-4 的输出数据
- Phase 7 由测试中发现的 bug 驱动

---

## 风险与回退

| 风险 | 概率 | 缓解 |
|------|------|------|
| Depth-Anything-V2 首次下载超时 | 中 | Phase 1.1 预先 import 触发下载 |
| SAM 模型 OOM（大图） | 低 | 已有 1600px 缩放保护 |
| WSL DrvFs 文件 I/O 错误 | 中 | 写入 `/tmp/` 而非 `/mnt/d/` 路径 |
| 前端 TypeScript 编译错误 | 低 | Phase 1 即可发现 |
| 浏览器缓存干扰测试 | 高 | Phase 4.2 Step 3 明确验证 cache-busting |

---

*文档结束。*
