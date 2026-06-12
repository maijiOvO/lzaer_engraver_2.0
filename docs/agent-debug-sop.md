# Agent 排错标准流程 (Debug SOP)

**适用对象**：AI Agent（Hermes、Claude Code 等），而非人类开发者。

**适用场景**：用户报告"功能不工作"但代码逻辑看起来正确时，Agent 必须执行本文档定义的标准诊断流程。

**核心原则**：不要猜。用工具验证每一层。数据流上的每一个环节都要独立证明它是对的。

---

## 1. 分层隔离：先确定问题在哪个边界

任何"预览不更新 / 结果不对 / 点了没反应"类 bug，第一步永远是分层：

```
浏览器缓存 ← 前端组件 ← API 调用 ← 后端逻辑 ← 算法引擎
```

**从最底层开始验证，逐层向上排除。**

### 1.1 后端独立验证

用 `curl` 直接打后端，绕过浏览器、绕过前端、绕过 Vite proxy：

```bash
# 验证后端健康
curl http://localhost:8080/api/health

# 用不同参数打同一个接口，对比结果是否真的不同
curl -s -X POST http://localhost:8080/api/pipeline/canny \
  -H "Content-Type: application/json" \
  -d '{"image_id":"xxx","low":10,"high":50,"smooth_level":2}' | python3 -m json.tool

# 下载两次结果到不同文件，用 md5sum 和像素对比验证
curl -o /tmp/test_a.png http://localhost:8080/outputs/xxx_canny.png
curl -o /tmp/test_b.png http://localhost:8080/outputs/xxx_canny.png
md5sum /tmp/test_a.png /tmp/test_b.png

# 如果 MD5 相同 → 后端确实没变（算法问题）
# 如果 MD5 不同 → 后端正常，问题在前端或浏览器缓存
```

**关键**：后端日志（`docker logs laser-backend`）必须显示每次请求收到了不同的参数。如果日志里参数确实在变，但输出一样 → 算法 bug。如果日志里参数在变、输出也在变 → 后端没问题，往上查。

### 1.2 API URL 稳定性检查

后端返回的 `result_url` 是否会因为参数不同而变化？

```python
# 常见反模式：固定文件名
result_path = f"{image_id}_canny.png"  # 永远返回同一个 URL

# 如果 URL 不变 → 浏览器缓存会吃掉更新 → 必须在前端做 cache-busting
```

### 1.3 浏览器缓存验证

打开 Chrome DevTools → Network 面板 → 筛选图片请求：

- 拖动参数滑块 → 看是否发出了新的 GET 请求
- 如果 Status 是 `304 Not Modified` → 浏览器缓存命中，没去服务器取
- 如果根本没有 GET 请求 → React 没重新渲染 `<img>`
- 如果 Status 是 `200` 但图没变 → 检查是不是同一个 URL

---

## 2. 预览不更新的标准诊断树

```
用户说"预览没变化"
│
├─ Step 1: 后端真的产出不同结果了吗？
│   ├─ curl 打两次不同参数 → md5sum 对比
│   ├─ MD5 相同 → 算法问题，检查参数是否真的传到了引擎函数
│   └─ MD5 不同 → 后端正常，进入 Step 2
│
├─ Step 2: 后端返回的 URL 变了吗？
│   ├─ URL 变了（如 /outputs/xxx_l10_h50.png） → 进入 Step 2a
│   ├─ URL 没变（如 /outputs/xxx_canny.png） → 进入 Step 2b
│   │
│   ├─ Step 2a: URL 变了但预览没变
│   │   └─ 检查 React：Canvas 组件的 <img> 标签有没有 key 属性？
│   │       ├─ 没有 → React 可能复用了 DOM，没触发重新加载
│   │       └─ 有 → 检查 Network 面板，浏览器是否发了 GET
│   │
│   └─ Step 2b: URL 没变 → 浏览器缓存问题（本节案）
│       └─ 修复模式：前端加 cache-busting（见 §3）
│
└─ Step 3: Network 面板确认
    ├─ 有新 GET 请求 → 图确实更新了 → 检查 CSS 是否遮挡或尺寸为 0
    └─ 没有 GET 请求 → React 渲染问题或缓存问题
```

---

## 3. Cache-Busting 标准修复模式

**适用条件**：后端返回的 URL 不随参数变化而变化（固定文件名）。

### 3.1 修复模板

**父组件（App.tsx 或管线容器）**：

```tsx
// 1. 加 version 状态
const [cannyVersion, setCannyVersion] = useState(0);

// 2. 每次拿到新结果后递增
const generate = useCallback(async () => {
  // ... 调 API ...
  setOverlayUrl(data.result_url);
  setCannyVersion(v => v + 1);  // ← 关键
}, []);

// 3. version 传给子组件
<Canvas
  imageUrl={imageUrl}
  overlayUrl={overlayUrl}
  version={cannyVersion}
/>
```

**子组件（Canvas.tsx）**：

```tsx
// img 标签加 key 和查询参数
<img
  src={`${overlayUrl}?v=${version}`}
  key={version}
  alt="Preview"
/>
```

### 3.2 为什么需要两层

| 机制 | 解决的问题 | 不加会怎样 |
|------|-----------|-----------|
| `?v=N` 查询参数 | 浏览器 HTTP 缓存 | 浏览器看到同一个 URL，直接返回缓存，不发网络请求 |
| `key={N}` 属性 | React DOM 复用 | React 看到同一个 src 字符串，不动 DOM，浏览器没机会重新加载 |

**两者必须同时加。只加一个不管用。**

### 3.3 扩展到多步骤管线

降噪、连通性修复、SVG 生成 —— 每个步骤各自维护一个 version 计数器：

```tsx
const [denoiseVersion, setDenoiseVersion] = useState(0);
const [connectivityVersion, setConnectivityVersion] = useState(0);
const [svgVersion, setSvgVersion] = useState(0);
```

各自的 `<img>` 标签各自用各自的 version。

---

## 4. 验证清单（修完后逐条过）

不依赖"看起来好了"，每一条都要有可观测的证据。

| # | 验证项 | 方法 | 通过标准 |
|---|--------|------|---------|
| 1 | 后端收到正确参数 | `docker logs -f laser-backend` | 日志显示每次拖动后 POST 到达，参数与滑块值一致 |
| 2 | 后端产出不同结果 | curl 两次不同参数，md5sum 对比 | MD5 不同 |
| 3 | 浏览器发出新图片请求 | Chrome DevTools → Network | 有新 GET 请求，Status 200（不是 304） |
| 4 | 图片肉眼可见变化 | 拖动 low 从 50→10 | 右侧预览线条明显变密 |
| 5 | 500ms 防抖正常 | 快速拖动滑块 3 秒后停住 | 只发出 1 次 API 请求 |
| 6 | 手动按钮仍有效 | 点"生成线稿" | 立即发请求（不等防抖），按钮变灰显示"处理中…" |
| 7 | 首次挂载不触发 | 刷新页面 | Network 面板没有自动发出的 pipeline 请求 |
| 8 | Loading 状态正常 | 拖动滑块 | 按钮置灰显示"处理中…"，滑块不可拖动 |
| 9 | 上传新图重置 | 上传新图片 | 预览清空，参数回默认值，version 归零 |

---

## 5. 禁止事项

| 禁止 | 原因 | 正确做法 |
|------|------|---------|
| 只看代码不动手测 | 代码看起来对不代表运行时对 | 用 curl / md5sum / DevTools 实际验证 |
| 改后端文件名来解决缓存 | 会产生大量文件，治标不治本 | 前端统一用 §3 的 cache-busting 模式 |
| 加 `Math.random()` 作为 version | 每次渲染都变，无限重新加载 | 只在拿到新结果时递增 |
| 只加 key 不加 ?v= | 浏览器缓存仍然生效 | 两者必须一起加（见 §3.2） |
| 只加 ?v= 不加 key | React 可能不重建 DOM | 两者必须一起加 |
| 猜测"应该是缓存问题" | 猜对一半没用，你没确认是哪一层 | 按 §2 诊断树逐层验证 |

---

## 6. 历史案例

### 案例 1：线稿预览不更新（2026-06-17）

- **症状**：拖动 Canny 参数滑块，右侧预览图没有变化
- **诊断**：curl 验证后端产出不同 MD5（13% 像素不同）→ 排除后端 → 发现 `result_url` 始终是 `/outputs/{id}_canny.png` → 确认浏览器缓存
- **根因**：Canvas.tsx 的 `<img>` 标签没有 key，URL 不带版本参数
- **修复**：App.tsx 加 `cannyVersion` 状态 + Canvas.tsx 加 `key={version}` 和 `?v={version}`

---

*本文档是给 Agent 看的排错行为准则。每修好一个新的 bug 类别，就把根因和修复模式追加到 §6。*
