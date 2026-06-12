# 🐳 Docker 基础设施标准指引 (Docker Infra Guide)

**【最高警告】本文件定义了 `client_app` 的容器化基建标准。所有涉及 Dockerfile、docker-compose.yml 的修改，必须以本文件为唯一真理。严禁 AI 代理擅自降级配置或使用碎片化挂载！**

---

## 1. 核心挂载原则 (Volume Mounting Strategies)

### 1.1 拒绝碎片化挂载 (Anti-Fragmentation)
- **禁止**：严禁在 `docker-compose.yml` 中单独挂载某个文件或子目录（例如 `- ./frontend/src:/app/src` 或 `- ./frontend/package.json:/app/package.json`）。
- **标准**：必须使用**整目录映射**（例如 `- ./frontend:/app`）。以确保新增的配置文件、静态资源或子目录能瞬间热重载至容器内部，避免因为漏写映射导致的模块丢失。

### 1.2 Node.js 环境隔离 (Anonymous Volumes)
- **标准**：前端服务必须强制配置匿名卷 `- /app/node_modules`。
- **原因**：防止 Windows/WSL 宿主机的文件系统权限或宿主机本地生成的 npm 缓存覆盖 Linux 容器内部的原生编译包（如 esbuild 等 C++ 扩展包），杜绝跨系统运行报错。

---

## 2. 深度学习模型持久化 (Model Caching Protocol)

本项目集成了 `transformers`、`depth-anything-v2` 等大模型库。为了避免每次容器重建（Rebuild）导致数 GB 的预训练权重丢失，必须强制执行缓存持久化。

- **环境变量配置**：后端服务必须配置 `HF_HOME=/root/.cache/huggingface`。
- **持久化数据卷配置**：必须在后端服务的 `volumes` 列表，以及文件最底部的全局 `volumes` 声明中，配置独立的数据卷映射：
  ```yaml
  volumes:
    - hf_model_cache:/root/.cache/huggingface
  ```

---

## 3. 标准 Compose 模板 (Golden Template)

未来的任何扩展，必须基于以下经过验证的模板进行增量修改：

```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    container_name: laser-frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app             # 整目录热重载
      - /app/node_modules           # 保护容器原生依赖
    depends_on:
      - backend
    networks:
      - laser-net

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    container_name: laser-backend
    ports:
      - "8080:8080"
    volumes:
      - ./backend:/app              # 整目录热重载
      - hf_model_cache:/root/.cache/huggingface  # 大模型永久缓存
    environment:
      - DEBUG_MODE=true
      - HF_HOME=/root/.cache/huggingface
    networks:
      - laser-net

networks:
  laser-net:
    driver: bridge

volumes:
  hf_model_cache:                   # 全局数据卷声明
```

## 4. 运行与构建禁令
- **禁止无意义的 Rebuild**：业务代码（`.py`, `.ts`）的修改会通过 Volume 自动热重载，严禁为了让业务代码生效而执行 `docker-compose build`。
- **依赖变更才 Rebuild**：只有当修改了 `requirements.txt` 或 `package.json`，或是 `Dockerfile.dev` 本身时，才允许执行 `docker-compose up -d --build [服务名]`。