# Linux Docker 一键部署

面向 **局域网个人偏好平台**：API + 前端静态资源 + 预测 Worker，通过 **Docker Compose** 一体启动。  
训练（GPU）仍建议在宿主机或临时 GPU 机器上跑，产物用候选 ZIP 导回平台。

---

## 1. 架构

```text
浏览器 ──► preference-api :8080
              │  FastAPI + frontend/dist
              │  SQLite / 训练包 / 候选  ──►  volume: PLATFORM_DATA_HOST
              │  默认模型（只读）        ──►  volume: MODEL_HOST_DIR → /data/models
              │  图片（只读）            ──►  volume: MEDIA_HOST_DIR  → /data/media
              │
         preference-worker
              同一 volumes + .env，消费 predict / export 任务
```

| 容器 | 作用 |
|------|------|
| `preference-api` | `uvicorn platform_app.main:app`，托管 SPA |
| `preference-worker` | `python -m platform_app.worker`，模型推理与训练包导出 |

**模型与媒体不打进镜像**，一律宿主机挂载，避免镜像膨胀与数据丢失。

---

## 2. 前置条件

| 项 | 建议 |
|----|------|
| OS | Linux x86_64（主测目标）；Docker Desktop for Mac/Win 可用但路径注意 |
| Docker | Engine 24+，Compose V2（`docker compose`） |
| 内存 | ≥ 8 GB（CPU 推理 + 首次拉 PyTorch/HF） |
| 磁盘 | ≥ 20 GB 空闲（镜像 + HF 缓存 + DB） |
| 网络 | 首次构建需访问 PyPI /（可选）Hugging Face 下载 `bert-base-chinese` |

检查：

```bash
docker version
docker compose version
```

---

## 3. 一键部署

在仓库根目录：

```bash
chmod +x scripts/deploy.sh scripts/deploy-down.sh

# 推荐：指定媒体目录与模型文件
export MEDIA_HOST_DIR=/path/to/your/images_or_downloads
export MODEL_FILE=/path/to/best_model.pth   # 将复制为 deploy_data/models/best_model.pth
export HOST_PORT=8080

./scripts/deploy.sh
```

脚本会：

1. 检查 Docker / Compose  
2. 创建 `deploy_data/{platform_data,models,media}`（可改路径）  
3. 若无 `.env`，从 `.env.docker.example` 生成并写入随机 `INGEST_API_KEY`  
4. 处理模型文件（`MODEL_FILE` 或目录内首个 `.pth` → `best_model.pth`）  
5. `docker compose build` + `up -d`  
6. 等待 `GET /health/ready`  
7. 打印访问地址与后续提示  

### 常用命令

```bash
./scripts/deploy.sh              # 构建并启动
./scripts/deploy.sh --skip-build # 不重建镜像
./scripts/deploy.sh --recreate   # 强制重建容器
./scripts/deploy.sh --build-only # 只构建
./scripts/deploy.sh --logs       # 跟踪日志
./scripts/deploy.sh --down       # 停止容器（保留宿主机数据）
./scripts/deploy-down.sh         # 同上
```

### 环境变量（部署脚本 / Compose）

| 变量 | 默认 | 说明 |
|------|------|------|
| `HOST_PORT` | `8080` | 宿主机端口 |
| `PLATFORM_DATA_HOST` | `./deploy_data/platform_data` | DB、训练包、候选、HF 缓存 |
| `MODEL_HOST_DIR` | `./deploy_data/models` | 挂载到 `/data/models` |
| `MEDIA_HOST_DIR` | `./deploy_data/media` | 挂载到 `/data/media` |
| `MODEL_FILE` | 空 | 复制到 `MODEL_HOST_DIR/best_model.pth` |
| `WORKER_BATCH_SIZE` | `8` | worker 批大小 |
| `WORKER_POLL_SECONDS` | `5` | 空闲轮询间隔 |
| `INGEST_API_KEY` | 自动生成 | 仅在**首次**生成 `.env` 时生效；也可手改 `.env` |
| `SKIP_BUILD` | `0` | `1` 跳过 build |
| `COMPOSE_PROJECT_NAME` | `preference-platform` | Compose 项目名 |

容器内应用变量见 `.env` / `.env.docker.example`（`PLATFORM_DATA_DIR`、`ALLOWED_MEDIA_ROOTS`、`DEFAULT_MODEL_PATH` 等）。

---

## 4. 首次使用

1. 浏览器打开 `http://<服务器IP>:8080/`  
2. **创建管理员**（密码 ≥ 12 位，仅一次）  
3. 若已挂载 `best_model.pth`：启动时自动登记 `default-env` 为 active  
4. 打开 **任务** 页查看 worker 心跳与打分进度  
5. 历史数据在容器内补预测（可选）：

```bash
docker compose exec worker python -m platform_app.requeue_predictions
```

6. **待筛选** 做人工标注；**内容库** 浏览全量  

---

## 5. 爬虫对接

爬虫与平台密钥一致：

```bash
export PLATFORM_API_URL=http://<服务器局域网IP>:8080
export INGEST_API_KEY=<与 .env 中相同>
```

**关键**：爬虫写入的图片绝对路径，必须落在 **宿主机 `MEDIA_HOST_DIR` 目录树内**，因为容器只读挂载该目录为 `/data/media`。

推荐做法：

- 爬虫下载目录 = `MEDIA_HOST_DIR`（或子目录）  
- 入库时 `source_path` 使用容器内路径，或  
- 更简单：宿主机路径与容器路径一致（例如都用 `/data/media/...`，在宿主机也 bind 到同一路径）

若爬虫仍写 Windows/旧路径，需要迁移文件到 `MEDIA_HOST_DIR` 并改库内路径，或增加额外 volume（改 `docker-compose.yml` 的 `volumes` + `ALLOWED_MEDIA_ROOTS`）。

---

## 6. 目录与数据

| 宿主机（默认） | 容器 | 内容 |
|----------------|------|------|
| `deploy_data/platform_data` | `/data/platform_data` | `platform.db`、候选、训练 ZIP、HF 缓存 |
| `deploy_data/models` | `/data/models` | `best_model.pth` 等 |
| `deploy_data/media` | `/data/media` | 原图（只读） |

备份建议：

```bash
# 停写更安全（可选）
./scripts/deploy.sh --down
tar czf backup-$(date +%Y%m%d).tgz deploy_data/platform_data
./scripts/deploy.sh --skip-build
```

注意：SQLite 若存在 `platform.db-wal` / `platform.db-shm`，请一并备份。

---

## 7. 升级代码

```bash
git pull
./scripts/deploy.sh --recreate
# 或
docker compose build && docker compose up -d
```

前端已在镜像构建阶段 `npm run build`，无需在服务器单独装 Node（除非改前端后要重建镜像）。

---

## 8. 运维速查

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f worker
curl -sS http://127.0.0.1:8080/health/live
curl -sS http://127.0.0.1:8080/health/ready
curl -sS http://127.0.0.1:8080/health/worker

# 进入容器
docker compose exec api bash
docker compose exec worker python -m platform_app.requeue_predictions
```

---

## 9. 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `deploy.sh` 报 port in use | 端口占用 | `HOST_PORT=8081 ./scripts/deploy.sh` |
| health 超时 | 构建/依赖/启动失败 | `docker compose logs api` |
| 前端空白 | 镜像未含 dist | 确认 Dockerfile 多阶段成功；重建 |
| 全是「待预测」 | 无模型 / worker 挂 / 无 job | 检查 `best_model.pth`、worker 日志、`requeue_predictions` |
| 入库 403 图片路径 | 不在 allowlist / 未挂载 | 扩展 volume + `ALLOWED_MEDIA_ROOTS` |
| 入库 401 | 密钥不一致 | 对齐 `.env` 与爬虫 `INGEST_API_KEY` |
| worker OOM | 批太大 / 内存不足 | 降低 `WORKER_BATCH_SIZE`，加大机器内存 |
| HF 下载失败 | 无外网 | 预置缓存到 `platform_data/hf_cache`，或配置镜像源 |
| 权限错误 | 目录属主 | `chown -R` 部署用户；SELinux 见下 |
| SELinux 拒挂载 | 强制策略 | 卷加 `:z` 或按发行版文档配置 |

### SELinux（RHEL/CentOS/Fedora）

若挂载被拒，可在 `docker-compose.yml` 的 volume 后加 `:z`（共享标签），或按站点策略调整。改后需 `compose up -d`。

---

## 10. 安全

1. **默认按局域网设计**：勿将 `0.0.0.0:8080` 直接暴露公网。  
2. 外网访问请加 **反向代理 + HTTPS**，并设 `COOKIE_SECURE=true`。  
3. 立刻修改/保管 `.env` 中的 `INGEST_API_KEY` 与管理员密码。  
4. 媒体卷只读挂载；平台数据卷需可写。  
5. 不要把 `.env`、模型、数据库提交进 Git。

### Nginx 反代示例（可选）

```nginx
server {
    listen 443 ssl;
    server_name preference.example.com;
    # ssl_certificate ...;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 600m;  # 候选包上传
    }
}
```

---

## 11. 从 Windows 本机迁到 Linux

1. 复制 `platform_data/`（或 Windows 上的 `platform_data`）→ `PLATFORM_DATA_HOST`  
2. 复制模型 `.pth` → `MODEL_HOST_DIR/best_model.pth`  
3. 复制/挂载图片树 → `MEDIA_HOST_DIR`（**路径若变化**，库内 `media_assets.source_path` 需批量改写或保持路径一致）  
4. `./scripts/deploy.sh`  
5. 核对 `.env` 密钥与 `ALLOWED_MEDIA_ROOTS`  

---

## 12. 明确不做 / 后续可选项

| 项 | 说明 |
|----|------|
| 默认 GPU 镜像 | 当前为 **CPU** PyTorch；GPU 需自定义 Dockerfile + `deploy.gpu.yml` |
| 爬虫进 Compose | 仍在独立仓库运行 |
| K8s | 未提供 |
| 自动 HTTPS | 需自备反代 |

---

## 13. 文件清单

| 文件 | 作用 |
|------|------|
| `Dockerfile` | 前端构建 + Python 运行时 |
| `docker-compose.yml` | api + worker |
| `.env.docker.example` | 容器环境模板 |
| `scripts/deploy.sh` | 一键部署 |
| `scripts/deploy-down.sh` | 停止 |
| `DEPLOY.md` | 本文档 |

---

## 14. 验收清单

- [ ] `./scripts/deploy.sh` 成功且 `health/ready` 200  
- [ ] 浏览器可打开并创建管理员  
- [ ] 挂载模型后 worker 心跳有 `model_version`  
- [ ] 媒体在 allowlist 内可显示图片  
- [ ] `deploy.sh --down` 后数据目录仍在  
- [ ] 爬虫使用相同 `INGEST_API_KEY` 可入库  

更多产品与 API 说明见根目录 `README.md`。
