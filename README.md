# 个人偏好内容筛选与持续学习平台

本仓库（`caoliu_deeplearning`）包含两套能力：

1. **多模态偏好模型**：用缩略图 + 标题预测内容是否符合个人偏好（ResNet-50 + 中文 BERT）。
2. **局域网个人平台**：爬虫入库 → 模型打分 → Web 喜欢/不喜欢标注 → 训练包导出 → 临时 GPU 训练 → 候选模型人工发布。

`caoliuSpider/` 已并入本仓库，负责采集、下载图片和调用平台入库 API。

---

## 目录

- [1. 系统逻辑](#1-系统逻辑)
- [2. 技术栈](#2-技术栈)
- [3. 仓库目录说明](#3-仓库目录说明)
- [4. 环境准备](#4-环境准备)
- [5. 快速启动（平台）](#5-快速启动平台)
- [6. 前端开发与打包](#6-前端开发与打包)
- [7. 后端部署](#7-后端部署)
- [7.1 Linux Docker 一键部署](#71-linux-docker-一键部署)
- [8. Worker 推理服务](#8-worker-推理服务)
- [9. 爬虫集成](#9-爬虫集成)
- [10. 历史数据迁移](#10-历史数据迁移)
- [11. 训练与候选模型回流](#11-训练与候选模型回流)
- [12. 环境变量一览](#12-环境变量一览)
- [13. 数据与标签约定](#13-数据与标签约定)
- [14. API 摘要](#14-api-摘要)
- [15. 测试](#15-测试)
- [16. 数据库迁移](#16-数据库迁移)
- [17. 安全与运维注意](#17-安全与运维注意)
- [18. 故障排查](#18-故障排查)
- [19. 实施状态](#19-实施状态)

---

## 1. 系统逻辑

```text
┌─────────────┐     入库 API      ┌──────────────────┐
│ caoliuSpider│ ───────────────► │  platform_app     │
│  采集/下图  │                  │  FastAPI + SQLite │
└─────────────┘                  └────────┬─────────┘
                                          │ predict job
                                          ▼
                                 ┌──────────────────┐
                                 │ platform worker  │
                                 │ active 模型推理  │
                                 └────────┬─────────┘
                                          │ 分数写入
                                          ▼
                                 ┌──────────────────┐
                                 │  React 前端      │
                                 │  喜欢/不喜欢/跳过 │
                                 └────────┬─────────┘
                                          │ 累计 ~200 新标签
                                          ▼
                                 ┌──────────────────┐
                                 │ 训练 ZIP 快照    │
                                 │ 固定 split       │
                                 └────────┬─────────┘
                                          │ 手动上传 GPU
                                          ▼
                                 ┌──────────────────┐
                                 │ train.py --package│
                                 │ pack_candidate.py │
                                 └────────┬─────────┘
                                          │ 候选 ZIP 导回
                                          ▼
                                 ┌──────────────────┐
                                 │ 人工对比并发布   │
                                 │ active 模型切换  │
                                 └──────────────────┘
```

### 硬原则

| 原则 | 说明 |
|------|------|
| 真值只来自人工 | 只有明确「喜欢 / 不喜欢」进入训练；跳过、浏览、复制链接只记行为 |
| 模型不自动上线 | 候选模型必须人工发布；可回滚 |
| 数据可追溯 | 标签事件不可原地覆盖；训练包、模型版本带哈希 |
| 原图不搬家 | 媒体原地引用（数据集/爬虫目录）；平台只存路径 + 自有产物 |
| 推荐非全量 | 待筛选 = 约 80% 高分推荐 + 20% 探索；内容库 = 全量 |

---

## 2. 技术栈

| 层 | 选型 |
|----|------|
| 前端 | React 19、TypeScript、Vite 6、React Router、TanStack Query、Tailwind CSS 4、Lucide |
| 后端 | FastAPI、Pydantic Settings、SQLAlchemy 2、Alembic |
| 数据库 | SQLite（WAL、外键、busy_timeout） |
| 任务 | 独立 Python worker，DB 任务表领取（无 Redis/Celery） |
| 模型 | PyTorch、ResNet-50、`bert-base-chinese`、温度校准 + 业务阈值 |
| 爬虫 | Scrapy（同仓库 Compose 服务） |

---

## 3. 仓库目录说明

```text
caoliu_deeplearning/
├── README.md                          # 本文档
├── PRODUCT.md                         # 产品语境与设计原则
├── PERSONAL_PREFERENCE_PLATFORM_PLAN.md  # 详细开发计划与交接状态
├── requirements.txt                   # Python 依赖
├── config.py                          # 训练/推理超参数（Config 类）
├── dataset.py                         # 数据集加载、分组切分、训练包加载
├── train.py                           # 训练入口（支持 --package）
├── predict.py                         # Predictor 推理（含 predict_from_paths）
├── pack_candidate.py                  # 将训练产物打成候选 ZIP
├── api.py                             # 旧版独立预测 API（兼容保留）
├── models/                            # 神经网络结构
│   ├── classifier.py                  # 多模态融合分类器
│   ├── image_encoder.py               # ResNet 图像编码
│   └── text_encoder.py                # BERT 文本编码
├── utils/                             # 训练工具
│   ├── trainer.py                     # 单轮训练/验证
│   ├── evaluation.py                  # 指标、阈值、温度
│   ├── losses.py
│   └── helpers.py
├── platform_app/                      # ★ 个人偏好平台后端
│   ├── main.py                        # FastAPI 路由 + 前端静态托管
│   ├── config.py                      # 平台环境变量
│   ├── database.py                    # 可重绑 SQLAlchemy engine
│   ├── models.py                      # ORM 表定义
│   ├── schemas.py                     # Pydantic 请求/响应
│   ├── auth.py                        # 管理员会话、CSRF、登录限流
│   ├── services.py                    # 入库、feed、idempotency 等
│   ├── training.py                    # 训练快照 ZIP 生成
│   ├── candidates.py                  # 候选模型安全导入
│   ├── worker.py                      # 预测 / 导出任务 worker
│   ├── model_manager.py               # active 模型缓存与热加载
│   ├── migrate_legacy.py              # 遗留数据集导入
│   ├── domain/                        # 纯逻辑（可单测）
│   │   ├── keys.py                    # content_key / magnet / group
│   │   ├── splits.py                  # 稳定 train/val/shadow split
│   │   ├── media.py                   # 图片校验
│   │   └── labels.py                  # 打标 / 撤销
│   └── README.md                      # 后端简要说明
├── frontend/                          # ★ Web 前端
│   ├── src/
│   │   ├── api/                       # HTTP 客户端与类型
│   │   ├── components/                # 布局、Toast
│   │   ├── hooks/                     # 鉴权
│   │   ├── pages/                     # 各业务页
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── dist/                          # npm run build 产物（Git 忽略）
│   ├── package.json
│   └── README.md
├── alembic/                           # 数据库迁移脚本
├── alembic.ini
├── tests/                             # pytest
│   ├── conftest.py                    # 临时库 / TestClient fixture
│   ├── domain/
│   ├── platform/
│   ├── migrate/
│   ├── worker/
│   └── training/
├── 数据集1/ … 数据集3/ …             # 历史标注数据（Git 忽略）
├── downloads/                         # 爬虫下载目录（常见，Git 忽略）
├── platform_data/                     # 运行时：DB、训练包、候选（Git 忽略）
├── best_model*.pth                    # 模型权重（Git 忽略 *.pth）
├── Dockerfile / docker-compose.yml    # Linux 一键部署（见 DEPLOY.md）
├── scripts/deploy.sh                  # Docker Compose 部署脚本
├── DEPLOY.md                          # 部署文档
└── pytest.ini
```

### 关键运行时目录（不进 Git）

| 路径 | 用途 |
|------|------|
| `platform_data/platform.db` | SQLite 主库 |
| `platform_data/training_snapshots/` | 导出的训练 ZIP |
| `platform_data/candidates/` | 导入的候选包解压目录 |
| `frontend/dist/` | 前端生产静态资源 |
| `frontend/node_modules/` | 前端依赖 |
| `.env` | 本地密钥与路径配置 |

### 相关外部目录

| 路径 | 用途 |
|------|------|
| `caoliuSpider/` | Scrapy 爬虫；下载完成后调用入库 API |

---

## 4. 环境准备

### 4.1 系统要求

- **OS**：Windows 10/11（当前主部署目标）
- **Python**：3.10+（推荐 3.11/3.12）
- **Node.js**：18+（前端开发/构建；仅跑后端可跳过）
- **可选 GPU**：本地训练或 worker 推理加速；临时 GPU 服务器用于完整训练

### 4.2 安装 Python 依赖

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning
python -m pip install -r requirements.txt
```

主要依赖：PyTorch、transformers、FastAPI、SQLAlchemy、Alembic、Pillow、pwdlib[argon2] 等。

### 4.3 预训练权重

首次训练或真实推理需要：

- Hugging Face：`bert-base-chinese`
- torchvision：ResNet-50 ImageNet 权重

无法访问 Hugging Face 时，请在可联网机器下载后配置本地缓存（如 `HF_HOME` / `TRANSFORMERS_CACHE`）。

### 4.4 安装前端依赖

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning\frontend
npm install
```

### 4.5 建议的 `.env` 示例

在项目根目录创建 `.env`（已被 `.gitignore` 忽略）：

```env
# 数据与库
PLATFORM_DATA_DIR=C:\Users\mysta\Documents\caoliu_deeplearning\platform_data
# DATABASE_URL=sqlite:///C:/Users/mysta/Documents/caoliu_deeplearning/platform_data/platform.db

# 允许读取的图片根目录（分号分隔，Windows）
ALLOWED_MEDIA_ROOTS=C:\Users\mysta\Documents\caoliu_deeplearning;C:\Users\mysta\Documents\caoliu_deeplearning\downloads

# 爬虫入库密钥（务必改成足够长的随机串）
INGEST_API_KEY=replace-with-a-long-random-value

# 推荐与训练
FEED_RECOMMENDATION_RATIO=0.8
TRAINING_LABEL_THRESHOLD=200
SKIP_COOLDOWN_DAYS=7

# 安全（局域网 HTTP 时 secure cookie 保持 false）
COOKIE_SECURE=false
CSRF_ENABLED=true
AUTO_CREATE_TABLES=true
```

---

## 5. 快速启动（平台）

### 5.1 开发模式（前后端分离，推荐日常开发）

**终端 1 — 后端**

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning
# 按需设置环境变量，或依赖 .env
$env:ALLOWED_MEDIA_ROOTS = "C:\Users\mysta\Documents\caoliu_deeplearning;C:\Users\mysta\Documents\caoliu_deeplearning\downloads"
$env:INGEST_API_KEY = "replace-with-a-long-random-value"

python -m uvicorn platform_app.main:app --host 0.0.0.0 --port 8080 --reload
```

**终端 2 — Worker（可选，有 active 模型后再开）**

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning
python -m platform_app.worker
```

**终端 3 — 前端**

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning\frontend
npm run dev
```

浏览器打开：`http://127.0.0.1:5173`  
Vite 将 `/api`、`/health` 代理到 `http://127.0.0.1:8080`。

### 5.2 首次管理员

打开前端登录页，选择「创建管理员」，或：

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/v1/auth/setup `
  -H "Content-Type: application/json" `
  -d "{\"username\":\"owner\",\"password\":\"change-this-to-a-long-password\"}"
```

- 密码至少 **12** 位  
- 仅允许创建 **一个** 管理员  
- 会话：HttpOnly Cookie + CSRF Cookie（写接口需 `X-CSRF-Token`）

### 5.3 生产一体模式（FastAPI 托管前端）

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning\frontend
npm run build

cd C:\Users\mysta\Documents\caoliu_deeplearning
python -m uvicorn platform_app.main:app --host 0.0.0.0 --port 8080
```

访问：`http://本机局域网IP:8080/`  
当存在 `frontend/dist` 时，后端自动托管静态资源与 SPA 路由回退。

---

## 6. 前端开发与打包

### 6.1 脚本

| 命令 | 说明 |
|------|------|
| `npm run dev` | 开发服务器（默认 5173） |
| `npm run build` | `tsc --noEmit` + Vite 生产构建 → `dist/` |
| `npm run preview` | 本地预览构建结果 |
| `npm run lint` | 仅 TypeScript 检查 |

### 6.2 代理与 API

- 开发：`frontend/vite.config.ts` 代理 `/api`、`/health` → `127.0.0.1:8080`
- 生产：同源，无需代理；`VITE_API_BASE` 默认可为空（见 `.env.example`）

### 6.3 页面路由

| 路径 | 说明 |
|------|------|
| `/login` | 登录 / 首次 setup |
| `/review` | 待筛选队列（主工作台） |
| `/library` | 全量内容库 |
| `/labels` | 标签历史与撤销 |
| `/training` | 训练进度、快照下载、候选上传 |
| `/models` | 模型列表、发布/拒绝/回滚 |
| `/crawler` | 任务列表与 worker 心跳 |
| `/settings` | 健康状态与运维说明 |

### 6.4 Review 快捷键

| 键 | 动作 |
|----|------|
| `1` | 喜欢 |
| `2` | 不喜欢 |
| `3` | 跳过（7 天冷却，不进训练） |
| `M` | 复制磁力链接 |
| `←` / `→` | 切换图片 |
| `Z` | 撤销最近标签 |

手机端使用底部大按钮；滑动只切图，不直接提交标签。

### 6.5 打包检查清单

```powershell
cd frontend
npm ci          # 或 npm install
npm run lint
npm run build
# 确认 dist/index.html 与 dist/assets/* 存在
```

将整仓部署到服务器时，**务必带上** `frontend/dist`，或在服务器上重新 `npm run build`。

---

## 7. 后端部署

### 7.1 Linux Docker 一键部署

完整说明见 **[DEPLOY.md](./DEPLOY.md)**。最短路径：

```bash
chmod +x scripts/deploy.sh
export MEDIA_HOST_DIR=/path/to/images
export MODEL_FILE=/path/to/best_model.pth
./scripts/deploy.sh
```

将启动 `api`（含前端）+ `worker` + `crawler`，爬虫与平台通过共享媒体目录衔接。

### 7.2 最小生产启动（Windows 本机）

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning

# 1) 依赖
python -m pip install -r requirements.txt

# 2) 环境变量（建议用 .env 或系统环境变量）
$env:ALLOWED_MEDIA_ROOTS = "C:\Users\mysta\Documents\caoliu_deeplearning;C:\Users\mysta\Documents\caoliu_deeplearning\downloads"
$env:INGEST_API_KEY = "你的长随机密钥"
$env:PLATFORM_DATA_DIR = "C:\Users\mysta\Documents\caoliu_deeplearning\platform_data"

# 3) 前端构建（若尚未构建）
cd frontend; npm ci; npm run build; cd ..

# 4) 数据库
# 开发默认可 auto_create_tables=true
# 生产建议：
# $env:AUTO_CREATE_TABLES = "false"
# alembic upgrade head

# 5) 启动 API（监听全网卡以便局域网访问）
python -m uvicorn platform_app.main:app --host 0.0.0.0 --port 8080

# 6) 另开终端启动 worker
python -m platform_app.worker --batch-size 16
```

### 7.3 局域网访问

1. 主机防火墙放行 **Private** 网络的 **TCP 8080**。  
2. 手机/其他电脑访问 `http://主机局域网IP:8080`。  
3. 勿将服务直接暴露到公网（当前无 HTTPS、无完整企业级加固）。

### 7.4 建议进程模型

| 进程 | 命令 | 说明 |
|------|------|------|
| API | `uvicorn platform_app.main:app --host 0.0.0.0 --port 8080` | Web + API + 静态前端 |
| Worker | `python -m platform_app.worker` | 预测与训练导出 job |
| 爬虫 | 见 §9 | 定时增量采集 |

可使用 Windows「任务计划程序」或 NSSM 注册为开机服务（P6 计划中的脚本尚未强制提供）。Docker 部署见 [DEPLOY.md](./DEPLOY.md)。

### 7.5 健康检查

| URL | 含义 |
|-----|------|
| `GET /health/live` | 进程存活 |
| `GET /health/ready` | 数据库可读 |
| `GET /health/worker` | worker 心跳列表 |

---

## 8. Worker 推理服务

```powershell
python -m platform_app.worker
# 只跑一轮（调试）
python -m platform_app.worker --once
# 批量大小
python -m platform_app.worker --batch-size 32 --poll-seconds 5
```

行为摘要：

1. 优先处理 `export_training_snapshot` 任务，再处理 `predict`。  
2. 从 DB 加载 **active** 模型，常驻内存；版本变化时热加载。  
3. 按内容的 **媒体路径列表** 调用 `Predictor.predict_from_paths`（不再假设「目录里只有该内容图片」）。  
4. 写入 `predictions` 表；失败最多重试 3 次。  
5. 定期写 `worker_heartbeats`。

**发布模型后**：worker 在下一批任务时加载新 active，无需重启 Web；若加载失败会保留旧模型逻辑（以代码行为为准，发布前请确认 checkpoint 可读）。

---

## 9. 爬虫集成

目录：`caoliuSpider/`

### 9.1 配置

```powershell
$env:PLATFORM_API_URL = "http://127.0.0.1:8080"
$env:INGEST_API_KEY = "与平台相同的密钥"
# 可选：平台不可用时是否让本次爬虫失败（默认 false，只审计失败）
$env:PLATFORM_INGEST_REQUIRED = "true"
```

爬虫侧从环境变量读取这些配置；不要把密钥写进 `settings.py`。平台侧必须把爬虫下载根目录加入 `ALLOWED_MEDIA_ROOTS`。

启动顺序：先启动平台 API 和 worker，再从爬虫项目目录执行：

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning\caoliuSpider\caoliu
scrapy crawl caoliu
```

Docker 部署无需手动执行上述命令；`crawler` 服务默认每 6 小时运行一次。通过 `CRAWLER_INTERVAL_SECONDS`、`CRAWLER_START_PAGE`、`CRAWLER_MAX_PAGES` 调整计划，详见 [DEPLOY.md](./DEPLOY.md)。

每个已完成图片下载的内容都会异步入库；入库结果写入 `downloads/platform_ingest.jsonl`，状态为 `succeeded`、`failed` 或 `skipped`。网络错误、429 和 5xx 默认最多重试 3 次；4xx 会立即记录失败，通常表示密钥、图片路径或请求内容需要修复。

### 9.2 流程

1. 分配稳定 `content_key`（优先 BT infohash，否则 URL hash）  
2. 下载最多 5 张图，校验可解码、最小约 224×224、去重  
3. 无有效图则丢弃，不入库  
4. `PlatformIngestPipeline` → `POST /api/v1/ingest/content`  
5. 后端创建 `predict` job  

### 9.3 入库请求字段

```json
{
  "content_key": "btih:...",
  "source_url": "https://...",
  "title_raw": "...",
  "title_clean": "...",
  "magnet_uri": "magnet:?xt=urn:btih:...",
  "info_hash": "40位hex",
  "media": [
    { "source_path": "C:\\...\\image_01.jpg", "ordinal": 1 }
  ]
}
```

Header：`X-Ingest-Key: <INGEST_API_KEY>`

响应含：`content_id`、`created`、`duplicate`、`prediction_job_id`。

---

## 10. 历史数据迁移

将 `数据集1`… 与 `downloads` 的 `index.csv` + 图片导入平台（**原图不移动**）：

```powershell
# 只扫描报告，不写库
python -m platform_app.migrate_legacy --dry-run

# 正式导入（幂等，可重复执行）
python -m platform_app.migrate_legacy --apply
```

常用参数：

```powershell
python -m platform_app.migrate_legacy --apply `
  --root C:\Users\mysta\Documents\caoliu_deeplearning `
  --folders 数据集1 数据集2 数据集3 downloads
```

说明：

- 有 `label=0/1` → `historical_import` 标签事件  
- `数据集3` → `dataset_role=external_test`（默认不进训练包梯度数据）  
- 报告默认：`platform_data/migration_report.csv`

---

## 11. 训练与候选模型回流

### 11.1 经典本地训练（多数据集目录）

数据布局：

```text
数据集N/
  index.csv
  video_xx/
    image_01.jpg
    ...
```

```powershell
python train.py
python train.py --batch-size 8 --epochs 20
python train.py --resume
```

产物（默认项目根或输出目录）：

- `best_model.pth`
- `evaluation_report.json`
- `training_history.json` / `.png`
- `validation_*.csv`、`external_test_*.csv`
- `split_manifest.csv`

`数据集3` 默认锁定为外部测试集，不参与训练梯度。

### 11.2 平台训练包模式（推荐与平台闭环）

**在平台导出 ZIP**（Web「训练」页，或 API `POST /api/v1/training/snapshots`）。

包内结构示例：

```text
manifest.csv
split_manifest.csv
config.json
README_TRAINING.md
SHA256SUMS.json
images/<content_id>/image_01.jpg
...
```

**在 GPU 机器上：**

```powershell
# 解压非必须：train.py 可直接读 zip
python train.py `
  --package training_snapshot_xxx.zip `
  --output-dir runs/run_20260711 `
  --run-id run_20260711 `
  --epochs 15 `
  --batch-size 16

# 打包候选
python pack_candidate.py `
  --source-dir runs/run_20260711 `
  --output candidate_run_20260711.zip `
  --run-id run_20260711
```

约束：

- **禁止**在远端对包内数据重新随机 `train_test_split`  
- `train` 才进梯度；`validation` 选阈值/温度；`production_shadow_test` / `external_test` 不进训练  

### 11.3 导回平台并发布

1. Web → **训练** → 上传 `candidate_*.zip`  
   或 `POST /api/v1/training/candidates/import`  
2. **模型** 页查看对比（PR-AUC 等）  
3. **发布** / 强制发布 / 拒绝 / 回滚  

候选包至少包含：

- `best_model.pth`（含 `model_state_dict`）  
- `evaluation_report.json`  

推荐同时包含：`model_manifest.json`、`SHA256SUMS.json`、训练历史与错误集合 CSV。

### 11.4 自动训练触发

当自上一快照后 **明确 Web 标签** 达到 `TRAINING_LABEL_THRESHOLD`（默认 200）时，后端会创建 `export_training_snapshot` job；worker 执行后生成 ZIP。  
**不会**自动租 GPU 或自动发布模型。

---

## 12. 环境变量一览

| 变量 | 默认 | 说明 |
|------|------|------|
| `PLATFORM_DATA_DIR` | `<repo>/platform_data` | 运行时数据根 |
| `DATABASE_URL` | `sqlite:///<PLATFORM_DATA_DIR>/platform.db` | 数据库 URL |
| `ALLOWED_MEDIA_ROOTS` | 项目根 | 允许的图片绝对路径根（`;` 分隔） |
| `INGEST_API_KEY` | 空 | 必填；为空时入库接口返回 503 |
| `FEED_RECOMMENDATION_RATIO` | `0.8` | mixed 队列推荐占比 |
| `TRAINING_LABEL_THRESHOLD` | `200` | 自动快照阈值 |
| `SKIP_COOLDOWN_DAYS` | `7` | 跳过冷却天数 |
| `COOKIE_SECURE` | `false` | HTTPS 时再设 `true` |
| `CSRF_ENABLED` | `true` | 写接口 CSRF |
| `AUTO_CREATE_TABLES` | `true` | 启动时 create_all；生产建议 false + Alembic |
| `CANDIDATE_MAX_UPLOAD_MB` | `512` | 候选上传上限 |
| `CANDIDATE_MAX_FILES` | `200` | ZIP 内文件数上限 |
| `CANDIDATE_MAX_UNCOMPRESSED_MB` | `1024` | 解压后体积上限 |
| `LOGIN_RATE_LIMIT_ATTEMPTS` | `10` | 登录失败次数 |
| `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | `300` | 限流窗口 |

前端：

| 变量 | 说明 |
|------|------|
| `VITE_API_BASE` | API 前缀；同源部署留空 |

爬虫：

| 变量 | 说明 |
|------|------|
| `PLATFORM_API_URL` | 如 `http://192.168.x.x:8080` |
| `INGEST_API_KEY` | 与平台一致 |

---

## 13. 数据与标签约定

### 13.1 内容唯一键

1. 有 40 位 BT infohash → `content_key = btih:<hex>`  
2. 否则 → `url:<sha256(规范化URL)>`  

相同 `content_key` 再次入库视为 **duplicate**，更新可变元数据，不重复插媒体（当前实现以代码为准）。

### 13.2 标签

| 行为 | 是否训练真值 | 说明 |
|------|--------------|------|
| 喜欢 (1) | 是 | `label_events`，source 多为 `explicit_web` |
| 不喜欢 (0) | 是 | 同上 |
| 跳过 | 否 | `view_events`，默认 7 天不进 feed |
| 浏览/复制磁力 | 否 | 仅统计 |

修改标签写入新事件并 `supersedes_event_id`；撤销通过补偿事件恢复 `current_label`，不删除历史。

### 13.3 Split

- `external_test`：`dataset_role=external_test`（如数据集3）固定  
- 其他组：对 `content_group_id` 哈希 → 约 80% train / 10% validation / 10% production_shadow_test  

---

## 14. API 摘要

基址：`http://主机:8080`

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/api/v1/auth/setup` | 无 | 首次管理员 |
| POST | `/api/v1/auth/login` | 无 | 登录 |
| POST | `/api/v1/auth/logout` | Cookie+CSRF | 登出 |
| GET | `/api/v1/auth/session` | Cookie | 当前会话 |
| POST | `/api/v1/ingest/content` | Ingest Key | 爬虫入库 |
| GET | `/api/v1/feed` | Cookie | 待筛选队列 |
| GET | `/api/v1/contents` | Cookie | 内容列表 |
| GET | `/api/v1/contents/{id}/media/{media_id}` | Cookie | 图片 |
| POST | `/api/v1/contents/{id}/label` | Cookie+CSRF | 标注 |
| POST | `/api/v1/contents/{id}/events` | Cookie+CSRF | 行为事件 |
| GET | `/api/v1/labels/history` | Cookie | 标签历史 |
| POST | `/api/v1/labels/{event_id}/undo` | Cookie+CSRF | 撤销 |
| GET/POST | `/api/v1/training/snapshots` | Cookie(+CSRF POST) | 快照 |
| GET | `/api/v1/training/snapshots/{id}/download` | Cookie | 下载 ZIP |
| POST | `/api/v1/training/candidates/import` | Cookie+CSRF | 导入候选 |
| GET | `/api/v1/models` | Cookie | 模型列表 |
| POST | `/api/v1/models/{id}/activate` | Cookie+CSRF | 发布 |
| POST | `/api/v1/models/{id}/reject` | Cookie+CSRF | 拒绝 |
| POST | `/api/v1/models/{id}/rollback` | Cookie+CSRF | 回滚 |
| GET | `/api/v1/jobs` | Cookie | 后台任务 |

写接口支持 `Idempotency-Key`（防重复提交）。

OpenAPI 文档（开发时）：`http://127.0.0.1:8080/docs`（若 SPA catch-all 与 docs 冲突，可暂时去掉 dist 或调整路由优先级）。

---

## 15. 测试

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning

# Windows 若默认 pytest 临时目录权限异常，请指定 basetemp
python -m pytest --basetemp=.pytest_tmp/run --tb=short
```

覆盖：domain 键/split、鉴权与入库、标签/撤销/幂等、训练 ZIP、候选路径穿越、遗留迁移幂等、worker mock 预测、训练包加载与 pack_candidate、既有 evaluation 单测。

**不在 CI 默认范围**：真实 BERT/ResNet 端到端、全量数据集 apply、真站爬虫。

---

## 16. 数据库迁移

```powershell
# 使用当前配置的 DATABASE_URL / PLATFORM_DATA_DIR
alembic upgrade head
```

- 初始迁移：`alembic/versions/0001_initial.py`；后续迁移：`0002_snapshot_label_events.py`
- 开发默认可 `AUTO_CREATE_TABLES=true` 自动建表  
- **生产建议** `AUTO_CREATE_TABLES=false`，只用 Alembic，避免 schema 漂移  

---

## 17. 安全与运维注意

1. **仅局域网**：不要裸奔公网；需要外网时加反向代理 + HTTPS + `COOKIE_SECURE=true`。  
2. **密钥**：`INGEST_API_KEY`、管理员密码勿提交 Git。  
3. **媒体路径**：所有图片路径必须在 `ALLOWED_MEDIA_ROOTS` 内；接口禁止用户任意读盘。  
4. **磁力链接**：仅接受 `magnet:` 协议。  
5. **备份**：定期复制 `platform_data/platform.db` 与训练/候选目录；**不等于**备份原图目录。  
6. **日志**：勿打印完整 Cookie / 密码 / 完整 API Key。  
7. **候选 ZIP**：有大小、文件数、解压体积与路径穿越检查；仍应只导入可信来源产物。

---

## 18. 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 入库 403 图片路径 | 不在 allowlist | 扩展 `ALLOWED_MEDIA_ROOTS` |
| 入库 401 | 密钥不匹配 | 对齐 `INGEST_API_KEY` 与 Header |
| feed 一直空 | 无未标注内容 / 全在 skip 冷却 | 查内容库；等冷却或迁入数据 |
| 分数全是空 | 无 active 模型或 worker 未跑 | 登记并 activate 模型；启动 worker |
| 前端写接口 403 | CSRF | 确认登录后带 CSRF cookie；同源访问 |
| 训练包生成 422 | 缺少正或负样本 | 至少各 1 条喜欢/不喜欢 |
| 候选导入失败 | 缺文件 / 坏 zip / 超限 | 用 `pack_candidate.py` 规范打包 |
| pytest tmp 权限 | Windows 临时目录 | `--basetemp=.pytest_tmp/run` |
| 图片 410 | 原图被删/移动 | 恢复文件或重新入库 |

---

## 19. 实施状态

已完成（代码级，详见 `PERSONAL_PREFERENCE_PLATFORM_PLAN.md` §1.1）：

- P0 后端稳定化 + pytest  
- P1 遗留数据迁移脚本  
- P2 爬虫稳定键与入库 pipeline  
- P3 worker / 发布门禁 / skip 冷却  
- P4 训练包与 `train.py --package` / `pack_candidate`  
- P5 React 前端 + FastAPI 静态托管  

未完成或需本机验收：

- P6 部分运维：Docker Compose 一键部署已提供（DEPLOY.md）；Windows 服务化 / 日志轮转策略仍可选增强  

- 真权重端到端推理与完整 GPU 训练闭环的现场验收  
- 部分高级内容库筛选 / 缩略图缓存等增强项  

---

## 一页纸操作顺序（新机器）

```powershell
# 1. 代码与依赖
cd C:\Users\mysta\Documents\caoliu_deeplearning
python -m pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..

# 2. 配置 .env（ALLOWED_MEDIA_ROOTS、INGEST_API_KEY 等）

# 3. 启动
python -m uvicorn platform_app.main:app --host 0.0.0.0 --port 8080
# 另一窗口
python -m platform_app.worker

# 4. 浏览器打开 http://127.0.0.1:8080 → 创建管理员

# 5. 可选：导入历史数据
python -m platform_app.migrate_legacy --apply

# 6. 可选：登记已有 best_model.pth 并 activate（Web 或 API）

# 7. 配置爬虫 PLATFORM_API_URL + INGEST_API_KEY 后定时抓取

# 8. 日常：Review 标注 → 满阈值出包 → GPU 训练 → 导回候选 → 人工发布
```

更细的产品原则见 `PRODUCT.md`；阶段计划与验收清单见 `PERSONAL_PREFERENCE_PLATFORM_PLAN.md`。
