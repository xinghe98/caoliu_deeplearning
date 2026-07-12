# 个人偏好平台后端

## 启动

```powershell
cd C:\Users\mysta\Documents\caoliu_deeplearning
python -m pip install -r requirements.txt
# 可选：空库迁移
# $env:DATABASE_URL = "sqlite:///platform_data/platform.db"
# alembic upgrade head

# 前端（开发）
# cd frontend; npm install; npm run dev

# 前端（生产构建，由 FastAPI 托管 frontend/dist）
# cd frontend; npm run build

python -m uvicorn platform_app.main:app --host 0.0.0.0 --port 8080 --reload
```

首次启动默认 `auto_create_tables=true`，会在 `platform_data/platform.db` 创建 SQLite/WAL 数据库。生产环境建议改用 Alembic：`alembic upgrade head`。数据库、训练包和运行时文件均被 Git 忽略。

## 测试

```powershell
python -m pytest -q
```

## 初始化管理员

首次运行只允许创建一个管理员：

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/v1/auth/setup `
  -H "Content-Type: application/json" `
  -d '{"username":"owner","password":"change-this-to-a-long-password"}'
```

管理员密码最少 12 个字符。之后使用 `/api/v1/auth/login` 获取 HttpOnly 会话 Cookie。

## 允许媒体目录

默认只允许引用训练项目根目录内的图片。爬虫目录位于其他路径时，在启动前设置：

```powershell
$env:ALLOWED_MEDIA_ROOTS = "C:\Users\mysta\Documents\caoliu_deeplearning;C:\Users\mysta\Documents\caoliuSpider\downloads"
```

多个路径用分号分隔。平台不会复制原图，但会拒绝允许根目录以外的文件路径。

## 爬虫入库

给爬虫设置独立密钥：

```powershell
$env:INGEST_API_KEY = "replace-with-a-long-random-value"
```

爬虫在图片下载成功后调用：

```text
POST /api/v1/ingest/content
X-Ingest-Key: <密钥>
```

请求必须包含稳定 `content_key`、标题、磁力链接和 1 至 5 个已下载图片的绝对路径。入库成功后会自动创建 `predict` 任务。

## 模型与 worker

先登记并发布一个真实 checkpoint：

```text
POST /api/v1/models
POST /api/v1/models/{model_id}/activate
```

然后运行 worker：

```powershell
python -m platform_app.worker
```

worker 会保持 active 模型在内存中，并处理数据库中的预测任务。启动前需要确保 BERT 和 ResNet 预训练文件可用。

## 训练包

有明确喜欢和不喜欢标签后，管理员可调用：

```text
POST /api/v1/training/snapshots
GET /api/v1/training/snapshots/{snapshot_id}/download
```

下载得到的 ZIP 包含固定 split 的 `manifest.csv`、所有训练图片、配置和哈希。远程训练不得重新随机划分数据。

在 GPU 服务器上：

```powershell
python train.py --package training_snapshot_xxx.zip --output-dir runs/run1 --run-id run1 --epochs 3
python pack_candidate.py --source-dir runs/run1 --output candidate_run1.zip --run-id run1
```

## 遗留数据迁移

```powershell
python -m platform_app.migrate_legacy --dry-run
python -m platform_app.migrate_legacy --apply
```

## 导回候选模型

GPU 服务器训练完成后，将结果打成 ZIP。包内必须有：

```text
best_model.pth
evaluation_report.json
```

可选加入 `model_manifest.json`。管理员把 ZIP 上传到：

```text
POST /api/v1/training/candidates/import
```

平台会安全检查 ZIP 路径、checkpoint 结构、指标报告和哈希，再登记为 candidate。使用
`GET /api/v1/training/candidates/{model_id}/comparison` 对比 active 模型；确认后调用
`POST /api/v1/models/{model_id}/activate` 发布。发布不会删除旧模型，可随时切换回历史版本。
