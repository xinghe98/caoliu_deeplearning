# 偏好筛选平台前端

## 开发

```powershell
cd frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。`/api` 与 `/health` 代理到 `http://127.0.0.1:8080`。

先启动后端：

```powershell
python -m uvicorn platform_app.main:app --host 0.0.0.0 --port 8080
```

## 生产构建

```powershell
cd frontend
npm run build
```

产物输出到 `frontend/dist`，由 FastAPI 同源托管。访问 `http://主机IP:8080/`。

## 路由

| 路径 | 说明 |
|------|------|
| `/login` | 登录 / 首次 setup |
| `/review` | 待筛选（80/20 队列） |
| `/library` | 全量内容库 |
| `/labels` | 标签历史与撤销 |
| `/training` | 训练进度、快照、候选导入 |
| `/models` | 模型发布/拒绝/回滚 |
| `/crawler` | 任务与 worker 心跳 |
| `/settings` | 健康检查与运维说明 |

## 键盘（Review）

`1` 喜欢 · `2` 不喜欢 · `3` 跳过 · `M` 复制磁力 · `←/→` 切图 · `Z` 撤销
