# 标题模糊搜索实施计划

## 目标

内容库支持按标题搜索：
- 忽略空白、标点、大小写
- 多关键词空格分隔，AND 匹配
- 例：`abc` 命中 `xxxab crrrs`；`foo bar` 需同时含两者

## 实现步骤

### 1. `platform_app/domain/search.py`（新建）

- `normalize_title_text(s)`: casefold + 去掉非 `\w` 与 `_`
- `parse_search_tokens(q)`: strip、最长 100、按空白 split、normalize、去重、最多 8 token

### 2. `platform_app/database.py`

- connect 钩子注册 SQLite UDF：`normalize_title(1)` → `normalize_title_text`

### 3. `platform_app/main.py` — `list_contents`

- 参数 `q: str | None = None`
- 对每个 token：`normalize_title(title_clean/raw) LIKE %token%`（OR 两列，token 间 AND）
- 与 label/unlabeled/watched/cursor 叠加

### 4. 前端

- `endpoints.ts`: `list` 增加 `q`
- `LibraryPage.tsx`: 搜索框 + 300ms debounce；`queryKey` 含 q；空结果文案

### 5. 测试 `tests/platform/test_api.py`

- `abc` → `xxxab crrrs`
- 大小写、标点、多词 AND、与 label 组合

### 6. 部署

```bash
docker compose up -d --build --no-deps api
```

## 状态

待退出 Plan 模式后执行代码修改。
