# 九寨沟景区智能服务与运营 Agent

面向九寨沟风景名胜区（`park_id = jiuzhaigou_scenic_area`）的 AI 智能客服 + 运营后台系统：
FastAPI + PostgreSQL(pgvector) + Redis + arq + Vue 3。

知识库内容来自公开可查证的九寨沟真实资料（景点、设施、票价、开放时间、游览规则、官方问答和带生效时间的景区公告），
数据来源、口径冲突与已知缺口见 [README_DATA.md](README_DATA.md)。

## 本地启动

```powershell
python -m pip install -r requirements.txt
$env:DATABASE_URL = "postgresql+psycopg://postgres:123456@localhost:5432/scenic_agent"
$env:PARK_ID = "jiuzhaigou_scenic_area"
$env:ADMIN_EMAIL = "admin@jiuzhaigou.local"
$env:ADMIN_PASSWORD = "admin123456"
alembic -c alembic.ini upgrade head
uvicorn app.main:app --app-dir backend --reload --port 8000
```

另开终端：

```powershell
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`。默认使用 `LLM_MODE=fallback`，不需要任何模型 Key。

生产环境必须设置 `ENVIRONMENT=production`、随机 `JWT_SECRET`、非默认管理员密码、真实
`DATABASE_URL` 和显式 `CORS_ORIGINS`；配置校验会拒绝开发默认值。管理员登录失败按 IP +
账号维度限流，运行指标可从 `/api/v1/metrics` 获取。

数据库需要先建表并导入九寨沟数据集：

```powershell
psql -f sql/init.sql
python scripts/generate_jiuzhaigou_data.py
python scripts/import_to_postgres.py --dsn "postgresql+psycopg://postgres:123456@localhost:5432/scenic_agent"
```

## Embedding Worker

## Milvus Hybrid RAG

The application keeps PostgreSQL as the transactional source of truth and uses
Milvus as the primary document index. Each chunk is indexed with both a dense vector
and Milvus's native BM25 sparse function. Docker Compose starts PostgreSQL, Redis,
Milvus, etcd and MinIO together:

```powershell
Copy-Item .env.example .env
docker compose up -d
docker compose ps
```

Import the dataset before building an index. With an embedding API configured, build
the PostgreSQL compatibility vectors and then index the same chunks into Milvus. The
importer now performs real paragraph/sentence chunking (default 800 characters with
120-character overlap) before writing `document_chunks`; it no longer treats every
source document as `chunk_index=0`:

```powershell
python scripts/generate_jiuzhaigou_data.py
python scripts/import_to_postgres.py --dsn "postgresql+psycopg://postgres:123456@localhost:5432/scenic_agent" --chunk-size 800 --chunk-overlap 120
python scripts/index_milvus.py --uri http://localhost:19530 --collection scenic_knowledge
```

To ingest a real Markdown/plain-text guide, use the same chunker and create a
durable embedding task for the Worker. This repository includes a detailed sample
at `data/jiuzhaigou/source_documents/jiuzhaigou_detailed_guide.md`:

```powershell
python scripts/ingest_markdown.py `
  --dsn "postgresql+psycopg://postgres:123456@localhost:5432/scenic_agent" `
  --input data/jiuzhaigou/source_documents/jiuzhaigou_detailed_guide.md `
  --chunk-size 800 --chunk-overlap 120 `
  --redis-url redis://127.0.0.1:6380/0
```

The command writes one `documents` row, multiple `document_chunks` rows, and one
durable `async_tasks` row. The Worker embeds every chunk and upserts deterministic
IDs such as `doc_jiuzhaigou_detailed_guide:0` and `:1` into Milvus. Re-running an
ingestion replaces the document's chunk layout and invalidates its previous vectors,
so the embedding task must complete before relying on semantic retrieval.

The migration is intentionally destructive for the selected Milvus collection:
`scripts/index_milvus.py` drops the existing Dense-only collection and creates a new
Dense + native BM25 schema. PostgreSQL is not deleted or changed. An embedding API key
is required, and the command must be rerun whenever the embedding dimension or schema
changes. Use `--keep-existing` only when the collection already has the hybrid schema.
The PostgreSQL compatibility embedding column uses `halfvec(2560)` with
`halfvec_cosine_ops`; pgvector's HNSW `vector` type cannot index more than 2000
dimensions. Existing databases can apply `sql/migrate_embedding_dimension_2560.sql`.

Runtime selection is controlled by `RAG_BACKEND`:

| Value | Behaviour |
|---|---|
| `postgres` | Existing PostgreSQL retrieval only; rollback mode. |
| `shadow` | Milvus serves the candidate path while PostgreSQL is also queried for comparison. |
| `milvus` | Milvus Dense + native BM25 plus PostgreSQL entity/structured evidence; failure falls back to PostgreSQL. |

`rag_search` is the canonical Agent RAG tool. It returns retrieval channels,
authority labels, citations and conflict markers. `search_knowledge` remains for
backward compatibility. High-risk ticketing, opening, restriction, safety and route
questions have a deterministic retrieval policy and cannot bypass evidence gathering.
The result also exposes `evidence_score`, `grounding_status` and `abstention_required`;
these are explainable rule scores, not calibrated probabilities.

`.env.example` contains the complete configuration surface without real secrets. Do not
commit a populated `.env`; use a secret manager or deployment environment variables.

### 离线评测

评测兼容 `expected_document_id` 和多文档标注字段，并输出 Recall、Precision、MRR、nDCG
及按问题类型分组结果：

```powershell
python scripts/evaluate_rag.py --input data/jiuzhaigou/evaluation_questions.jsonl --output reports/eval_baseline.json
```

当前还提供复杂场景评测：

```powershell
python scripts/evaluate_rag.py --input data/jiuzhaigou/evaluation_questions_v2.jsonl --split complex --output reports/eval_complex.json
```

详细的架构、数据来源、上线检查和评测口径见 [`docs/项目解读-v2.0.md`](docs/项目解读-v2.0.md)。

没有 PostgreSQL 时也会快速返回有效问题的降级报告，但不能将全零结果当成模型基线。

### RAG 证据质量

检索结果不是简单的「向量距离最小的前 N 条」：

- **RRF 融合**：dense / sparse / entity / structured 四路候选按倒数排名融合，而不是直接比较距离；
- **权威性加权**：`official` 高于 `community`；
- **时效性加权**：按 `updated_at` 指数衰减，权重被限制在 ±10%，只用于打破近似平局，不会推翻明显更匹配的文本；
- **冲突检测**：按 `source_id`（描述对象）与「带标签的事实」分组比较，因此
  「门票 190 元、观光车票 90 元」**不会**被误判为冲突。区分两种严重级别：
  `conflict`（不同文档对同一事实给出不同值）与 `ambiguous`（同一文档给出旺季/淡季等多个取值，
  属于条件差异，回答必须说明适用条件）。出现冲突时会下调 `confidence` 并在结果中给出 `advice`。

### 运行时可靠性

- **审计与会话写入不阻塞事件循环**：`backend/app/core/db_write.py` 用有界队列 + 单后台线程承载
  `agent_runs` / `agent_steps` / `agent_delegations` / `run_traces` / `conversation_messages` 的写入。
  psycopg 是同步驱动，此前在事件循环内直接建连会让一个 ReAct 循环卡住 30 秒并耗尽 45 秒请求预算。
  数据库不可用时按队列丢弃审计明细（观测数据可以让步，访客的回答不能）。
- **熔断**：`database_available()` 只在重试窗口内信任一次成功探测；写入线程在真实建连失败后打开熔断，
  停止对每个任务重复重试。`probe_connection()` 探测归一化后的主机地址，避免 `localhost` 对每个解析地址
  各等一次 `connect_timeout`。
- 失败降级：`_salvage` 路径自身也是防御性的——错误处理器崩溃会让「降级回答」变成「没有回答」。

游客反馈先进入候选知识，管理员采纳后才创建正式文档和 `embed_document` 任务。配置有效的 `EMBEDDING_BASE_URL` 与 `EMBEDDING_API_KEY` 后，可启动 arq worker：

```powershell
$env:PYTHONPATH = "backend"
arq app.worker.WorkerSettings
```

未配置 Key 时任务会保持 `waiting_for_configuration`，不会生成伪造向量。也可以手动触发文档重建索引：

```text
POST /api/v1/admin/documents/{document_id}/reindex
```

Redis 用于聊天/RAG 缓存、限流和 arq 队列；Redis 不可用时会自动降级到进程内缓存，数据库中的会话、审计和任务记录仍然保留。

## 管理员

首次后端启动时使用环境变量创建管理员。示例账号：`admin@jiuzhaigou.local`，密码为运行时设置的 `ADMIN_PASSWORD`。

## 数据

| 文件 | 内容 |
|---|---|
| `data/jiuzhaigou/facts/*.json` | 人工整理的九寨沟公开事实（园区、40 个景点、37 项设施、观光车与栈道衔接、24 条官方问答、20 篇景区知识、规则政策、28 条来源） |
| `data/jiuzhaigou/*.json(l)` | 由 `scripts/generate_jiuzhaigou_data.py` 生成的入库数据集 |
| `data/jiuzhaigou/_research_raw.md` | 调研原始底稿（逐条标注来源 URL 与资料日期） |

每条记录都带 `data_source` 字段：`official` / `third_party` / `derived` / `derived_simulation`，
游客反馈等演示数据一律标记为 `derived_simulation` 并带 `is_simulated` 与免责说明。

## 测试

无需数据库即可运行全部单元与回归测试（约 13 秒）：

```powershell
python -m pytest -q                     # 无数据库：123 passed、5 skipped；完整基础设施：128 passed
```

需要 PostgreSQL + Redis 的接口冒烟测试：

```powershell
python -m pytest backend/tests -v       # 数据库不可用时会显式 skip，不会挂起
```

关键回归用例（都是「曾经静默通过」的缺陷，见
[实施计划](docs/superpowers/plans/2026-09-19-milvus-react-supervisor.md) Task 7–8）：

| 文件 | 保护的契约 |
|---|---|
| `tests/test_answer_path.py` | 请求整体可用：状态必须是 `success`、必须带引用、且不得退化成「暂未查询到足够信息」；单请求 < 2 秒（上限远低于 3 秒 `connect_timeout`，一旦有同步数据库调用重新回到事件循环就会失败） |
| `tests/test_db_write_queue.py` | 写入不阻塞调用方、后台线程真实执行、熔断生效、审计写入绝不在调用线程上建连 |
| `tests/test_route_constraints.py` | 季节闭园基线：**同一日期**下规划与校验必须一致；不给日期时校验保持保守口径 |
| `tests/test_intent_routing.py` | 反馈/投诉分流，包含「厕所不能使用」这类无「反馈」字样的设施问题，以及不应误判的反例 |
| `tests/test_rag_fusion.py` | 时效性权重有界、同标签不同值才算冲突、「门票 190 元 + 观光车票 90 元」不是冲突 |
| `tests/test_rag_event_loop.py` | 检索（含 embedding HTTP 与 Milvus）不得阻塞事件循环；并发检索必须重叠 |
| `tests/test_audit_regressions.py` | 路线参数必须来自游客原话而非内部指令；强制检索策略按游客问题判定；畸形模型响应只能变成 `LLMUnavailable`；检索证据必须能到达回答层 |
| `tests/test_health_endpoint.py` | 健康检查必须暴露写入队列与熔断状态 |

## 压测

```powershell
locust -f loadtests/locustfile.py --host http://127.0.0.1:8000
```

打开 `http://localhost:8089` 并逐步从 10、50、100 用户增加压测。

也可以按场景单独运行：

```powershell
locust -f loadtests/locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 4 -t 1m --tags cache
locust -f loadtests/locustfile.py --host http://127.0.0.1:8000 --headless -u 10 -r 2 -t 1m --tags sse
```

`browse`、`recommend`、`cache`、`model`、`sse` 分别对应浏览、路线、缓存命中、模型问答和流式问答。报告中重点关注失败率、P95、RPS；Agent 运行记录中的 `cache_hit` 可核对缓存命中。
