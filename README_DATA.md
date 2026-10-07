# 九寨沟风景名胜区知识库

`park_id = jiuzhaigou_scenic_area`

本目录是九寨沟景区智能服务 Agent 的知识库数据。与早期的纯合成数据不同，**景点、设施、票价、开放时间、
优惠政策、游览规则、官方问答与景区级知识文档都来自公开可查证的九寨沟资料**，并逐条标注来源；
只有演示与压测必需的派生数据（景点级问答、栈道距离与时长、评测题、游客反馈）由程序生成。

## 数据来源与可信度分层

每条记录都带 `data_source` 字段，取值含义如下：

| `data_source` | 含义 | 出现范围 |
|---|---|---|
| `official` | 来自官方来源（九寨沟景区官网、四川省/阿坝州政府网站，或权威媒体转述的官方口径） | 园区信息、景点、设施、24 条官方问答、景区级知识文档 |
| `third_party` | 来自可查证但非官方的第三方公开资料 | 观光车站间里程与运行时间（9 条） |
| `derived` | 依据真实数据派生或推算，**不是官方发布内容** | 步行栈道距离/时长/难度、景点级派生问答（80 条）、评测题（121 条） |
| `derived_simulation` | 程序模拟数据，仅用于演示与压测，**不代表真实游客意见** | 游客反馈与候选知识（各 3000 条，带 `is_simulated` 与免责说明） |

来源清单见 `sources.json`（28 条，含 URL、发布方、资料日期与用于哪些内容）与调研原始底稿 `_research_raw.md`。

## 坐标口径（重要）

**官方从未公布任何景点、设施或景区四至的经纬度。** 数据集中的所有 `latitude` / `longitude`
都是按沟谷走向与官方海拔推算的估算值，`coordinate_source` 标记为 `estimated`，精度约 ±0.01°（约 ±1 km），
**不可用于测绘或导航**。官方未公布位置的设施（如九寨娱乐场所与县属医院）坐标留空并标记为 `not_published`，
不做填充。

## 目录结构

```text
data/jiuzhaigou/
├── facts/                     # 人工整理的公开事实（可审阅、可追溯）
│   ├── park.json              # 园区基础信息、票务、开放时间、承载量、口径冲突
│   ├── attractions.json       # 40 个具名景点（所属沟、类型、海拔、坐标、描述、看点）
│   ├── facilities.json        # 37 项真实服务设施
│   ├── transport.json         # 9 条观光车站间里程 + 38 段栈道衔接
│   ├── faqs.json              # 24 条官方高频问答
│   ├── knowledge.json         # 20 篇景区级知识文档
│   ├── policies.json          # 禁止事项、票务与优惠政策、安全提示、交通、线路、轮休保育
│   └── sources.json           # 28 条来源清单
├── _research_raw.md           # 调研原始底稿（912 行，逐条标注来源与资料日期）
├── parks.json                 # ↓ 以下均由生成脚本产出
├── attractions.jsonl
├── facilities.jsonl
├── routes.jsonl
├── faqs.jsonl
├── notices.jsonl                # 带生效/失效时间的官方公告
├── feedbacks.jsonl
├── feedback_candidates.jsonl
├── rag_documents.jsonl
├── evaluation_questions.jsonl
├── evaluation_complex_questions.jsonl
├── evaluation_questions_v2.jsonl
├── sources.json
└── dataset_manifest.json      # 计数、口径说明与已知缺口
```

## 数据规模

| 数据集 | 数量 | 说明 |
|---|---|---|
| 景点 | 40 | 树正沟 13、日则沟 17、则查洼沟 5、扎如沟 5 |
| 服务设施 | 37 | 游客服务中心、检票口、救护中心、诺日朗服务中心、11 个观光车站、11 个停车场、观景台、购物点、医院 |
| 路线 | 47 | 观光车 9 段（第三方里程）+ 步行栈道 38 段（坐标派生） |
| 问答 | 104 | 官方 24 条 + 景点级派生 80 条 |
| 知识文档 | 288 | 园区 20 + 景点 80 + 设施 37 + 路线 47 + 问答 104 |
| 评测题 | 121 + 5 | 基础集 40 景点 + 37 设施 + 24 官方问答 + 20 园区；复杂集 5 条 |
| 官方公告 | 4 | 开放恢复、售罄/候补、旺淡季承载、临时优惠 |
| 游客反馈 | 3000 | 全部为模拟数据，标注 `derived_simulation` |
| 来源 | 28 | 官网、省政府/州政府、四川在线、四川广播电视台、人民网、成都本地宝等 |

## 生成与校验

```powershell
python scripts/generate_jiuzhaigou_data.py           # 重新生成（默认写入 data/jiuzhaigou）
python scripts/generate_jiuzhaigou_data.py --feedback-rows 200   # 只做小幅演示数据
python -m pytest tests -v                            # 数据集契约测试（不依赖数据库）
```

生成过程是确定性的：脚本只读取 `facts/` 下的整理结果，不联网、不引入随机数。

## PostgreSQL + pgvector

## Milvus Index

Milvus is an index, not the source of truth. PostgreSQL retains documents and every
structured business record; Milvus stores a copy of document chunks and embeddings.
Run `scripts/index_milvus.py` after importing or updating the generated dataset. The
script upserts by `document_id:chunk_index`, so it can be safely re-run after updates.

先在启用了 pgvector 的数据库执行 `sql/init.sql`，再导入：

```powershell
python scripts/import_to_postgres.py --dsn "postgresql+psycopg://USER:PASSWORD@localhost:5432/scenic_agent"
```

导入为幂等操作，不删除既有记录。向量维度默认为 2560；若使用已有数据库，请先执行 `sql/migrate_embedding_dimension_2560.sql`，再重新生成全部向量。

## 生成向量

```powershell
$env:EMBEDDING_API_KEY="your-key"
$env:EMBEDDING_BASE_URL="https://api.openai.com/v1"
$env:EMBEDDING_MODEL="text-embedding-3-large"
python scripts/build_embeddings.py --input data/jiuzhaigou/rag_documents.jsonl --dsn "postgresql+psycopg://USER:PASSWORD@localhost:5432/scenic_agent"
```

脚本只更新缺少向量的文档，可安全断点续跑；未配置密钥时会终止且不生成伪造向量。

## 已知缺口与口径冲突

数据集中已显式记录以下问题（见 `dataset_manifest.json` 与 `sources.json`）：

- 官方未公布任何景点与设施的经纬度，坐标为估算值；景区几何中心与四至边界同样为估算。
- 官方未公布卫生间点位清单、行李寄存点位与收费标准、轮椅与婴儿车租用费用。
- 官方未公布观光车完整站点名单，本数据集只收录公开可检索到的 11 个站点。
- 官方未公布秋季彩林精确起止日期，口径为「十月中下旬」。
- 景区救护电话存在 `0837-7738818`（2025-09-27 通告）与 `0837-7739309`（医疗信息页）两个公开口径。
- 门票价格存在冲突：官网 FAQ 页仍保留旺季 220 元旧口径，本数据集采用现行 190 元（2026-03-24 通告）。
- 旺季入园时间存在年度差异：2025 年为 07:30—14:00，2026 年起为 08:00—14:00。
- 季节性轮休保育清单时效区间为 2025-11-16 至 2026-03-31，出行前应以景区最新公告为准。

## 数据时效

资料访问日期为 **2026-09-17**，其中票务、开放时间、限流与轮休保育政策会随官方通告调整，
上述字段请以九寨沟景区最新公告为准。
