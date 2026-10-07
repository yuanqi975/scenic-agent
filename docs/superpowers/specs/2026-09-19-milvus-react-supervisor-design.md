# Milvus Hybrid RAG + ReAct Supervisor Design

## Goal

将当前九寨沟 Agent 链路收敛为“策略约束的 Supervisor + Specialist ReAct + 混合 RAG 工具 + PostgreSQL 事实库 + Milvus 向量索引”，并保留 PostgreSQL 检索作为兼容与影子对照路径。

## Decisions

- PostgreSQL 继续保存事务事实、对话、反馈、审计和结构化业务数据。
- Milvus 负责文档 chunk 的 dense/sparse 检索索引；Milvus 不替代事务数据库。
- `rag_search` 是唯一面向 Agent 的统一 RAG 工具；`search_knowledge` 保留为兼容别名。
- 检索结果采用 dense、sparse、entity、structured 四路候选，经 RRF、权威性/时效性加权、去重和冲突标记后返回。
- `RAG_BACKEND=postgres|shadow|milvus` 控制迁移阶段；Milvus 不可用时安全回退 PostgreSQL。
- 高风险意图必须检索；模型不能通过直接回答绕过强制工具策略。
- Supervisor 只负责计划、依赖批次和汇总；Specialist Agent 执行 ReAct。

## Non-goals

- 本次不替换现有 LLM provider。
- 本次不把 PostgreSQL 结构化查询迁移到 Milvus。
- 本次不修改前端 SSE 事件协议。

## Success criteria

- Docker Compose 能定义 Milvus、etcd、MinIO，并保留现有服务。
- 无 Milvus、无 embedding 配置时，现有测试和 PostgreSQL/fallback 路径仍可工作。
- Milvus 配置存在时，RAG 服务能执行混合检索并返回统一证据结构。
- 高风险知识问题会执行 RAG 工具并产生引用。
- 季节闭园和路线约束回归测试通过。
