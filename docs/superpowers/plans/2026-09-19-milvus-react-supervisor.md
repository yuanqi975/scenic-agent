# Milvus ReAct Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将项目升级为可灰度切换的 Milvus 混合 RAG、策略约束 ReAct 和 Supervisor 多 Agent 架构。

**Architecture:** PostgreSQL 保留为事实与事务数据库，Milvus 作为文档向量索引；统一 `RagRetriever` 接口通过 `postgres`、`shadow`、`milvus` 三种后端工作。Supervisor 负责任务编排，Specialist Agent 执行受预算和强制工具策略约束的 ReAct。

**Tech Stack:** FastAPI, Python 3.13, SQLAlchemy, PostgreSQL/pgvector, Milvus 2.5, Redis, Pydantic, pytest, Docker Compose。

**Spec:** `docs/superpowers/specs/2026-09-19-milvus-react-supervisor-design.md`

## Global Constraints

- PostgreSQL remains the source of truth for structured facts and writes.
- Milvus is optional at startup; unavailable Milvus must fall back to PostgreSQL.
- `rag_search` is the canonical RAG tool and `search_knowledge` remains backward compatible.
- High-risk intents cannot bypass required retrieval tools.
- Existing SSE response fields and fallback behavior remain compatible.
- Every production behavior change gets a failing test before implementation.

### Task 1: Add infrastructure and configuration

**Files:**
- Modify: `docker-compose.yml`
- Modify: `requirements.txt`
- Modify: `backend/app/core/config.py`
- Test: `tests/test_infrastructure_config.py`

- [x] Add failing tests for Milvus environment defaults and Compose service names.
- [x] Run the focused tests and confirm they fail because the settings/services are absent.
- [x] Add optional `pymilvus`, Milvus endpoint/collection/backend settings, and standalone Milvus dependencies to Compose.
- [x] Run the focused tests and then validate Compose YAML parsing.

### Task 2: Introduce the RAG service boundary

**Files:**
- Create: `backend/app/services/rag.py`
- Modify: `backend/app/services/retrieval.py`
- Test: `tests/test_rag_service.py`

- [x] Add failing tests for RRF fusion, authority weighting, deduplication, and backend fallback.
- [x] Implement `RagHit`, `RagSearchRequest`, `RagRetriever`, `PostgresRetriever`, `MilvusRetriever`, and `HybridRagService`.
- [x] Keep existing PostgreSQL retrieval behavior behind `PostgresRetriever`.
- [x] Add optional Milvus dense/sparse/entity searches with graceful unavailable handling.
- [x] Wire the legacy `retrieve()` function through the configured service.
- [x] Run focused RAG tests.

### Task 3: Make RAG a canonical tool

**Files:**
- Modify: `backend/app/tools/knowledge.py`
- Modify: `backend/app/tools/registry.py`
- Test: `tests/test_rag_tool.py`

- [x] Add failing tests asserting `rag_search` returns evidence, citations, channel metadata and conflict markers.
- [x] Implement `rag_search` with typed filters and preserve `search_knowledge` as an alias.
- [x] Ensure tool output remains untrusted and injection-sanitized.
- [x] Run focused tool tests.

### Task 4: Enforce retrieval policy in ReAct

**Files:**
- Create: `backend/app/agents/policy.py`
- Modify: `backend/app/agents/runtime.py`
- Test: `tests/test_agent_policy.py`

- [x] Add failing tests for mandatory retrieval on high-risk questions and route planning.
- [x] Implement policy functions that return required tool names and structured arguments.
- [x] Execute required tools before accepting a model final answer when no evidence exists.
- [x] Preserve whitelist, repeated-call and budget guards.
- [x] Run focused policy and existing multi-agent tests.

### Task 5: Fix deterministic route constraints and evaluation hooks

**Files:**
- Modify: `backend/app/services/route_algo.py`
- Create: `backend/app/services/evaluation.py`
- Test: `tests/test_route_constraints.py`

- [x] Add a failing regression test for seasonal closure and off-standard-route attractions.
- [x] Make route candidate filtering apply closure/status/standard-route constraints before ranking.
- [x] Add a small retrieval/groundedness evaluation helper for dataset questions.
- [x] Run route and data tests.

### Task 6: Update deployment and operator documentation

**Files:**
- Modify: `README.md`
- Modify: `README_DATA.md`
- Create: `.env.example`
- Test: `tests/test_deployment_docs.py`

- [x] Add failing checks for documented RAG backend and Milvus startup commands.
- [x] Document Docker startup, shadow mode, collection initialization and rollback.
- [x] Add environment variable examples without real credentials.
- [x] Run the full test suite and a Compose configuration check.

## Task 7: Repair the runtime correctness defects (added after the first review)

The tasks above built the architecture but the suite could not actually exercise it.
The full suite raised a collection error and, once that was repaired, hung for ten
minutes and still failed. Every defect below was found by **running the request path**,
not by reading it - each one is now covered by a test that drives `answer_async` the way
the API does.

**Files:**
- Create: `backend/app/core/db_write.py`
- Modify: `backend/app/services/audit.py`
- Modify: `backend/app/services/conversations.py`
- Modify: `backend/app/services/orchestrator.py`
- Modify: `backend/app/core/db.py`
- Modify: `backend/app/services/intent.py`
- Modify: `backend/app/services/brains.py`
- Modify: `backend/app/services/retrieval.py`
- Modify: `backend/app/agents/runtime.py`
- Modify: `backend/app/tools/route.py`
- Modify: `tests/fakes.py`
- Test: `tests/test_db_write_queue.py`, `tests/test_answer_path.py`,
  `tests/test_intent_routing.py`, `tests/test_rag_fusion.py`,
  `tests/test_route_constraints.py`

- [x] **Syntax error**: `backend/app/services/intent.py` had a `'''legacy_classifier_body_removed`
      block left open, so the module failed to import and three test modules could not be
      collected at all.
- [x] **Blocking persistence**: audit and conversation writers opened a synchronous
      PostgreSQL connection *inline on the asyncio event loop*. One audit step cost 6 s
      and one ReAct loop 30 s, blowing the 45 s budget. All writers now hand a callable to
      `db_write.enqueue`; one daemon thread owns the connection.
- [x] **Availability false positive**: `database_available()` cached a single success
      forever and treated an open TCP port as proof of a reachable server, so a dropped
      handshake passed every check. It now re-probes after one retry window, and the
      writer thread opens a circuit breaker on a *real* connection failure.
- [x] **`probe_connection` double timeout**: probing `localhost` waited out one
      `connect_timeout` per resolved address (6 s for a 3 s setting). It now probes the
      normalised host.
- [x] **`AttributeError` on every request**: `_multi_agent_pass` called `.get("agent")`
      on `AgentResult` objects, so `status` was always `error`. `_salvage` also spread a
      non-iterable into the result, turning a degraded answer into a crash.
- [x] **Retrieval searched the wrong text**: the rule brain passed the supervisor's
      *instruction* to `search_knowledge`, so the deterministic path never retrieved
      anything and answered "暂未查询到足够信息" for questions the knowledge base answers.
      The visitor's question is now threaded into `Brain.decide`.
- [x] **Reader captured at import time**: `retrieve` and `known_entity_ids` bound their
      catalog reader as a default argument, so an injected reader was ignored and every
      call paid ~12 s of connect timeouts. Both now resolve the reader at call time.
- [x] **Test double gaps**: `FakeConnection` had no context-manager protocol, so `with
      engine.connect()` raised into a bare `except` and every query silently returned
      `[]` - assertions like "no results" passed for the wrong reason. The fake also did
      not cover the conversation reader.
- [x] **Seasonal-closure determinism**: `calculate_route` started honouring the seasonal
      window while `validate_route` still treated *any* published closure as a violation,
      so a plan could be reported as both valid and invalid depending on the calendar.
      `travel_date` is now plumbed through planning, validation and the rule brain.
- [x] **Complaint misrouting**: "游客中心的厕所不能使用" was classified as a knowledge
      question, so a facility complaint never reached human review.
- [x] **RAG conflict detection and freshness**: `fuse_hits` applied equal weight to two
      official pages that disagree, and `conflicts` was hardcoded to `[]`. Conflicts are
      now detected per subject and per *labelled* fact (so "门票 190 元、观光车票 90 元"
      is not a conflict), and freshness breaks near-ties within a bounded range.

**Result:** `91 passed, 5 skipped in 10.8s` (previously `2 failed, 49 passed` after a
10-minute run, and 3 collection errors before that).

## Task 8: Second audit pass (independent review)

Task 7 was repaired by running the code. Task 8 was an **independent adversarial review**
of the same defect classes across the whole codebase. It found that the first repair had
been applied to the *tool arguments* in one place but not the *parameter extraction* in
another, and that several blocking reads were still inline. Every finding below was
reproduced before being fixed.

**Files:**
- Modify: `backend/app/services/brains.py`
- Modify: `backend/app/agents/runtime.py`
- Modify: `backend/app/services/orchestrator.py`
- Modify: `backend/app/services/llm.py`
- Modify: `backend/app/tools/realtime.py`
- Modify: `backend/app/api/chat.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/services/rag.py`
- Test: `tests/test_audit_regressions.py`

- [x] **Route parameters came from the instruction, not the visitor** (highest severity
      finding). `tool_arguments` passed the supervisor's instruction
      (`"按游客的时长、人群、天气与设施要求规划并校验路线"`) to `route_arguments`, whose regexes
      found no duration, audience or preference in it. In the **default**
      `LLM_MODE=fallback` configuration every route request therefore planned on the
      240-minute default with no filters, and then validated that plan against itself, so
      nothing ever surfaced the loss. Reproduced: 「带老人玩整整一天，想看瀑布」 planned 240
      minutes (the visitor's wording yields 480 / 老年游客 / 瀑布).
- [x] **Retrieval policy keyed on the instruction**. The mandated-evidence rule
      (`agents/runtime.py`) ran its topic regex against the supervisor's instruction, so a
      generically worded delegation ("回答游客的问题") skipped the forced `rag_search` and a
      ticket-price question could be answered from the model's own memory.
- [x] **`agent_cards()` was broken on every call**: it read `card.__dict__` on a
      `@dataclass(slots=True)` type, which has none.
- [x] **Malformed model responses escaped as `AttributeError`** instead of the documented
      `LLMUnavailable`, collapsing the run to the catch-all salvage path and bypassing
      `complete_json`'s retry loop.
- [x] **Three more blocking reads on the event loop**: `recent_history` and
      `create_conversation` per request, `payload_rows` inside the async legacy composer,
      and `audit.run_tree` on the debug-trace path.
- [x] **The rule brain could report a query bug as "no data source"**: `tools/realtime.py`
      swallowed any exception from an already-guarded query, so a broken query told the
      visitor 「九寨沟景区尚未接入天气与客流实时数据源」 and reported
      `attractions_checked: 0` ("nothing closed"). The registry's own handlers now turn it
      into a visible `error`.
- [x] **Retrieved documents never reached the legacy composer**: tool output stayed in
      each agent's local list, so `retrieval_items` read a key that never existed and
      always returned `[]`. `RunContext` now records per-tool evidence (`tool_evidence`).
- [x] **Milvus client had no timeout** on construction or search, so an unreachable Milvus
      could stall the health probe and the first retrieval well past the request budget.

**Result:** `112 passed, 5 skipped in 12.9s`.
