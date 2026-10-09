"""Application composition root.

``main.py`` used to hold the whole system; it now wires the pieces together and keeps
re-exporting the historical symbols so existing callers, tests and the worker module
continue to work unchanged.

Run locally with::

    uvicorn app.main:app --app-dir backend --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agents.registry import AGENT_CARDS, naming_contract  # noqa: F401  (admin/introspection)
from .api import admin as admin_api
from .api import catalog as catalog_api
from .api import chat as chat_api
from .api import feedback as feedback_api
from .api import metrics as metrics_api
from .api import weather as weather_api
from .api.admin import LoginRequest  # noqa: F401  (re-exported)
from .api.catalog import RecommendationRequest  # noqa: F401  (re-exported)
from .api.chat import ChatRequest  # noqa: F401  (re-exported)
from .api.feedback import FeedbackRequest  # noqa: F401  (re-exported)
from .core import config
from .core import db_write
from .core.cache import cache_get, cache_set, enforce_public_rate_limit, redis_client  # noqa: F401
from .services.llm import get_client as get_llm_client
from .core.db import (  # noqa: F401
    RUNTIME_TABLES,
    db_ready,
    ensure_runtime_tables,
    execute,
    execute_write,
    payload_rows,
)
from .core.events import enqueue_embedding_task  # noqa: F401
from .core.security import admin_from_token  # noqa: F401
from .services import audit  # noqa: F401
from .services import brains, intent, retrieval
from .services.rag import get_rag_service
from .services.orchestrator import (  # noqa: F401
    answer_async,
    build_recommendation as _orchestrated_recommendation,
    exact_answer,
    generate_answer,
    retrieve_for_state,
)
from .services.route_algo import facilities_for_attraction, greedy_itinerary, mentioned_entity_ids as _mentioned
from .tools import registry as tool_registry  # noqa: F401  (import registers every tool)

# --------------------------------------------------------------------------- config aliases
# Re-exported so the documented environment surface (and the deployment identity test)
# stays exactly where it always was.
PARK_ID = os.getenv("PARK_ID", "jiuzhaigou_scenic_area")
DATABASE_URL = config.DATABASE_URL
REDIS_URL = config.REDIS_URL
JWT_SECRET = config.JWT_SECRET
ADMIN_EMAIL = config.ADMIN_EMAIL
ADMIN_PASSWORD = config.ADMIN_PASSWORD
LLM_MODE = config.LLM_MODE
LLM_BASE_URL = config.LLM_BASE_URL
LLM_API_KEY = config.LLM_API_KEY
LLM_MODEL = config.LLM_MODEL
EMBEDDING_BASE_URL = config.EMBEDDING_BASE_URL
EMBEDDING_API_KEY = config.EMBEDDING_API_KEY
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION
CACHE_TTL_SECONDS = config.CACHE_TTL_SECONDS
RATE_LIMIT_PER_MINUTE = config.RATE_LIMIT_PER_MINUTE
AGENT_MODE = config.AGENT_MODE

engine = __import__("app.core.db", fromlist=["engine"]).engine
_llm_sem = asyncio.Semaphore(config.LLM_MAX_CONCURRENCY)

app = FastAPI(title="九寨沟景区智能服务 Agent", version="2.0.0")
app.middleware("http")(metrics_api.metrics_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)

app.include_router(chat_api.router, prefix="/api/v1")
app.include_router(catalog_api.router, prefix="/api/v1")
app.include_router(feedback_api.router, prefix="/api/v1")
app.include_router(admin_api.router, prefix="/api/v1")
app.include_router(metrics_api.router, prefix="/api/v1")
app.include_router(weather_api.router, prefix="/api/v1")


@app.on_event("startup")
def startup() -> None:
    config.validate_production_config()
    if db_ready():
        ensure_runtime_tables()


@app.get("/api/v1/health")
def health():
    """Liveness plus the *effective* brain, so the UI never overstates the mode."""
    available = db_ready()
    mode = config.resolved_agent_mode()
    return {
        "status": "healthy" if available else "degraded",
        "database": available,
        "park_id": PARK_ID,
        "llm_mode": LLM_MODE,
        "llm_provider": get_llm_client().provider_status(),
        "agent_mode": mode,
        "multi_agent": mode != "single",
        "degraded": mode != "agent",
        "tools": tool_registry.tool_names(),
        "agents": list(AGENT_CARDS),
        "naming_contract": naming_contract(),
        "rag_backend": config.RAG_BACKEND,
        "milvus": _milvus_health(),
        # Audit/conversation writes are queued to a background writer, so "the database
        # is up" is not the whole story: these counters are how an operator sees that
        # detail is being shed (and why) without reading logs.
        "write_queue": _write_queue_health(),
    }


def _write_queue_health() -> dict[str, Any]:
    """Counters for the background audit/conversation writer.

    ``dropped`` growing while ``database`` is true means writes are being shed for
    another reason (a full queue); ``breaker.open`` with a false ``database`` is the
    normal unreachable-database story and needs no action.
    """
    return {
        "breaker": db_write.breaker_state(),
        "queues": db_write.stats(),
    }


def _milvus_health() -> dict[str, Any]:
    """Non-blocking-ish Milvus status for operators; RAG still falls back safely."""
    try:
        client = get_rag_service().primary._client_or_none()  # type: ignore[attr-defined]
        if client is None:
            return {"available": False, "reason": "client_unavailable"}
        return {"available": bool(client.has_collection(config.MILVUS_COLLECTION)), "collection": config.MILVUS_COLLECTION}
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


# --------------------------------------------------------------------------- compatibility
# These helpers are the historical public API. ``classify`` and ``mentioned_entity_ids``
# are delegated to the modules that now own them, so callers keep working while the
# behaviour lives in exactly one place.
classify = intent.classify
retrieve = retrieval.retrieve


def mentioned_entity_ids(message: str) -> list[str]:
    """POI ids literally named in the question (kept for backwards compatibility)."""
    return _mentioned(message, payload_rows("attractions", 100), payload_rows("facilities", 100))


def build_recommendation(request: RecommendationRequest | None = None, **overrides: Any) -> dict[str, Any]:
    """Deterministic recommendation, callable with either the model object or kwargs.

    The standalone endpoint, the agent tool and the legacy tests all funnel into the
    same algorithm so they can never disagree.
    """
    payload = request or RecommendationRequest(**overrides)
    return _orchestrated_recommendation(
        duration_minutes=payload.duration_minutes,
        groups=payload.groups,
        preferences=payload.preferences,
        weather=payload.weather,
        difficulty=payload.difficulty,
        # Read through this module's ``payload_rows`` so callers that patch it (tests,
        # scripts, future caching layers) keep working.
        attractions=payload_rows("attractions", 100),
    )


def answer_sync(request: ChatRequest | None = None, **overrides: Any) -> dict[str, Any]:
    """Blocking entry point kept for scripts and existing tests.

    Inside a running event loop it runs in a worker thread, so it stays safe to call from
    async code without deadlocking the loop.
    """
    payload = request or ChatRequest(**overrides)
    coroutine = answer_async(payload.message, payload.conversation_id, mode=payload.mode)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coroutine)).result()


def save_run(
    conversation_id: str,
    intent_name: str,
    agent: str,
    docs: list[str],
    latency: int,
    cache_hit: bool = False,
    status: str = "success",
    error: str | None = None,
) -> None:
    """Legacy audit writer: one flat ``agent_runs`` row per answer."""
    import json as _json

    from .core.db import engine as _engine
    from sqlalchemy import text as _text

    try:
        with _engine.begin() as conn:
            conn.execute(
                _text(
                    """INSERT INTO agent_runs(run_id, conversation_id, park_id, intent, agent_name,
                           retrieved_document_ids, cache_hit, latency_ms, status, error, agent_role, created_at)
                       VALUES (:id,:cid,:park_id,:intent,:agent,CAST(:docs AS jsonb),:cache,:latency,:status,:error,
                               'legacy_summary', now())"""
                ),
                {
                    "id": __import__("uuid").uuid4().hex,
                    "cid": conversation_id,
                    "park_id": PARK_ID,
                    "intent": intent_name,
                    "agent": agent,
                    "docs": _json.dumps(docs),
                    "cache": cache_hit,
                    "latency": latency,
                    "status": status,
                    "error": error,
                },
            )
    except Exception:
        pass


__all__ = [
    "AGENT_CARDS",
    "ChatRequest",
    "FeedbackRequest",
    "LoginRequest",
    "RecommendationRequest",
    "agent_cards",
    "answer_async",
    "answer_sync",
    "app",
    "audit",
    "brains",
    "build_recommendation",
    "cache_get",
    "cache_set",
    "classify",
    "db_ready",
    "engine",
    "ensure_runtime_tables",
    "enqueue_embedding_task",
    "exact_answer",
    "generate_answer",
    "mentioned_entity_ids",
    "naming_contract",
    "payload_rows",
    "retrieve",
    "retrieve_for_state",
    "save_run",
]


def agent_cards() -> dict[str, Any]:
    """Introspection helper for the docs and the admin panel.

    ``AgentCard`` is a slotted dataclass, so it has no ``__dict__``; ``asdict`` is the
    only correct way to serialise it. The previous version raised ``AttributeError`` on
    every call.
    """
    from dataclasses import asdict

    return {name: asdict(card) for name, card in AGENT_CARDS.items()}
