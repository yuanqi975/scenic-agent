"""Central runtime configuration.

Every environment variable the backend reads is declared here exactly once so that
``main.py`` stays a thin composition root and tests can reason about defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - requirements include python-dotenv
    dotenv_values = None

# ``backend/app/core/config.py`` -> repository root is three levels up.
ROOT = Path(__file__).resolve().parents[3]
_LOCAL_DOTENV = dotenv_values(ROOT / ".env") if dotenv_values else {}


def _env_or_local(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    return str(_LOCAL_DOTENV.get(name) or default)


def _int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- infra
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:123456@localhost:5432/scenic_agent")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PARK_ID = os.getenv("PARK_ID", "jiuzhaigou_scenic_area")
AMAP_WEATHER_KEY = _env_or_local("AMAP_WEATHER_KEY").strip()
AMAP_WEATHER_CITY = _env_or_local("AMAP_WEATHER_CITY", "513225").strip()
AMAP_API_BASE_URL = _env_or_local("AMAP_API_BASE_URL", "https://restapi.amap.com").rstrip("/")

#: Published service-centre phone number used in visitor-facing fallback wording.
SERVICE_PHONE = os.getenv("SERVICE_PHONE", "0837-7739753")

# --------------------------------------------------------------------------- auth
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret-in-production")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@jiuzhaigou.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123456")

# --------------------------------------------------------------------------- llm
# ``agent``    -> full multi-agent system driven by a real model
# ``fallback`` -> same multi-agent skeleton driven by deterministic rules
# ``single``   -> legacy single-pass pipeline, kept for before/after comparison
LLM_MODE = os.getenv("LLM_MODE", "fallback")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6-terra")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", "0.2")
LLM_MAX_CONCURRENCY = _int("LLM_MAX_CONCURRENCY", "2")
LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", "30")
LLM_CIRCUIT_COOLDOWN_SECONDS = _int("LLM_CIRCUIT_COOLDOWN_SECONDS", "60")

# --------------------------------------------------------------------------- embeddings
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = _int("EMBEDDING_DIMENSION", "2560")

# Optional cross-encoder/API reranker. SiliconFlow's /rerank endpoint is compatible
# with this shape; when unset, the hybrid service uses a deterministic lexical rerank.
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", EMBEDDING_BASE_URL).rstrip("/")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", EMBEDDING_API_KEY)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TIMEOUT_SECONDS = _float("RERANK_TIMEOUT_SECONDS", "3")

# --------------------------------------------------------------------------- rag / milvus
# Deployments use ``RAG_BACKEND=milvus`` (see .env.example and Compose): Milvus
# supplies Dense + native BM25 while PostgreSQL contributes entity/structured
# evidence. The no-environment fallback stays PostgreSQL so scripts and tests do not
# pay a Milvus connection/import penalty before a deployment has opted in.
RAG_BACKEND = os.getenv("RAG_BACKEND", "postgres").strip().lower()
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530").rstrip("/")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "scenic_knowledge")
MILVUS_TIMEOUT_SECONDS = _float("MILVUS_TIMEOUT_SECONDS", "3")
# Retrieval needs a wider candidate pool than the visitor-facing top-k.  RRF and the
# reranker cannot recover a relevant document which was cut before fusion, especially
# for park-level questions whose evidence shares one source id.
RAG_DENSE_TOP_K = _int("RAG_DENSE_TOP_K", "40")
RAG_SPARSE_TOP_K = _int("RAG_SPARSE_TOP_K", "40")
RAG_ENTITY_TOP_K = _int("RAG_ENTITY_TOP_K", "40")
RAG_MAX_QUERY_VARIANTS = _int("RAG_MAX_QUERY_VARIANTS", "4")
RAG_RERANK_CANDIDATES = _int("RAG_RERANK_CANDIDATES", "40")

# --------------------------------------------------------------------------- cache / limits
CACHE_TTL_SECONDS = _int("CACHE_TTL_SECONDS", "300")
RATE_LIMIT_PER_MINUTE = _int("RATE_LIMIT_PER_MINUTE", "60")
RATE_LIMIT_IP_PER_MINUTE = _int("RATE_LIMIT_IP_PER_MINUTE", "600")
RATE_LIMIT_RECOMMEND_PER_MINUTE = _int("RATE_LIMIT_RECOMMEND_PER_MINUTE", str(min(30, RATE_LIMIT_PER_MINUTE)))
RATE_LIMIT_STREAM_PER_MINUTE = _int("RATE_LIMIT_STREAM_PER_MINUTE", str(RATE_LIMIT_PER_MINUTE))
SSE_MAX_CONCURRENCY = _int("SSE_MAX_CONCURRENCY", "100")
DB_POOL_SIZE = _int("DB_POOL_SIZE", "10")
DB_MAX_OVERFLOW = _int("DB_MAX_OVERFLOW", "20")
DB_POOL_TIMEOUT_SECONDS = _float("DB_POOL_TIMEOUT_SECONDS", "5")
ADMIN_LOGIN_RATE_LIMIT = _int("ADMIN_LOGIN_RATE_LIMIT", "8")
ADMIN_LOGIN_WINDOW_SECONDS = _int("ADMIN_LOGIN_WINDOW_SECONDS", "300")
CORS_ORIGINS = tuple(
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
)

# --------------------------------------------------------------------------- multi-agent
# ``auto`` follows LLM_MODE; ``multi`` forces the supervisor path; ``single`` forces legacy.
AGENT_MODE = os.getenv("AGENT_MODE", "auto")
AGENT_MAX_STEPS = _int("AGENT_MAX_STEPS", "6")
AGENT_MAX_DELEGATION_ROUNDS = _int("AGENT_MAX_DELEGATION_ROUNDS", "3")
AGENT_MAX_AGENTS = _int("AGENT_MAX_AGENTS", "4")
AGENT_MAX_CONCURRENCY = _int("AGENT_MAX_CONCURRENCY", "4")
AGENT_MAX_TOTAL_TOOL_CALLS = _int("AGENT_MAX_TOTAL_TOOL_CALLS", "15")
AGENT_TOOL_TIMEOUT_SECONDS = _float("AGENT_TOOL_TIMEOUT_SECONDS", "8")
AGENT_REQUEST_TIMEOUT_SECONDS = _float("AGENT_REQUEST_TIMEOUT_SECONDS", "45")
AGENT_TRACE_ENABLED = _bool("AGENT_TRACE_ENABLED", "true")
AGENT_TOOL_OUTPUT_MAX_CHARS = _int("AGENT_TOOL_OUTPUT_MAX_CHARS", "2000")
AGENT_REVIEWER_ENABLED = _bool("AGENT_REVIEWER_ENABLED", "false")

# --------------------------------------------------------------------------- retrieval
RETRIEVAL_TOP_K = _int("RETRIEVAL_TOP_K", "5")
RETRIEVAL_MAX_PER_SOURCE = _int("RETRIEVAL_MAX_PER_SOURCE", "2")
# Cosine distance ceiling; 1.0 means "accept anything" (pure vector order).
RETRIEVAL_MAX_DISTANCE = _float("RETRIEVAL_MAX_DISTANCE", "0.72")
RAG_FINAL_TOP_K = _int("RAG_FINAL_TOP_K", str(RETRIEVAL_TOP_K))


def resolved_agent_mode() -> str:
    """Return ``agent`` or ``fallback``: the brain actually used for a request."""
    if AGENT_MODE == "single" or LLM_MODE == "single":
        return "single"
    if LLM_MODE == "agent" and LLM_BASE_URL and LLM_API_KEY:
        return "agent"
    if AGENT_MODE == "multi" and LLM_MODE == "agent":
        # Explicitly requested multi-agent but no usable model: degrade loudly.
        return "fallback"
    return "fallback"


def llm_configured() -> bool:
    return bool(LLM_BASE_URL and LLM_API_KEY)


def production_config_errors() -> list[str]:
    """Return actionable deployment errors without leaking secret values."""
    if ENVIRONMENT not in {"production", "prod"}:
        return []
    errors: list[str] = []
    if JWT_SECRET in {"", "change-this-secret-in-production"} or len(JWT_SECRET) < 32:
        errors.append("JWT_SECRET must be a random value of at least 32 characters")
    if ADMIN_PASSWORD in {"", "admin123456"}:
        errors.append("ADMIN_PASSWORD must not use the development default")
    if "123456" in DATABASE_URL or "postgres:postgres" in DATABASE_URL:
        errors.append("DATABASE_URL must not use the development database password")
    if not CORS_ORIGINS or "*" in CORS_ORIGINS:
        errors.append("CORS_ORIGINS must be an explicit allowlist")
    if LLM_MODE == "fallback":
        errors.append("LLM_MODE=fallback is not allowed in production")
    return errors


def validate_production_config() -> None:
    errors = production_config_errors()
    if errors:
        raise RuntimeError("Invalid production configuration: " + "; ".join(errors))
