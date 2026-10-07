"""Database engine, readiness probe and lazily created runtime tables.

The imported Jiuzhaigou dataset is never altered here: every statement is
``CREATE ... IF NOT EXISTS`` or ``ALTER ... ADD COLUMN IF NOT EXISTS`` so the
service can start against an existing database.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Iterable
from urllib.parse import urlsplit

import bcrypt
from sqlalchemy import create_engine, text

from .config import ADMIN_EMAIL, ADMIN_PASSWORD, DATABASE_URL


def _database_url() -> str:
    """Append short connect/statement timeouts so an unreachable database fails fast.

    Without this, a connect attempt can block for the OS TCP timeout (~2 minutes) even
    when the port is open but the server never completes the handshake, which would
    stall a request instead of degrading. ``connect_timeout`` covers the handshake and
    ``options=-c statement_timeout=...`` bounds any single statement.
    """
    timeout = os.getenv("DB_CONNECT_TIMEOUT", "3")
    statement_ms = int(float(os.getenv("DB_STATEMENT_TIMEOUT_SECONDS", "4")) * 1000)
    try:
        parts = urlsplit(DATABASE_URL)
        query = parts.query or ""
        additions: list[str] = []
        if "connect_timeout" not in query:
            additions.append(f"connect_timeout={timeout}")
        if "statement_timeout" not in query:
            additions.append(f"options=-c%20statement_timeout%3D{statement_ms}")
        if not additions:
            return DATABASE_URL
        separator = "&" if query else "?"
        return f"{DATABASE_URL}{separator}{'&'.join(additions)}"
    except Exception:
        return DATABASE_URL


def _probe_host_port() -> tuple[str, int]:
    """Resolve the host/port used by the fast reachability probe.

    ``localhost`` is deliberately normalised to ``127.0.0.1``: psycopg otherwise treats
    the name as multi-host and waits out one connect timeout per resolved address
    (IPv6 then IPv4), which doubles or triples the stall when no database is running.
    """
    try:
        from sqlalchemy.engine import make_url

        url = make_url(_database_url())
        host = url.host or "localhost"
        if host in {"localhost", "::1"} and os.getenv("DB_KEEP_LOCALHOST", "").lower() not in {"1", "true"}:
            host = "127.0.0.1"
        return host, url.port or 5432
    except Exception:
        return "127.0.0.1", 5432


engine = create_engine(
    _database_url(),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

#: Tri-state availability cache: ``None`` = unknown, ``True``/``False`` = last probe.
#: Once the database is known to be down, reads and writes short-circuit instead of
#: retrying per call - the same graceful-degradation contract the Redis cache follows.
_DB_AVAILABLE: bool | None = None
_DB_RETRY_SECONDS = float(os.getenv("DB_RETRY_SECONDS", "60"))
_DB_RETRY_MAX_SECONDS = float(os.getenv("DB_RETRY_MAX_SECONDS", "600"))
_DB_FAILED_AT: float = 0.0
_DB_FAILURES: int = 0
_DB_PROBED_AT: float = 0.0


def note_database_failure() -> None:
    """Record that an actual connection attempt failed.

    Called by real call sites (the probe and :func:`mark_failed`) so the cooldown
    reflects a genuine failure rather than a guess.
    """
    global _DB_AVAILABLE, _DB_FAILED_AT, _DB_FAILURES
    _DB_AVAILABLE = False
    _DB_FAILED_AT = time.monotonic()
    _DB_FAILURES += 1


def note_database_success() -> None:
    global _DB_AVAILABLE, _DB_FAILURES, _DB_PROBED_AT
    _DB_AVAILABLE = True
    _DB_FAILURES = 0
    _DB_PROBED_AT = time.monotonic()


def _cooldown_seconds() -> float:
    """Exponential backoff: a database that has been down for a while is not retried
    on every request, because each retry can cost a full connect timeout."""
    return min(_DB_RETRY_SECONDS * (2 ** max(0, _DB_FAILURES - 1)), _DB_RETRY_MAX_SECONDS)

#: Tables created at startup. Extended by the multi-agent audit schema.
RUNTIME_TABLES: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS notices (park_id text NOT NULL REFERENCES parks(park_id), notice_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY (park_id, notice_id))""",
    """CREATE TABLE IF NOT EXISTS admin_users (id bigserial primary key, email text unique not null, password_hash text not null, is_active boolean not null default true, created_at timestamptz not null default now())""",
    """CREATE TABLE IF NOT EXISTS conversations (conversation_id text primary key, park_id text not null, visitor_id text, title text, summary text, created_at timestamptz not null default now(), updated_at timestamptz not null default now())""",
    """CREATE TABLE IF NOT EXISTS conversation_messages (id bigserial primary key, conversation_id text not null references conversations(conversation_id), role text not null, content text not null, citations jsonb not null default '[]', agent_name text, created_at timestamptz not null default now())""",
    """ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS agent_name text""",
    """CREATE TABLE IF NOT EXISTS agent_runs (run_id text primary key, conversation_id text, park_id text not null, intent text not null, agent_name text not null, retrieved_document_ids jsonb not null default '[]', cache_hit boolean not null default false, latency_ms integer, status text not null, error text, created_at timestamptz not null default now())""",
    """CREATE TABLE IF NOT EXISTS knowledge_versions (version text primary key, park_id text not null, status text not null, created_at timestamptz not null default now())""",
    """CREATE TABLE IF NOT EXISTS knowledge_reviews (id bigserial primary key, candidate_id text not null, park_id text not null, action text not null, reviewer text not null, created_at timestamptz not null default now())""",
    """CREATE TABLE IF NOT EXISTS async_tasks (task_id text primary key, park_id text not null, task_type text not null, status text not null, payload jsonb not null default '{}', attempt_count integer not null default 0, max_attempts integer not null default 5, last_error text, created_at timestamptz not null default now(), updated_at timestamptz not null default now())""",
    """ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS attempt_count integer NOT NULL DEFAULT 0""",
    """ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS max_attempts integer NOT NULL DEFAULT 5""",
    """ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS last_error text""",
    """CREATE TABLE IF NOT EXISTS system_settings (key text primary key, value text not null, updated_at timestamptz not null default now())""",
    """CREATE INDEX IF NOT EXISTS documents_park_updated_idx ON documents(park_id, updated_at DESC)""",
    """CREATE INDEX IF NOT EXISTS documents_content_fts_idx ON documents USING gin (to_tsvector('simple', content))""",
    # ---------------------------------------------------------------- multi-agent audit
    """CREATE TABLE IF NOT EXISTS agent_steps (step_id bigserial primary key, run_id text not null, park_id text not null, agent_name text not null, step_index integer not null, event_type text not null, text_summary text, tool_name text, tool_arguments jsonb, tool_result_summary jsonb, status text not null default 'ok', latency_ms integer, created_at timestamptz not null default now())""",
    """CREATE INDEX IF NOT EXISTS agent_steps_run_idx ON agent_steps(run_id, step_index)""",
    """CREATE TABLE IF NOT EXISTS agent_delegations (delegation_id bigserial primary key, park_id text not null, parent_run_id text not null, child_run_id text, from_agent text not null, to_agent text not null, instruction text, round_index integer not null default 0, status text not null default 'pending', created_at timestamptz not null default now())""",
    """CREATE INDEX IF NOT EXISTS agent_delegations_parent_idx ON agent_delegations(parent_run_id, round_index)""",
    """CREATE TABLE IF NOT EXISTS run_traces (run_id text primary key, park_id text not null, conversation_id text, mode text not null default 'fallback', goal text, plan jsonb not null default '{}', agent_chain jsonb not null default '[]', final_answer text, citations jsonb not null default '[]', total_steps integer not null default 0, total_tool_calls integer not null default 0, total_latency_ms integer, degraded boolean not null default false, status text not null default 'success', error text, created_at timestamptz not null default now())""",
    """CREATE TABLE IF NOT EXISTS pending_feedback_intents (intent_id text primary key, park_id text not null, conversation_id text not null, draft jsonb not null, idempotency_key text unique, status text not null default 'awaiting_confirmation', created_at timestamptz not null default now(), expires_at timestamptz)""",
    """CREATE TABLE IF NOT EXISTS agent_registry (agent_name text not null, park_id text not null, role text not null, description text, enabled boolean not null default true, call_count bigint not null default 0, step_count bigint not null default 0, error_count bigint not null default 0, total_latency_ms bigint not null default 0, updated_at timestamptz not null default now(), primary key (agent_name, park_id))""",
    """CREATE TABLE IF NOT EXISTS realtime_status (park_id text not null, subject_type text not null, subject_id text not null, metric text not null, value jsonb not null, source text not null default 'internal', observed_at timestamptz not null, expires_at timestamptz, primary key (park_id, subject_type, subject_id, metric))""",
    # ---------------------------------------------------------------- agent_runs extension
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS parent_run_id text""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS trace_id text""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS step_index integer""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS agent_role text""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS tool_calls jsonb NOT NULL DEFAULT '[]'""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS goal text""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS instruction text""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS model text""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS mode text""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS iterations integer not null default 0""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS tool_call_count integer not null default 0""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS started_at timestamptz""",
    """ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS completed_at timestamptz""",
)


def _tcp_reachable() -> bool:
    """Fast pre-flight check so a down database costs milliseconds, not minutes."""
    host, port = _probe_host_port()
    try:
        with socket.create_connection((host, port), timeout=float(os.getenv("DB_TCP_TIMEOUT", "0.3"))):
            return True
    except OSError:
        return False


def probe_connection() -> bool:
    """Authoritative check: actually open a connection and run ``SELECT 1``.

    Expensive when the port is open but nothing answers the handshake, so it is used by
    the readiness endpoint and by tests - never on a per-call hot path.

    The URL is rebuilt against the **normalised probe host** first. ``localhost``
    resolves to both ``::1`` and ``127.0.0.1``, and psycopg waits out one connect
    timeout per resolved address, so probing the raw default URL cost 6 s for a 3 s
    ``connect_timeout`` - enough to make a test session look hung.
    """
    probe_engine = engine
    report_failure = True
    try:
        host, port = _probe_host_port()
        from sqlalchemy import create_engine
        from sqlalchemy.engine import make_url

        url = make_url(_database_url())
        if (url.host or "localhost") != host or (url.port or 5432) != port:
            probe_engine = create_engine(
                url.set(host=host, port=port).render_as_string(hide_password=False),
                pool_pre_ping=False,
                future=True,
            )
        else:
            # The shared engine already points at a single address; probing it must not
            # be allowed to open the circuit breaker for every other caller.
            report_failure = False
        with probe_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        note_database_success()
        return True
    except Exception:
        if report_failure:
            note_database_failure()
        return False
    finally:
        if probe_engine is not engine:
            probe_engine.dispose()


def database_available() -> bool:
    """Circuit-breaker availability check used by every read and write path.

    Four deliberate choices:

    1. it trusts the **TCP probe** only. A real ``SELECT 1`` here would cost a full
       connect timeout on every call whenever the port is open but the handshake is
       swallowed (a stopped-but-listening service, a proxy, a firewall drop) - measured
       at ~6 s per call, which turns graceful degradation into a multi-minute stall;
    2. once an outage is known, the cooldown is checked *before* any probe, so a read
       never re-opens a socket;
    3. the cooldown backs off exponentially, so a database that stays down is not
       re-probed on every request;
    4. a **successful** probe is only trusted for one retry window. Without that, a
       single stale success pinned ``_DB_AVAILABLE`` to true forever and every later
       call skipped the reachability check entirely - which is exactly how a
       "database is up" false positive turned into a per-step blocking connect.

    Use :func:`probe_connection` when a definite answer is required.
    """
    global _DB_AVAILABLE
    now = time.monotonic()
    if _DB_AVAILABLE is True and (now - _DB_PROBED_AT) < _DB_RETRY_SECONDS:
        return True
    if _DB_AVAILABLE is False and (now - _DB_FAILED_AT) < _cooldown_seconds():
        return False
    if not _tcp_reachable():
        note_database_failure()
        return False
    note_database_success()
    return True


def db_ready() -> bool:
    """Readiness for the health endpoint: cheap and honest about a down database."""
    if not _tcp_reachable():
        note_database_failure()
        return False
    return database_available()


def enabled() -> bool:
    """Guard for hot write paths.

    Reads already go through :func:`database_available`, but a bare ``engine.begin()``
    in the audit or conversation writers would still open a socket on every call. With
    the database down that costs one connect timeout per audit step - dozens per
    request - which turns graceful degradation into a multi-minute stall. Every writer
    must call this first and skip the write when it returns false.
    """
    return database_available()


def reset_database_state() -> None:
    """Test helper: forget a cached outage."""
    global _DB_AVAILABLE, _DB_FAILED_AT, _DB_FAILURES, _DB_PROBED_AT
    _DB_AVAILABLE = None
    _DB_FAILED_AT = 0.0
    _DB_FAILURES = 0
    _DB_PROBED_AT = 0.0


def ensure_runtime_tables() -> None:
    """Create every runtime table and the default admin account when absent."""
    with engine.begin() as conn:
        for statement in RUNTIME_TABLES:
            conn.execute(text(statement))
        exists = conn.execute(
            text("SELECT 1 FROM admin_users WHERE email=:email"), {"email": ADMIN_EMAIL}
        ).scalar()
        if not exists:
            hashed = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                text("INSERT INTO admin_users(email,password_hash) VALUES (:email,:password_hash)"),
                {"email": ADMIN_EMAIL, "password_hash": hashed},
            )


def payload_rows(table: str, limit: int = 100, offset: int = 0, park_id: str | None = None) -> list[dict[str, Any]]:
    """Read ``payload`` jsonb rows for one park.

    The park filter is mandatory across the codebase, so a missing/unreachable
    database degrades to an empty list rather than leaking every park's data.
    """
    from .config import PARK_ID as _PARK_ID

    park = park_id or _PARK_ID
    if not database_available():
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT payload FROM {table} WHERE park_id=:park_id ORDER BY 1 LIMIT :limit OFFSET :offset"),
                {"park_id": park, "limit": limit, "offset": offset},
            )
            return [dict(row[0]) for row in rows]
    except Exception:
        return []


def execute(statement: str, params: dict[str, Any] | None = None) -> list[Any]:
    """Run one statement, returning fetched rows when the statement yields any."""
    if not database_available():
        return []
    with engine.connect() as conn:
        result = conn.execute(text(statement), params or {})
        return result.fetchall() if result.returns_rows else []


def execute_write(statement: str, params: dict[str, Any] | None = None) -> None:
    if not database_available():
        return
    with engine.begin() as conn:
        conn.execute(text(statement), params or {})


def execute_many(statements: Iterable[tuple[str, dict[str, Any]]]) -> None:
    with engine.begin() as conn:
        for statement, params in statements:
            conn.execute(text(statement), params)
