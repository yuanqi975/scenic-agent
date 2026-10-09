"""Shared pytest configuration and tiny async-test support.

The suite is split in two on purpose:

* the multi-agent tests run with **no** PostgreSQL and no model, using injected
  in-memory repositories and a scripted LLM client;
* ``backend/tests/test_api_smoke.py`` genuinely needs a live PostgreSQL (catalog reads,
  feedback persistence, the embedding task row). Rather than failing with a confusing
  connection error, it is skipped with an explicit reason.

``async_test`` exists so the project does not need ``pytest-asyncio`` as a new
dependency just to exercise the agent runtime.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def async_test(func):
    """Run an ``async def`` test through ``asyncio.run`` without a plugin."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    if not inspect.iscoroutinefunction(func):
        raise TypeError("@async_test 只能用于 async def 测试函数")
    return wrapper


def _database_available() -> bool:
    """One authoritative probe for the whole session (never on a hot path).

    Gated behind a TCP reachability check first. Without it, a machine whose 5432 port
    is closed but whose loopback is slow to refuse pays a full connect timeout per
    probe - and this probe backs an autouse fixture, so it runs once per test.
    """
    try:
        from app.core import db

        if not db._tcp_reachable():
            return False
        return bool(db.probe_connection())
    except Exception:
        return False


#: Resolved once per session: the multi-agent unit tests never need PostgreSQL, so they
#: must not pay a connection attempt each time a repository is touched.
_DB_UP: bool | None = None


def database_is_up() -> bool:
    global _DB_UP
    if _DB_UP is None:
        _DB_UP = _database_available()
    return _DB_UP


@pytest.fixture(autouse=True)
def _degrade_without_database(monkeypatch):
    """Make the degraded path deterministic: no database, no probing, no waiting."""
    if database_is_up():
        from app.core import db

        # Failure-injection tests mutate these circuit-breaker globals. Restore
        # readiness per test so subsequent live API checks see the real database.
        monkeypatch.setattr(db, "_DB_AVAILABLE", True)
        monkeypatch.setattr(db, "_DB_FAILURES", 0)
        monkeypatch.setattr(db, "_DB_FAILED_AT", 0.0)
        monkeypatch.setattr(db, "_DB_PROBED_AT", time.monotonic())
        yield
        return
    from app.core import db

    monkeypatch.setattr(db, "database_available", lambda: False, raising=False)
    monkeypatch.setattr(db, "enabled", lambda: False, raising=False)
    monkeypatch.setattr(db, "db_ready", lambda: False, raising=False)
    yield


@pytest.fixture(autouse=True)
def _isolate_public_cache():
    from app.core import cache

    for key in list(cache._memory_cache):
        if key.startswith("public:"):
            cache._memory_cache.pop(key, None)
    yield


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    """Skip database-dependent modules when PostgreSQL is unreachable."""
    if database_is_up():
        return
    reason = "需要本地 PostgreSQL（安装并启动后重跑；多 Agent 测试不需要数据库）"
    for item in items:
        path = str(getattr(item, "fspath", ""))
        if path.endswith("test_api_smoke.py"):
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def fake_catalog(monkeypatch):
    """Install the in-memory Jiuzhaigou catalog for one test."""
    from tests.fakes import install_fake_database

    install_fake_database(monkeypatch)
    return True


@pytest.fixture
def emitted():
    """Collect every event an agent run emits, without needing SSE plumbing."""
    events: list[tuple[str, dict]] = []

    def emitter(event: str, data: dict) -> None:
        events.append((event, data))

    emitter.events = events  # type: ignore[attr-defined]
    emitter.names = lambda: [name for name, _ in events]  # type: ignore[attr-defined]
    return emitter
