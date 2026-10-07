"""Non-blocking persistence for audit and conversation writes.

**Why this module exists.** The audit writers (``agent_runs``, ``agent_steps``,
``agent_delegations``, ``run_traces``) and the conversation writers used to open a
SQLAlchemy connection *inline*, directly on the asyncio event loop. psycopg is a
synchronous driver, so a connect or a statement that has to wait lands in
``selectors._select`` and freezes **every** concurrent agent task for the whole
connect timeout - measured at 6 s per attempt, and 30 s+ for one ReAct loop once the
pool retried. A single audit step could therefore blow a 45 s request budget on its
own, and the visitor's answer arrived degraded for no visitor-facing reason.

The contract here is deliberately narrow:

* callers hand over a **plain callable** and never wait for it (``enqueue``);
* one daemon thread owns the database connection, so the event loop never blocks;
* a bounded queue means a slow or unreachable database sheds audit detail instead of
  growing memory or stalling the request;
* audit is *observability*, so losing a row is acceptable and losing the answer is not.

Tests can install an in-process sink with :func:`register_sink`, which is how the
suite asserts on audit rows without a database.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Audit detail is allowed to degrade under load; 4096 pending writes is far beyond
#: anything a single request can produce, so the cap only ever triggers during an
#: outage.
DEFAULT_QUEUE_SIZE = 4096

#: How long ``drain`` waits for the writer to catch up before giving up.
DEFAULT_DRAIN_TIMEOUT_SECONDS = 5.0

_queues: dict[str, "queue.Queue[Callable[[], Any] | None]"] = {}
_threads: dict[str, threading.Thread] = {}
_lock = threading.Lock()

#: In-process destinations installed by tests (or by a future in-memory backend).
_sinks: dict[str, list[Callable[[Callable[[], Any]], None]]] = {}

_stats: dict[str, dict[str, int]] = {}

#: Consecutive real connection failures observed by the writer thread.
_consecutive_failures = 0
#: When the writer last saw the database fail, and how long it should stay quiet.
_last_failure_at: float = 0.0

_SHUTDOWN = object()


def _bump(name: str, key: str, amount: int = 1) -> None:
    """Update counters defensively when tests or a reload replace the stats map."""
    bucket = _stats.setdefault(name, {"queued": 0, "written": 0, "failed": 0, "dropped": 0})
    bucket[key] = bucket.get(key, 0) + amount


def _cooldown_seconds() -> float:
    """Exponential backoff from the first genuine connection failure.

    ``_tcp_reachable`` only proves that *something* accepted a TCP connection on the
    database port. A firewall that accepts the connection and then drops the PostgreSQL
    handshake passes that probe, so every queued write paid a full ``connect_timeout``
    - measured at 6 s per attempt, enough to blow a 45 s request budget on its own. Once
    the writer has actually failed to connect, this short-circuits the queue instead of
    retrying per job.
    """
    if _consecutive_failures == 0:
        return 0.0
    return min(_retry_seconds() * (2 ** (_consecutive_failures - 1)), _retry_max_seconds())


def _retry_seconds() -> float:
    return float(os.getenv("DB_WRITE_RETRY_SECONDS", "60"))


def _retry_max_seconds() -> float:
    return float(os.getenv("DB_WRITE_RETRY_MAX_SECONDS", "600"))


def _breaker_open() -> bool:
    return _consecutive_failures > 0 and (time.monotonic() - _last_failure_at) < _cooldown_seconds()


def note_write_failure() -> None:
    """Record one real connection failure so pending and future writes are shed."""
    global _consecutive_failures, _last_failure_at
    _consecutive_failures += 1
    _last_failure_at = time.monotonic()
    try:
        from .db import note_database_failure

        note_database_failure()
    except Exception:  # pragma: no cover - defensive
        pass


def note_write_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


def breaker_state() -> dict[str, Any]:
    """Observability for the health endpoint and for tests."""
    return {
        "open": _breaker_open(),
        "consecutive_failures": _consecutive_failures,
        "cooldown_seconds": round(_cooldown_seconds(), 1),
    }


def _queue_for(name: str) -> "queue.Queue[Callable[[], Any] | None]":
    with _lock:
        existing = _queues.get(name)
        if existing is not None:
            return existing
        created: "queue.Queue[Callable[[], Any] | None]" = queue.Queue(maxsize=DEFAULT_QUEUE_SIZE)
        _queues[name] = created
        _stats[name] = {"queued": 0, "written": 0, "failed": 0, "dropped": 0}
        return created


def _worker(name: str) -> None:
    """Drain one queue forever. Runs on a daemon thread, never on the event loop."""
    pending = _queue_for(name)
    while True:
        job = pending.get()
        try:
            if job is None:  # shutdown sentinel
                return
            _run_one(name, job)
        except Exception:  # a broken job must not kill the writer
            _bump(name, "failed")
            logger.debug("background write failed", exc_info=True)
        finally:
            pending.task_done()


def _run_one(name: str, job: Callable[[], Any]) -> None:
    if _breaker_open():
        _bump(name, "dropped")
        return
    sinks = _sinks.get(name)
    if sinks:
        for sink in list(sinks):
            sink(job)
        _bump(name, "written")
        note_write_success()
        return
    job()
    _bump(name, "written")
    note_write_success()


def _drain_inline(name: str) -> None:
    """Run every pending job on the calling thread, in order.

    Only used when a sink is installed: a sink *is* the destination, so the jobs must
    be executed on the caller's thread to be observable without sleeping. Doing it this
    way also removes the race where a queue is handed a job before its sink (or its
    worker) is wired up.
    """
    pending = _queues.get(name)
    if pending is None:
        return
    while True:
        try:
            job = pending.get_nowait()
        except queue.Empty:
            return
        try:
            if job is not None:
                _run_one(name, job)
        except Exception:
            _bump(name, "failed")
            logger.debug("sink write failed", exc_info=True)
        finally:
            pending.task_done()


def _ensure_worker(name: str) -> "queue.Queue[Callable[[], Any] | None]":
    # ``_queue_for`` takes ``_lock`` itself, so it must be called *outside* the critical
    # section below; nesting the two deadlocks the calling thread instantly. The queue is
    # returned so callers do not re-look-it-up.
    pending = _queue_for(name)
    with _lock:
        thread = _threads.get(name)
        if thread is not None and thread.is_alive():
            return pending
        created = threading.Thread(target=_worker, args=(name,), name=f"dsh-write-{name}", daemon=True)
        _threads[name] = created
        created.start()
    return pending


def enqueue(name: str, job: Callable[[], Any], *, key: str | None = None) -> bool:
    """Hand one write to a background thread. Returns ``False`` when it was shed.

    ``key`` labels the write in the stats and in the warning log; it is deliberately
    *not* used for de-duplication, because audit rows are append-only and their order
    matters more than their uniqueness.
    """
    if _breaker_open():
        _stats.setdefault(name, {"queued": 0, "written": 0, "failed": 0, "dropped": 0})
        _bump(name, "dropped")
        return False
    pending = _ensure_worker(name)
    try:
        pending.put_nowait(job)
        _bump(name, "queued")
        return True
    except queue.Full:
        _bump(name, "dropped")
        logger.warning("write queue %r is full; shedding one %s write", name, key or "background")
        return False


def register_sink(name: str, sink: Callable[[Callable[[], Any]], None]) -> None:
    """Route one queue's jobs into an in-process sink instead of the database.

    The sink receives the *job callable*, so a test can decide whether to execute it.
    Jobs already sitting in the queue are flushed through the new sink immediately, so
    a caller never has to care whether the writer thread was already running.
    """
    _queue_for(name)
    _sinks.setdefault(name, []).append(sink)
    _drain_inline(name)


def unregister_sink(name: str, sink: Callable[[Callable[[], Any]], None] | None = None) -> None:
    if sink is None:
        _sinks.pop(name, None)
        return
    remaining = [item for item in _sinks.get(name, []) if item is not sink]
    if remaining:
        _sinks[name] = remaining
    else:
        _sinks.pop(name, None)


def drain(name: str | None = None, *, timeout: float = DEFAULT_DRAIN_TIMEOUT_SECONDS) -> bool:
    """Block until pending writes are flushed. For tests and graceful shutdown only."""
    names = [name] if name else list(_queues)
    drained = True
    for item in names:
        if _sinks.get(item):
            # A sink is the destination, so there is nothing to wait for: run the
            # pending jobs here, in order, on the calling thread.
            _drain_inline(item)
            continue
        pending = _queues.get(item)
        if pending is None:
            continue
        deadline = time.monotonic() + timeout
        while pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        if pending.unfinished_tasks:
            drained = False
    return drained


def stats(name: str | None = None) -> dict[str, Any]:
    """Observability for the admin health endpoint and for tests."""
    if name is not None:
        return dict(_stats.get(name, {}))
    return {key: dict(value) for key, value in _stats.items()}


def reset(*, stop_workers: bool = True) -> None:
    """Test helper: forget queues, sinks, counters and the circuit breaker.

    The writer threads are daemon threads that hold no request state, so a test that
    resets everything must also stop them; otherwise a stale thread keeps draining a
    queue nobody reads.
    """
    global _consecutive_failures, _last_failure_at
    with _lock:
        queues = dict(_queues)
        threads = dict(_threads)
        _queues.clear()
        _threads.clear()
        _sinks.clear()
        _stats.clear()
    _consecutive_failures = 0
    _last_failure_at = 0.0
    for pending in queues.values():
        try:
            pending.put_nowait(None)
        except queue.Full:
            pass
    if stop_workers:
        for thread in threads.values():
            thread.join(timeout=1.0)


def _shutdown() -> None:  # pragma: no cover - interpreter teardown
    try:
        drain(timeout=0.5)
    except Exception:
        pass


atexit.register(_shutdown)

#: Queue names used across the project.
AUDIT_QUEUE = "audit"
CONVERSATION_QUEUE = "conversation"

__all__ = [
    "AUDIT_QUEUE",
    "CONVERSATION_QUEUE",
    "DEFAULT_QUEUE_SIZE",
    "breaker_state",
    "drain",
    "enqueue",
    "note_write_failure",
    "note_write_success",
    "register_sink",
    "reset",
    "stats",
    "unregister_sink",
]
