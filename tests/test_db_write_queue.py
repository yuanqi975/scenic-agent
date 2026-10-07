"""Regression tests for the two failure modes that made the suite unusable.

Both were the same class of bug - a synchronous PostgreSQL call on the asyncio event
loop - and both were invisible from the visitor's perspective except as latency:

1. ``audit.save_step`` opened a connection inline. A closed-but-slow database port made
   one ReAct loop take 30 s and blew the 45 s request budget, so ``answer_async``
   returned ``budget_exceeded`` for requests that had done nothing wrong.
2. ``database_available`` treated the **port being open** as proof that PostgreSQL was
   reachable, and cached a single success forever. A firewall that accepts the TCP
   connection and then drops the handshake therefore passed every check while every
   write paid a full connect timeout.

These tests must never need PostgreSQL, so they drive the queue through an in-process
sink and the breaker through an injected failure.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.core import db_write


@pytest.fixture(autouse=True)
def clean_writer():
    """Every test gets a fresh queue, sink set and circuit breaker."""
    db_write.reset()
    yield
    db_write.reset()


def test_enqueue_returns_immediately_for_a_slow_job():
    """The whole point: a 0.5 s write must not cost the caller 0.5 s."""
    release = threading.Event()

    def slow_job() -> None:
        release.wait(timeout=5)

    started = time.perf_counter()
    assert db_write.enqueue("audit", slow_job, key="slow") is True
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"enqueue blocked for {elapsed:.3f}s"
    release.set()
    assert db_write.drain("audit", timeout=5) is True


def test_queued_jobs_actually_run_and_drain_waits_for_them():
    done: list[int] = []
    for index in range(20):
        db_write.enqueue("audit", lambda index=index: done.append(index))

    assert db_write.drain("audit", timeout=5) is True
    assert sorted(done) == list(range(20))
    assert db_write.stats("audit")["written"] == 20


def test_a_failing_job_is_counted_and_does_not_kill_the_writer():
    """One broken write must not stop the writer thread from serving the rest."""
    db_write.enqueue("audit", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    seen: list[str] = []
    db_write.enqueue("audit", lambda: seen.append("after"))

    assert db_write.drain("audit", timeout=5) is True
    assert seen == ["after"]
    assert db_write.stats("audit")["failed"] == 1


def test_circuit_breaker_sheds_writes_after_a_real_failure():
    """A database that just refused a connection is not retried per queued job."""
    db_write.note_write_failure()
    stored: list[str] = []
    db_write.register_sink("audit", lambda job: stored.append("ran"))

    assert db_write.breaker_state()["open"] is True
    assert db_write.enqueue("audit", lambda: stored.append("job")) is False
    assert stored == []
    assert db_write.stats("audit")["dropped"] == 1

    # A successful write closes the breaker again.
    db_write.note_write_success()
    assert db_write.breaker_state()["open"] is False
    assert db_write.enqueue("audit", lambda: stored.append("job")) is True


def test_sink_receives_the_job_instead_of_the_database():
    """This is what lets the rest of the suite assert on audit writes with no database."""
    jobs: list[str] = []
    db_write.register_sink(db_write.AUDIT_QUEUE, lambda job: jobs.append(job))

    db_write.enqueue(db_write.AUDIT_QUEUE, lambda: None, key="agent_step")
    assert db_write.drain(db_write.AUDIT_QUEUE, timeout=5) is True
    assert len(jobs) == 1
    assert callable(jobs[0])

    db_write.unregister_sink(db_write.AUDIT_QUEUE)
    db_write.enqueue(db_write.AUDIT_QUEUE, lambda: None)
    assert db_write.drain(db_write.AUDIT_QUEUE, timeout=5) is True
    assert len(jobs) == 1, "an unregistered sink must stop receiving jobs"


def test_audit_writers_never_touch_the_engine_inline(monkeypatch):
    """``start_run``/``save_step``/``finish_run`` must be pure queue submissions."""
    from app.agents.schemas import AgentResult
    from app.services import audit

    monkeypatch.setattr(audit, "enabled", lambda: True)
    opened: list[str] = []
    monkeypatch.setattr(audit, "engine", _ExplodingEngine(opened))

    captured: list[str] = []
    db_write.register_sink(db_write.AUDIT_QUEUE, lambda job: captured.append("job"))

    run_id = audit.start_run(
        agent_name="knowledge_agent",
        role="worker",
        conversation_id="c1",
        instruction="查票价",
    )
    audit.save_step({"run_id": run_id, "agent_name": "knowledge_agent", "event_type": "thought"})
    audit.finish_run(
        AgentResult(agent="knowledge_agent", run_id=run_id, instruction="查票价"),
        role="worker",
    )

    assert db_write.drain(db_write.AUDIT_QUEUE, timeout=5) is True
    assert opened == [], "audit must not open a connection on the calling thread"
    assert len(captured) == 3


class _ExplodingEngine:
    """Any inline use of the engine is a bug, so fail loudly and record it."""

    def __init__(self, opened: list[str]) -> None:
        self._opened = opened

    def begin(self):  # noqa: ANN201
        self._opened.append("begin")
        raise AssertionError("audit opened a connection on the calling thread")

    def connect(self):  # noqa: ANN201
        self._opened.append("connect")
        raise AssertionError("audit opened a connection on the calling thread")
