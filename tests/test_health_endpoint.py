"""Health endpoint contract.

The health payload is the operator's only view of *why* the system is degraded, so it
must keep reporting the two things that used to be invisible: the database circuit
breaker and the background write queue. Without the latter, "database: true" would read
as "everything is being persisted" even while audit rows are being shed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_write_queue_and_breaker():
    from app.core import db_write
    from app.main import app

    db_write.reset()
    client = TestClient(app)
    payload = client.get("/api/v1/health").json()

    assert "database" in payload
    assert "status" in payload
    write_queue = payload["write_queue"]
    assert set(write_queue) == {"breaker", "queues"}
    assert "open" in write_queue["breaker"]
    assert "consecutive_failures" in write_queue["breaker"]


def test_health_reflects_an_open_breaker(monkeypatch):
    """A shedding writer must be visible in the payload, not just in the logs."""
    from app.core import db_write
    from app.main import app

    db_write.reset()
    db_write.note_write_failure()
    try:
        client = TestClient(app)
        breaker = client.get("/api/v1/health").json()["write_queue"]["breaker"]
        assert breaker["open"] is True
        assert breaker["consecutive_failures"] >= 1
    finally:
        db_write.reset()


def test_metrics_endpoint_exposes_request_counter():
    from app.main import app

    client = TestClient(app)
    client.get("/api/v1/health")
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
