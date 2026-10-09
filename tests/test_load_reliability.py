import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from fastapi.testclient import TestClient

from app.core import cache, config


def request(path, ip="127.0.0.1", headers=()):
    return Request({"type": "http", "method": "POST", "path": path, "headers": list(headers), "client": (ip, 1234), "scheme": "http", "server": ("test", 80), "query_string": b""})


@pytest.fixture
def isolated_cache(monkeypatch):
    monkeypatch.setattr(cache, "redis_client", lambda: None)
    cache._memory_rate_limits.clear()
    cache._memory_cache.clear()
    yield
    cache._memory_rate_limits.clear()
    cache._memory_cache.clear()


def test_scopes_do_not_exhaust_each_other_and_retry_header(monkeypatch, isolated_cache):
    monkeypatch.setattr(cache, "RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(config, "RATE_LIMIT_RECOMMEND_PER_MINUTE", 1)
    monkeypatch.setattr(config, "RATE_LIMIT_IP_PER_MINUTE", 100)
    cache.enforce_public_rate_limit(request("/api/v1/chat/messages"))
    cache.enforce_public_rate_limit(request("/api/v1/chat/messages"))
    cache.enforce_public_rate_limit(request("/api/v1/recommendations"))
    with pytest.raises(HTTPException) as error:
        cache.enforce_public_rate_limit(request("/api/v1/chat/messages"))
    assert error.value.status_code == 429
    assert 1 <= int(error.value.headers["Retry-After"]) <= 60
    cache.enforce_public_rate_limit(request("/api/v1/chat/messages", "127.0.0.2"))


def test_untrusted_identity_header_cannot_bypass_quota(monkeypatch, isolated_cache):
    monkeypatch.setattr(cache, "RATE_LIMIT_PER_MINUTE", 1)
    cache.enforce_public_rate_limit(request("/api/v1/chat/messages", headers=[(b"x-user-id", b"a")]))
    with pytest.raises(HTTPException) as error:
        cache.enforce_public_rate_limit(request("/api/v1/chat/messages", headers=[(b"x-user-id", b"b")]))
    assert error.value.status_code == 429


def test_sse_timeout_has_terminal_error_and_releases_capacity(monkeypatch, isolated_cache):
    from app.main import app
    from app.api import chat

    async def stalled(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(chat, "answer_async", stalled)
    monkeypatch.setattr(chat, "AGENT_REQUEST_TIMEOUT_SECONDS", -4.98)
    monkeypatch.setattr(chat, "_stream_slots", asyncio.Semaphore(1))
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post("/api/v1/chat/stream", json={"message": "x"})
            assert response.status_code == 200
            assert "event: citations" in response.text
            assert "event: result" in response.text
            assert '"status": "degraded"' in response.text
            assert "request_timeout" in response.text


def test_sync_chat_timeout_returns_bounded_degraded_result(monkeypatch, isolated_cache):
    from app.main import app
    from app.api import chat

    async def stalled(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(chat, "answer_async", stalled)
    monkeypatch.setattr(chat, "AGENT_REQUEST_TIMEOUT_SECONDS", 0.01)
    with TestClient(app) as client:
        response = client.post("/api/v1/chat/messages", json={"message": "x"})
    assert response.status_code == 200
    assert response.json()["error"] == "request_timeout"
    assert response.json()["degraded"] is True


def test_public_cache_keeps_conversation_and_trace_unique(fake_catalog, isolated_cache):
    from app.services.orchestrator import answer_async

    async def run():
        first = await answer_async("九寨沟门票多少钱", mode="fallback")
        second = await answer_async("九寨沟门票多少钱", mode="fallback")
        assert first["conversation_id"] != second["conversation_id"]
        assert first["trace_id"] != second["trace_id"]
        assert first["message"] == second["message"]
        assert second["plan"]["cache_hit"] is True
        assert second["cache_hit"] is True

    asyncio.run(run())


def test_provider_balance_failure_opens_shared_circuit(isolated_cache):
    import httpx
    from app.services.llm import LLMClient, LLMUnavailable

    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    async def run():
        first = LLMClient(base_url="https://test-provider", api_key="test", transport=httpx.MockTransport(handler))
        with pytest.raises(LLMUnavailable):
            await first.complete([{"role": "user", "content": "x"}])
        second = LLMClient(base_url="https://test-provider", api_key="test", transport=httpx.MockTransport(handler))
        with pytest.raises(LLMUnavailable, match="熔断"):
            await second.complete([{"role": "user", "content": "x"}])
        assert len(calls) == 1
        assert second.provider_status()["reason"] == "provider_402_balance"
        assert second.provider_status()["available"] is False
        await first._client().aclose()

    asyncio.run(run())


def test_llm_queue_wait_has_timeout_and_does_not_leak_slot():
    from app.services.llm import LLMClient, LLMUnavailable

    async def run():
        client = LLMClient(base_url="https://unused", api_key="test", timeout=.02, max_concurrency=1)
        await client._semaphore.acquire()
        with pytest.raises(LLMUnavailable, match="超时"):
            await client.complete([{"role": "user", "content": "x"}])
        assert client.call_count == 0
        client._semaphore.release()
        await asyncio.wait_for(client._semaphore.acquire(), timeout=.1)
        client._semaphore.release()

    asyncio.run(run())
