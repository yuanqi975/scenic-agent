"""The RAG service must never run a synchronous retriever on the event loop.

``embed_query`` performs a real HTTP call (12 s timeout) and ``pymilvus`` blocks on the
network. Both retrievers are synchronous, so ``HybridRagService.search`` has to offload
them. Calling them directly from the coroutine stalls every concurrent agent in the same
request, which is invisible to a single-request test and fatal under load.
"""

from __future__ import annotations

import asyncio
import time

from tests.conftest import async_test

from app.services.rag import HybridRagService, RagSearchRequest

BLOCK_SECONDS = 0.4


class _BlockingRetriever:
    """Stands in for the Milvus / embedding round-trip: synchronous and slow."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def search(self, request):  # noqa: ANN001, ANN201
        self.calls += 1
        time.sleep(BLOCK_SECONDS)
        return self.payload


@async_test
async def test_concurrent_searches_are_not_serialised_by_a_blocking_retriever():
    """Two searches must overlap: total wall time near one block, not two."""
    primary = _BlockingRetriever({"dense": [{"document_id": "d1", "content": "x"}]})
    fallback = _BlockingRetriever([{"document_id": "d1", "content": "x"}])
    service = HybridRagService(primary=primary, fallback=fallback, backend="milvus")

    started = time.perf_counter()
    await asyncio.gather(
        service.search(RagSearchRequest(query="a")),
        service.search(RagSearchRequest(query="b")),
    )
    elapsed = time.perf_counter() - started

    # Serialised execution would need at least 2 x BLOCK_SECONDS.
    assert elapsed < BLOCK_SECONDS * 1.8, f"searches were serialised: {elapsed:.2f}s"
    assert primary.calls == 2


@async_test
async def test_the_event_loop_keeps_running_during_a_blocking_search():
    """A heartbeat task must keep ticking while the retriever is blocked."""
    primary = _BlockingRetriever({"dense": [{"document_id": "d1", "content": "x"}]})
    fallback = _BlockingRetriever([{"document_id": "d1", "content": "x"}])
    service = HybridRagService(primary=primary, fallback=fallback, backend="milvus")

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        await service.search(RagSearchRequest(query="a"))
    finally:
        beat.cancel()

    assert ticks >= 5, f"the event loop was blocked (only {ticks} ticks)"


@async_test
async def test_postgres_backend_also_offloads_and_reports_conflicts():
    """The default backend gets the same offload plus real conflict detection."""
    fallback = _BlockingRetriever(
        [
            {"document_id": "d1", "source_id": "park", "content": "旺季门票 190 元。", "authority": "official"},
            {"document_id": "d2", "source_id": "park", "content": "旺季门票 220 元。", "authority": "official"},
        ]
    )
    service = HybridRagService(primary=_BlockingRetriever({}), fallback=fallback, backend="postgres")

    result = await service.search(RagSearchRequest(query="门票"))

    assert result["backend"] == "postgres"
    assert len(result["items"]) == 2
    conflicts = result["conflicts"]
    assert [item["severity"] for item in conflicts] == ["conflict"]
    assert conflicts[0]["values"] == ["190", "220"]
    # Disagreement must be reflected in confidence, not hidden behind a citation.
    assert result["confidence"] < 0.35 + 0.12 * 2
