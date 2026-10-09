from tests.conftest import async_test


def test_rrf_fusion_prefers_documents_seen_by_multiple_channels():
    from app.services.rag import RagHit, fuse_hits

    hits = fuse_hits(
        {
            "dense": [RagHit("same", "a", 0.2, "dense"), RagHit("dense", "b", 0.1, "dense")],
            "sparse": [RagHit("same", "a", 0.1, "sparse"), RagHit("sparse", "c", 0.1, "sparse")],
            "entity": [],
            "structured": [],
        },
        top_k=2,
    )
    assert hits[0].chunk_id == "same"
    assert set(hits[0].channels) == {"dense", "sparse"}


@async_test
async def test_hybrid_rag_falls_back_when_milvus_is_unavailable():
    from app.services.rag import HybridRagService, RagSearchRequest

    class Fallback:
        def search(self, request):
            return [{"document_id": "doc-1", "content": "官方内容", "authority": "official"}]

    class Broken:
        def search(self, request):
            raise RuntimeError("milvus unavailable")

    service = HybridRagService(primary=Broken(), fallback=Fallback(), backend="milvus")
    result = await service.search(RagSearchRequest(query="门票"))
    assert result["items"][0]["document_id"] == "doc-1"
    assert result["backend"] == "postgres-fallback"


@async_test
async def test_rerank_preserves_the_primary_exact_entity_document():
    """A generic FAQ must not outrank a document for the POI named by the visitor."""
    from app.services.rag import HybridRagService

    service = HybridRagService(primary=object(), fallback=object(), backend="milvus")
    result = {
        "items": [
            {
                "document_id": "faq-1",
                "content": "五花海 开放时间 游玩建议 五花海",
                "entity_rank": None,
            },
            {
                "document_id": "attraction-1",
                "content": "五花海位于日则沟。",
                "entity_rank": 1,
            },
        ]
    }
    ranked = await service._rerank_result(result, "五花海开放时间和游玩建议", final_top_k=1)
    assert ranked["items"][0]["document_id"] == "attraction-1"


@async_test
async def test_rerank_returns_requested_top_k_after_a_wider_candidate_pool():
    from app.services.rag import HybridRagService

    service = HybridRagService(primary=object(), fallback=object(), backend="milvus")
    result = {"items": [{"document_id": str(index), "content": f"资料 {index}"} for index in range(12)]}
    ranked = await service._rerank_result(result, "资料", final_top_k=5)
    assert len(ranked["items"]) == 5
    assert ranked["rerank"]["count"] == 12


def test_embed_queries_preserves_provider_index_order(monkeypatch):
    from app.services import retrieval

    monkeypatch.setattr(retrieval, "EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setattr(retrieval, "EMBEDDING_API_KEY", "test-key")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}

    class Client:
        def post(self, url, **kwargs):
            assert kwargs["json"]["input"] == ["first", "second"]
            return Response()

    assert retrieval.embed_queries(["first", "second"], client=Client()) == ["[1.0]", "[2.0]"]
