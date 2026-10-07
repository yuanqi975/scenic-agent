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
