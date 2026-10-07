from __future__ import annotations

import json


def test_milvus_schema_contains_dense_sparse_and_bm25_function():
    from scripts.index_milvus import build_schema

    schema = build_schema(8)
    fields = {field.name: field.to_dict() for field in schema.fields}
    assert fields["vector"]["type"].name == "FLOAT_VECTOR"
    assert fields["sparse_vector"]["type"].name == "SPARSE_FLOAT_VECTOR"
    assert fields["content"]["type"].name == "VARCHAR"
    assert fields["content"]["params"]["enable_analyzer"] is True
    analyzer = json.loads(fields["content"]["params"]["analyzer_params"])
    assert analyzer["tokenizer"] == "jieba"
    assert any(function.name == "bm25_function" for function in schema.functions)


def test_milvus_retriever_returns_dense_and_native_bm25(monkeypatch):
    from app.services import retrieval
    from app.services.rag import MilvusRetriever, RagSearchRequest

    class FakeClient:
        def search(self, *, anns_field, **kwargs):
            suffix = "dense" if anns_field == "vector" else "sparse"
            return [[{
                "id": f"chunk-{suffix}",
                "distance": 0.2 if suffix == "dense" else 4.5,
                "entity": {
                    "chunk_id": f"chunk-{suffix}",
                    "document_id": f"doc-{suffix}",
                    "content": f"jiuzhaigou {suffix} evidence",
                    "source_type": "park",
                    "source_id": "jiuzhaigou_scenic_area",
                    "authority": "official",
                },
            }]]

    monkeypatch.setattr(retrieval, "embed_query", lambda query: "[0.1, 0.2]")
    retriever = MilvusRetriever()
    retriever._client = FakeClient()
    result = retriever.search(RagSearchRequest(query="ticket and opening", top_k=2))
    assert result["dense"][0]["retrieval"] == "dense"
    assert result["sparse"][0]["retrieval"] == "sparse"
    assert result["sparse"][0]["score"] == 4.5


def test_milvus_dense_survives_sparse_failure(monkeypatch):
    from app.services import retrieval
    from app.services.rag import MilvusRetriever, RagSearchRequest

    class PartialClient:
        def search(self, *, anns_field, **kwargs):
            if anns_field == "sparse_vector":
                raise RuntimeError("BM25 index unavailable")
            return [[{"id": "chunk-1", "distance": 0.1, "entity": {
                "document_id": "doc-1", "content": "dense evidence",
            }}]]

    monkeypatch.setattr(retrieval, "embed_query", lambda query: "[0.1, 0.2]")
    retriever = MilvusRetriever()
    retriever._client = PartialClient()
    result = retriever.search(RagSearchRequest(query="park introduction"))
    assert result["dense"]
    assert result["sparse"] == []


def test_milvus_bm25_survives_dense_failure(monkeypatch):
    from app.services import retrieval
    from app.services.rag import MilvusRetriever, RagSearchRequest

    class PartialClient:
        def search(self, *, anns_field, **kwargs):
            if anns_field == "vector":
                raise RuntimeError("dense index unavailable")
            return [[{"id": "chunk-1", "distance": 3.1, "entity": {
                "document_id": "doc-1", "content": "BM25 evidence",
            }}]]

    monkeypatch.setattr(retrieval, "embed_query", lambda query: "[0.1, 0.2]")
    retriever = MilvusRetriever()
    retriever._client = PartialClient()
    result = retriever.search(RagSearchRequest(query="opening hours"))
    assert result["dense"] == []
    assert result["sparse"]
