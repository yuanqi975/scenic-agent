"""Destructively rebuild the Milvus Dense + native BM25 index.

The previous collection was Dense-only. This command intentionally drops the target
collection before creating the new schema. PostgreSQL remains the source of truth and
is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.chunking import split_document  # noqa: E402


def build_schema(dimension: int):
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba"},
    )
    schema.add_field("document_id", DataType.VARCHAR, max_length=256)
    schema.add_field("source_type", DataType.VARCHAR, max_length=64)
    schema.add_field("source_id", DataType.VARCHAR, max_length=256)
    schema.add_field("authority", DataType.VARCHAR, max_length=32)
    schema.add_field("park_id", DataType.VARCHAR, max_length=128)
    schema.add_field("metadata", DataType.JSON)
    schema.add_function(
        Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        )
    )
    return schema


def create_indexes(client: Any, collection: str) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_name="dense_index", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_bm25_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    client.create_index(collection_name=collection, index_params=index_params)


def _embedding_rows(http: Any, batch: list[dict[str, Any]], base_url: str, model: str, dimension: int) -> list[list[float]]:
    response = http.post(
        f"{base_url}/embeddings",
        headers={"Authorization": f"Bearer {os.getenv('EMBEDDING_API_KEY')}"},
        json={"model": model, "input": [item["content"] for item in batch], "dimensions": dimension},
    )
    response.raise_for_status()
    vectors = [item["embedding"] for item in response.json()["data"]]
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError(f"embedding dimension must be {dimension} for Milvus collection")
    return vectors


def index_documents(
    input_path: Path,
    uri: str,
    collection: str,
    batch_size: int = 32,
    *,
    recreate: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    from pymilvus import MilvusClient
    import httpx

    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "2560"))
    if not api_key:
        raise SystemExit("EMBEDDING_API_KEY is required; no Milvus vectors were generated.")

    client = MilvusClient(uri=uri, token=os.getenv("MILVUS_TOKEN") or None, timeout=float(os.getenv("MILVUS_TIMEOUT_SECONDS", "30")))
    exists = client.has_collection(collection)
    if exists and recreate:
        print(f"Dropping existing Milvus collection: {collection}")
        client.drop_collection(collection)
        exists = False
    if not exists:
        client.create_collection(collection_name=collection, schema=build_schema(dimension))
        create_indexes(client, collection)
    else:
        fields = {field.get("name") for field in client.describe_collection(collection).get("fields", [])}
        missing = {
            "chunk_id", "vector", "sparse_vector", "content", "document_id",
            "source_type", "source_id", "authority", "park_id", "metadata",
        } - fields
        if missing:
            raise RuntimeError(f"existing collection lacks hybrid fields {sorted(missing)}; rerun without --keep-existing")

    source_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for item in source_rows:
        if "chunk_index" in item:
            rows.append(item)
            continue
        for chunk in split_document(item["content"], target_size=chunk_size, overlap=chunk_overlap):
            rows.append({
                **item,
                "chunk_index": chunk.index,
                "content": chunk.content,
                "chunk_id": f"{item['document_id']}:{chunk.index}",
            })
    total = 0
    with httpx.Client(timeout=60) as http:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            vectors = _embedding_rows(http, batch, base_url, model, dimension)
            data = []
            for item, vector in zip(batch, vectors):
                metadata = item.get("metadata") or {}
                chunk_id = str(item.get("chunk_id") or f"{item['document_id']}:{item.get('chunk_index', 0)}")
                data.append({
                    "chunk_id": chunk_id,
                    "vector": vector,
                    "content": item["content"],
                    "document_id": item["document_id"],
                    "source_type": item["source_type"],
                    "source_id": item["source_id"],
                    "authority": metadata.get("authority", "official"),
                    "park_id": item["park_id"],
                    "metadata": metadata,
                })
            client.upsert(collection_name=collection, data=data)
            total += len(data)
    client.load_collection(collection_name=collection)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Destructively rebuild Milvus Dense + BM25 indexes")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "jiuzhaigou" / "rag_documents.jsonl")
    parser.add_argument("--uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "scenic_knowledge"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--keep-existing", action="store_true", help="do not drop an existing collection")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()
    count = index_documents(
        args.input,
        args.uri,
        args.collection,
        args.batch_size,
        recreate=not args.keep_existing,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Rebuilt Dense + BM25 Milvus collection {args.collection}: {count} chunks")
