"""Resume-safe OpenAI-compatible embedding writer for PostgreSQL pgvector."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

def build_embeddings(input_path: Path, dsn: str, batch_size: int = 32, max_retries: int = 3):
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise SystemExit("EMBEDDING_API_KEY is required; no vectors were generated.")
    import httpx
    from sqlalchemy import create_engine, text
    endpoint = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1") + "/embeddings"
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "2560"))
    engine = create_engine(dsn, future=True)
    documents = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    document_ids = {row["document_id"] for row in documents}
    with engine.begin() as conn, httpx.Client(timeout=60) as client:
        pending_rows = conn.execute(
            text(
                "SELECT park_id,document_id,chunk_index,content FROM document_chunks "
                "WHERE embedding IS NULL AND park_id=:park_id ORDER BY document_id,chunk_index"
            ),
            {"park_id": documents[0]["park_id"] if documents else ""},
        ).mappings().all()
        pending = [dict(row) for row in pending_rows if row["document_id"] in document_ids]
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            for attempt in range(max_retries):
                try:
                    response = client.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json={"model": model, "input": [r["content"] for r in batch], "dimensions": dimension})
                    response.raise_for_status(); vectors = [x["embedding"] for x in response.json()["data"]]
                    if any(len(vector) != dimension for vector in vectors):
                        raise ValueError(f"embedding dimension must be {dimension} for document_chunks.embedding")
                    break
                except (httpx.HTTPError, KeyError):
                    if attempt == max_retries - 1: raise
                    time.sleep(2 ** attempt)
            for record, vector in zip(batch, vectors):
                conn.execute(
                    text(
                        "UPDATE document_chunks SET embedding=CAST(:vector AS halfvec) "
                        "WHERE park_id=:park_id AND document_id=:document_id AND chunk_index=:chunk_index"
                    ),
                    {
                        "vector": str(vector),
                        "park_id": record["park_id"],
                        "document_id": record["document_id"],
                        "chunk_index": record["chunk_index"],
                    },
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--dsn", required=True); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args(); build_embeddings(args.input, args.dsn, args.batch_size, args.max_retries); print("Embedding completed.")
