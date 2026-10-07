"""Idempotently import the generated Jiuzhaigou dataset into PostgreSQL."""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.chunking import split_document  # noqa: E402

TABLES = {
    "attractions.jsonl": ("attractions", "attraction_id"), "facilities.jsonl": ("facilities", "facility_id"),
    "routes.jsonl": ("routes", "route_id"), "faqs.jsonl": ("faqs", "faq_id"), "notices.jsonl": ("notices", "notice_id"), "feedbacks.jsonl": ("feedbacks", "feedback_id"),
    "feedback_candidates.jsonl": ("feedback_candidates", "candidate_id"), "evaluation_questions.jsonl": ("evaluation_questions", "question_id"),
}

def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        yield from (json.loads(line) for line in f if line.strip())

def import_dataset(dsn: str, data_dir: Path, *, chunk_size: int = 800, chunk_overlap: int = 120):
    from sqlalchemy import create_engine, text
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy import MetaData, Table
    engine = create_engine(dsn, future=True)
    meta = MetaData()
    meta.reflect(bind=engine)
    with engine.begin() as conn:
        park = json.loads((data_dir / "parks.json").read_text(encoding="utf-8"))
        conn.execute(insert(meta.tables["parks"]).values(park_id=park["park_id"], payload=park, created_at=park["created_at"]).on_conflict_do_nothing())
        for filename, (table_name, key) in TABLES.items():
            table = meta.tables[table_name]
            values = [{"park_id": item["park_id"], key: item[key], "payload": item} for item in rows(data_dir / filename)]
            conn.execute(insert(table).values(values).on_conflict_do_nothing())
        docs = []
        chunks = []
        for item in rows(data_dir / "rag_documents.jsonl"):
            docs.append({k: item[k] for k in ("park_id", "document_id", "source_type", "source_id", "content", "metadata", "knowledge_version", "updated_at")})
            split = split_document(item["content"], target_size=chunk_size, overlap=chunk_overlap)
            for chunk in split:
                chunks.append({
                    "park_id": item["park_id"],
                    "document_id": item["document_id"],
                    "chunk_index": chunk.index,
                    "content": chunk.content,
                    "updated_at": item["updated_at"],
                })
        document_insert = insert(meta.tables["documents"]).values(docs)
        conn.execute(
            document_insert.on_conflict_do_update(
                index_elements=["park_id", "document_id"],
                set_={
                    "source_type": document_insert.excluded.source_type,
                    "source_id": document_insert.excluded.source_id,
                    "content": document_insert.excluded.content,
                    "metadata": document_insert.excluded.metadata,
                    "knowledge_version": document_insert.excluded.knowledge_version,
                    "updated_at": document_insert.excluded.updated_at,
                },
            )
        )
        # Re-importing a document must replace its chunk layout; otherwise an
        # earlier one-chunk import would leave stale rows and embeddings behind.
        for document in docs:
            conn.execute(
                text("DELETE FROM document_chunks WHERE park_id=:park_id AND document_id=:document_id"),
                {"park_id": document["park_id"], "document_id": document["document_id"]},
            )
        if chunks:
            conn.execute(insert(meta.tables["document_chunks"]).values(chunks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True, help="PostgreSQL SQLAlchemy DSN")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "jiuzhaigou")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args(); import_dataset(args.dsn, args.data_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap); print("Import completed.")
