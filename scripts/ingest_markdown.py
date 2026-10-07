"""Ingest one Markdown/plain-text scenic document through the RAG pipeline.

The command writes the canonical document, replaces its chunk rows, creates a
durable embedding task, and (when Redis is available) enqueues that task for the
running worker. It is intended for real source documents, unlike the generated
JSONL fixture importer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.chunking import split_document  # noqa: E402


def ingest(
    *,
    dsn: str,
    input_path: Path,
    document_id: str,
    source_id: str,
    source_type: str,
    chunk_size: int,
    chunk_overlap: int,
    enqueue: bool,
    redis_url: str,
) -> dict[str, object]:
    from sqlalchemy import create_engine, text

    content = input_path.read_text(encoding="utf-8")
    chunks = split_document(content, target_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        raise ValueError("document is empty")
    now = datetime.now(timezone.utc)
    version = f"manual-{now.strftime('%Y%m%d%H%M%S')}-{document_id[-8:]}"
    metadata = {
        "source_type": source_type,
        "source_id": source_id,
        "authority": "official",
        "document_kind": "scenic_guide",
        "filename": input_path.name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "updated_at": now.isoformat(),
    }
    engine = create_engine(dsn, future=True)
    task_id = f"task_embed_{uuid.uuid4().hex[:16]}"
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO knowledge_versions(version,park_id,status) VALUES (:version,'jiuzhaigou_scenic_area','published') "
                "ON CONFLICT (version) DO NOTHING"
            ),
            {"version": version},
        )
        conn.execute(
            text(
                """INSERT INTO documents(park_id,document_id,source_type,source_id,content,metadata,knowledge_version,updated_at)
                   VALUES ('jiuzhaigou_scenic_area',:document_id,:source_type,:source_id,:content,CAST(:metadata AS jsonb),:version,:updated_at)
                   ON CONFLICT (park_id,document_id) DO UPDATE SET source_type=EXCLUDED.source_type,source_id=EXCLUDED.source_id,
                     content=EXCLUDED.content,metadata=EXCLUDED.metadata,knowledge_version=EXCLUDED.knowledge_version,updated_at=EXCLUDED.updated_at"""
            ),
            {
                "document_id": document_id,
                "source_type": source_type,
                "source_id": source_id,
                "content": content,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "version": version,
                "updated_at": now,
            },
        )
        conn.execute(
            text("DELETE FROM document_chunks WHERE park_id='jiuzhaigou_scenic_area' AND document_id=:document_id"),
            {"document_id": document_id},
        )
        conn.execute(
            text(
                """INSERT INTO document_chunks(park_id,document_id,chunk_index,content,updated_at)
                   VALUES ('jiuzhaigou_scenic_area',:document_id,:chunk_index,:content,:updated_at)"""
            ),
            [
                {"document_id": document_id, "chunk_index": chunk.index, "content": chunk.content, "updated_at": now}
                for chunk in chunks
            ],
        )
        conn.execute(
            text(
                "INSERT INTO async_tasks(task_id,park_id,task_type,status,payload) VALUES (:task_id,'jiuzhaigou_scenic_area','embed_document','pending',CAST(:payload AS jsonb))"
            ),
            {"task_id": task_id, "payload": json.dumps({"document_id": document_id})},
        )

    queued = False
    if enqueue:
        try:
            # The script commonly runs on the host while Compose exposes Redis on
            # 6380; set this before importing the app config used by the enqueue helper.
            os.environ["REDIS_URL"] = redis_url
            from app.core.events import enqueue_embedding_task

            queued = bool(asyncio.run(enqueue_embedding_task(task_id, document_id)))
        except Exception:
            queued = False
    return {"document_id": document_id, "task_id": task_id, "chunks": len(chunks), "queued": queued}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--document-id", default="doc_jiuzhaigou_detailed_guide")
    parser.add_argument("--source-id", default="jiuzhaigou_detailed_guide")
    parser.add_argument("--source-type", default="park")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0"))
    parser.add_argument("--no-enqueue", action="store_true")
    args = parser.parse_args()
    print(json.dumps(ingest(
        dsn=args.dsn,
        input_path=args.input,
        document_id=args.document_id,
        source_id=args.source_id,
        source_type=args.source_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enqueue=not args.no_enqueue,
        redis_url=args.redis_url,
    ), ensure_ascii=False))
