"""Durable Embedding worker.

Run with ``arq app.worker.WorkerSettings`` after Redis and embedding settings
are configured.  Tasks live in PostgreSQL first, so a Redis outage never loses
an approved knowledge publication.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
from sqlalchemy import text

from .main import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_DIMENSION, EMBEDDING_MODEL, PARK_ID, REDIS_URL, engine
from .core import config


EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))


def _sync_milvus(
    *,
    document_id: str,
    content: str,
    vector: list[float],
    source_type: str,
    source_id: str,
    metadata: dict,
    chunk_index: int = 0,
) -> dict:
    """Upsert the freshly embedded chunk into the configured Milvus index.

    PostgreSQL remains the source of truth. A Milvus outage is recorded in the task
    payload instead of losing the approved feedback or inventing a vector; the next
    reindex task can safely retry the same deterministic primary key.
    """
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=config.MILVUS_URI,
            token=config.MILVUS_TOKEN or None,
            timeout=config.MILVUS_TIMEOUT_SECONDS,
        )
        if not client.has_collection(config.MILVUS_COLLECTION):
            return {"status": "unavailable", "reason": "collection_missing", "collection": config.MILVUS_COLLECTION}
        client.upsert(
            collection_name=config.MILVUS_COLLECTION,
            data=[
                {
                    "chunk_id": f"{document_id}:{chunk_index}",
                    "vector": vector,
                    "content": content,
                    "document_id": document_id,
                    "source_type": source_type or "feedback_review",
                    "source_id": source_id or document_id,
                    "authority": metadata.get("authority", "community"),
                    "park_id": PARK_ID,
                    "metadata": metadata,
                }
            ],
        )
        return {"status": "synced", "collection": config.MILVUS_COLLECTION, "chunk_id": f"{document_id}:{chunk_index}"}
    except Exception as exc:  # Milvus is an optional/rebuildable index.
        return {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def _set_status(task_id: str, status: str, extra: dict | None = None) -> None:
    with engine.begin() as conn:
        if extra is None:
            conn.execute(text("UPDATE async_tasks SET status=:status,updated_at=now() WHERE task_id=:task_id AND park_id=:park_id"), {"task_id": task_id, "park_id": PARK_ID, "status": status})
        else:
            conn.execute(text("UPDATE async_tasks SET status=:status,last_error=:error,payload=payload || CAST(:extra AS jsonb),updated_at=now() WHERE task_id=:task_id AND park_id=:park_id"), {"task_id": task_id, "park_id": PARK_ID, "status": status, "error": extra.get("error"), "extra": json.dumps(extra)})


def process_embedding_task(task_id: str) -> str:
    """Embed one pending document with a real provider, never a synthetic vector."""
    with engine.connect() as conn:
        task = conn.execute(text("SELECT payload,status,attempt_count,max_attempts FROM async_tasks WHERE task_id=:task_id AND park_id=:park_id AND task_type='embed_document'"), {"task_id": task_id, "park_id": PARK_ID}).first()
    if not task:
        return "not_found"
    if not (EMBEDDING_API_KEY and EMBEDDING_BASE_URL):
        _set_status(task_id, "waiting_for_configuration")
        return "waiting_for_configuration"
    payload = task[0] if isinstance(task[0], dict) else json.loads(task[0])
    attempt_count, max_attempts = int(task[2] or 0), int(task[3] or 5)
    if attempt_count >= max_attempts:
        _set_status(task_id, "dead_letter", {"error": "maximum retry attempts exceeded"})
        return "dead_letter"
    with engine.begin() as conn:
        conn.execute(text("UPDATE async_tasks SET attempt_count=attempt_count+1,status='running',updated_at=now() WHERE task_id=:task_id AND park_id=:park_id"), {"task_id": task_id, "park_id": PARK_ID})
    document_id = payload.get("document_id")
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """SELECT c.content, d.source_type, d.source_id, d.metadata, c.chunk_index
                   FROM document_chunks c
                   LEFT JOIN documents d ON d.park_id=c.park_id AND d.document_id=c.document_id
                   WHERE c.park_id=:park_id AND c.document_id=:document_id
                   ORDER BY c.chunk_index"""
            ),
            {"park_id": PARK_ID, "document_id": document_id},
        ).all()
    if not rows:
        _set_status(task_id, "failed", {"error": "document chunk not found"})
        return "failed"
    try:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=45) as client:
            for offset in range(0, len(rows), EMBEDDING_BATCH_SIZE):
                batch = rows[offset : offset + EMBEDDING_BATCH_SIZE]
                response = client.post(
                    f"{EMBEDDING_BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
                    json={"model": EMBEDDING_MODEL, "input": [row[0] for row in batch], "dimensions": EMBEDDING_DIMENSION},
                )
                response.raise_for_status()
                batch_vectors = [item["embedding"] for item in response.json()["data"]]
                if len(batch_vectors) != len(batch) or any(
                    not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION for vector in batch_vectors
                ):
                    raise ValueError(
                        f"embedding response must contain {len(batch)} vectors of dimension {EMBEDDING_DIMENSION}"
                    )
                vectors.extend(batch_vectors)
        with engine.begin() as conn:
            for row, vector in zip(rows, vectors):
                conn.execute(
                    text("UPDATE document_chunks SET embedding=CAST(:vector AS halfvec),updated_at=now() WHERE park_id=:park_id AND document_id=:document_id AND chunk_index=:chunk_index"),
                    {"vector": str(vector), "park_id": PARK_ID, "document_id": document_id, "chunk_index": row[4]},
                )
        metadata = rows[0][3] if isinstance(rows[0][3], dict) else (json.loads(rows[0][3]) if rows[0][3] else {})
        metadata = dict(metadata or {})
        metadata.setdefault("authority", "community" if str(rows[0][1] or "").startswith("feedback") else "official")
        milvus_results = []
        for row, vector in zip(rows, vectors):
            milvus_results.append(_sync_milvus(
                document_id=document_id,
                content=row[0],
                vector=vector,
                source_type=str(row[1] or "feedback_review"),
                source_id=str(row[2] or document_id),
                metadata={**metadata, "chunk_index": row[4]},
                chunk_index=row[4],
            ))
        _set_status(
            task_id,
            "completed",
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": len(rows),
                "milvus": milvus_results,
            },
        )
        return "completed"
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        next_status = "dead_letter" if attempt_count + 1 >= max_attempts else "retry"
        _set_status(task_id, next_status, {"error": str(exc)[:500], "attempt_count": attempt_count + 1})
        return next_status


async def embed_document(_: dict, task_id: str) -> str:
    """arq entry point; blocking database/API work is isolated in a thread."""
    import asyncio
    return await asyncio.to_thread(process_embedding_task, task_id)


async def on_startup(ctx: dict) -> None:
    """Requeue durable tasks left behind by a worker restart.

    PostgreSQL is the source of truth for embedding tasks, while Redis only carries
    notifications. Without this recovery pass, a task created while the worker was
    stopped remains ``pending`` forever even though the admin UI reports it queued.
    """
    if not (EMBEDDING_API_KEY and EMBEDDING_BASE_URL):
        return
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """SELECT task_id FROM async_tasks
                       WHERE park_id=:park_id AND task_type='embed_document'
                         AND status IN ('pending','retry','waiting_for_configuration')
                       ORDER BY created_at"""
                ),
                {"park_id": PARK_ID},
            ).all()
        redis = ctx.get("redis")
        if redis is None:
            return
        for (task_id,) in rows:
            try:
                await redis.enqueue_job("embed_document", task_id, _job_id=task_id)
            except Exception:
                # Duplicate jobs are harmless; the PostgreSQL task remains durable.
                continue
    except Exception:
        return


class WorkerSettings:
    functions = [embed_document]
    on_startup = on_startup
    from arq.connections import RedisSettings
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
