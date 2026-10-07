"""Background task helpers.

The embedding queue stays exactly as before: PostgreSQL holds the task row, Redis only
carries the notification, so a Redis outage cannot lose an approved publication.
"""

from __future__ import annotations

from ..core.config import REDIS_URL


async def enqueue_embedding_task(task_id: str, document_id: str) -> bool:
    """Best-effort enqueue; the ``async_tasks`` row remains the source of truth."""
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        await pool.enqueue_job("embed_document", task_id, _job_id=task_id)
        await pool.close()
        return True
    except Exception:
        return False
