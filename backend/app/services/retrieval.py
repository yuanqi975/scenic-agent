"""RAG retrieval over the Jiuzhaigou knowledge base.

Four layers, cheapest-and-most-precise first:

1. **Named entity** - a known POI name literally appears in the question. Chinese has
   no word delimiters, so PostgreSQL ``simple`` FTS cannot match「五花海怎么去？」;
   matching published names ourselves is the only reliable anchor.
2. **pgvector** cosine search over ``document_chunks`` (real vectors only).
3. **PostgreSQL FTS** over ``documents``.
4. **ILIKE** fallback, which is what actually works for unsegmented Chinese.

Every layer feeds one deduplicating, diversity-aware selector so ``top_k`` reflects
distinct knowledge rather than five copies of the same paragraph.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from sqlalchemy import text

from ..core.cache import cache_get, cache_set
from ..core.config import (
    CACHE_TTL_SECONDS,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PARK_ID,
    RETRIEVAL_MAX_DISTANCE,
    RETRIEVAL_MAX_PER_SOURCE,
    RETRIEVAL_TOP_K,
)
from ..core import db as db_core
from ..core.db import engine, payload_rows

#: Retrieval layer -> authority label used by the response agent to separate
#: official facts from live data and from unsupported inference.
SOURCE_AUTHORITY = {
    "faq": "official",
    "park": "official",
    "attraction": "official",
    "facility": "official",
    "route": "official",
    "feedback_review": "community",
    "notice": "official",
}


def _authority(source_type: str | None) -> str:
    return SOURCE_AUTHORITY.get(source_type or "", "reference")


def embed_query(message: str, *, client: Any | None = None) -> str | None:
    """Return a real query vector, or ``None`` when no embedding provider is set."""
    if not (EMBEDDING_BASE_URL and EMBEDDING_API_KEY):
        return None
    try:
        import httpx

        request = client.post if client is not None else httpx.post
        response = request(
            f"{EMBEDDING_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
            json={"model": EMBEDDING_MODEL, "input": message, "dimensions": EMBEDDING_DIMENSION},
            timeout=12,
        )
        response.raise_for_status()
        return str(response.json()["data"][0]["embedding"])
    except Exception:
        return None


def known_entity_ids(
    message: str,
    *,
    rows: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[str]:
    """POI ids whose published name matches the question in either direction.

    ``rows`` should be passed explicitly by callers that already hold a catalog reader.
    Falling back to the module global keeps the function usable standalone, but note
    that a default *argument* would be worse than a fallback: it captures the function
    object at import time, so a caller that swapped the reader afterwards was silently
    ignored and every lookup went to the database (~12 s of connect timeouts per
    question when the database is unreachable).
    """
    reader = rows or payload_rows
    found: list[str] = []
    for table, id_key in (("attractions", "attraction_id"), ("facilities", "facility_id")):
        try:
            items = reader(table, 100)
        except Exception:
            items = []
        for item in items:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            identifier = item.get(id_key)
            if identifier and (name in message or message.strip() in name):
                found.append(identifier)
    return list(dict.fromkeys(found))


def _row_to_item(row: Any, method: str, distance: float | None = None) -> dict[str, Any]:
    if isinstance(row, dict):
        document_id, source_type, source_id, content, metadata = (
            row.get("document_id"),
            row.get("source_type"),
            row.get("source_id"),
            row.get("content"),
            row.get("metadata"),
        )
    else:
        document_id, source_type, source_id, content, metadata = row[0], row[1], row[2], row[3], row[4]
    return {
        "document_id": document_id,
        "source_type": source_type,
        "source_id": source_id,
        "content": content,
        "metadata": metadata or {},
        "retrieval": method,
        "distance": distance,
        "authority": _authority(source_type),
        "title": (metadata or {}).get("title") if isinstance(metadata, dict) else None,
    }


def select_diverse(
    items: list[dict[str, Any]],
    *,
    top_k: int = RETRIEVAL_TOP_K,
    max_per_source: int = RETRIEVAL_MAX_PER_SOURCE,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Drop near-duplicate vectors and cap how many hits one source entity may take.

    The knowledge base repeats the same fact across many document ids, so a raw
    ``ORDER BY distance LIMIT 5`` commonly returns five paraphrases of one sentence.
    """
    selected: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    seen_content: set[str] = set()
    overflow: list[dict[str, Any]] = []

    for item in items:
        if max_distance is not None and item.get("distance") is not None and item["distance"] > max_distance:
            continue
        fingerprint = hashlib.sha1(
            " ".join((item.get("content") or "").split()).encode("utf-8", "ignore")
        ).hexdigest()
        if fingerprint in seen_content:
            continue
        source_key = str(item.get("source_id") or item.get("document_id"))
        if per_source.get(source_key, 0) >= max_per_source:
            overflow.append(item)
            continue
        seen_content.add(fingerprint)
        per_source[source_key] = per_source.get(source_key, 0) + 1
        selected.append(item)
        if len(selected) >= top_k:
            return selected

    # Backfill with capped-out hits only if there is nothing else to show, so a
    # single-entity question still returns evidence instead of an empty list.
    for item in overflow:
        if len(selected) >= top_k:
            break
        selected.append(item)
    return selected


def retrieve(
    message: str,
    top_k: int = RETRIEVAL_TOP_K,
    *,
    rows: Callable[..., list[dict[str, Any]]] | None = None,
    connection: Any | None = None,
    vector: str | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Run the four-layer funnel and return de-duplicated evidence.

    ``rows`` is resolved inside the body rather than as a default argument. A default
    binds the *function object* at import time, so replacing the catalog reader later
    (an in-memory catalog in tests, a different park in production) had no effect and
    every call fell through to the database.
    """
    catalog = rows or payload_rows
    cache_key = f"retrieval:v2:{PARK_ID}:{top_k}:{hashlib.sha256(message.strip().lower().encode()).hexdigest()}"
    if use_cache:
        cached = cache_get(cache_key)
        if cached:
            return cached["items"]

    candidates: list[dict[str, Any]] = []
    entity_ids = known_entity_ids(message, rows=catalog)

    def run(sql: str, params: dict[str, Any]) -> list[Any]:
        if connection is not None:
            return connection.execute(text(sql), params).fetchall()
        # The engine has a short TCP circuit breaker; do not open a new SQLAlchemy
        # connection for every channel when staging dependencies are unavailable.
        if rows is None and not db_core.database_available():
            return []
        try:
            with engine.connect() as conn:
                return conn.execute(text(sql), params).fetchall()
        except Exception:
            return []

    # Layer 1 -- published POI names mentioned in the question.
    for entity_id in entity_ids[:top_k]:
        entity_rows = run(
            """SELECT document_id, source_type, source_id, content, metadata FROM documents
               WHERE park_id=:park_id AND source_id=:source_id ORDER BY document_id LIMIT :limit""",
            {"park_id": PARK_ID, "source_id": entity_id, "limit": top_k},
        )
        candidates.extend(_row_to_item(row, "entity") for row in entity_rows)

    # Layer 1b -- time-sensitive operating facts.  These are deliberately kept
    # separate from lexical recall: an announcement can be textually relevant but
    # expired, while an active notice should remain visible to the evidence validator.
    if any(token in message for token in ("现在", "今天", "明天", "开放", "预约", "售罄", "承载", "公告", "临时", "优惠", "泥石流", "暴雨")):
        notice_rows = run(
            """SELECT document_id, source_type, source_id, content, metadata FROM documents
               WHERE park_id=:park_id AND (source_type='notice' OR metadata->>'document_kind'='notice')
                 AND (content ILIKE :q OR content ILIKE :q2 OR content ILIKE :q3)
               ORDER BY (metadata->>'status' = 'active') DESC, updated_at DESC LIMIT :limit""",
            {"park_id": PARK_ID, "q": f"%{message}%", "q2": "%开放%", "q3": "%预约%", "limit": max(top_k, 5)},
        )
        candidates.extend(_row_to_item(row, "structured") for row in notice_rows)

    # Layer 2 -- real vectors only; never a synthetic embedding.
    vector = vector if vector is not None else embed_query(message)
    if vector:
        vector_rows = run(
            """SELECT d.document_id, d.source_type, d.source_id, c.content, d.metadata,
                      (c.embedding <=> CAST(:vector AS halfvec)) AS distance
               FROM document_chunks c
               JOIN documents d ON (d.park_id=c.park_id AND d.document_id=c.document_id)
               WHERE c.park_id=:park_id AND c.embedding IS NOT NULL
               ORDER BY distance LIMIT :limit""",
            {"park_id": PARK_ID, "vector": vector, "limit": top_k * 3},
        )
        for row in vector_rows:
            distance = row[5] if not isinstance(row, dict) else row.get("distance")
            candidates.append(_row_to_item(row, "vector", float(distance) if distance is not None else None))

    # Layer 3 -- PostgreSQL full text search.
    fts_rows = run(
        """SELECT document_id, source_type, source_id, content, metadata FROM documents
           WHERE park_id=:park_id AND to_tsvector('simple', content) @@ plainto_tsquery('simple', :message)
           ORDER BY updated_at DESC LIMIT :limit""",
        {"park_id": PARK_ID, "message": message, "limit": top_k},
    )
    candidates.extend(_row_to_item(row, "fts") for row in fts_rows)

    # Layer 4 -- Chinese-safe fuzzy fallback.
    if not fts_rows:
        first_token = message.split()[0] if message.split() else message
        like_rows = run(
            """SELECT document_id, source_type, source_id, content, metadata FROM documents
               WHERE park_id=:park_id AND (content ILIKE :q OR content ILIKE :q2)
               ORDER BY updated_at DESC LIMIT :limit""",
            {"park_id": PARK_ID, "q": f"%{message}%", "q2": f"%{first_token}%", "limit": top_k * 2},
        )
        candidates.extend(_row_to_item(row, "like") for row in like_rows)

    # Vector hits carry a distance; keep the threshold only where it is meaningful.
    has_vector = any(item["retrieval"] == "vector" for item in candidates)
    items = select_diverse(
        candidates,
        top_k=top_k,
        max_per_source=RETRIEVAL_MAX_PER_SOURCE,
        max_distance=RETRIEVAL_MAX_DISTANCE if has_vector else None,
    )
    if use_cache:
        cache_set(cache_key, {"items": items}, ttl=CACHE_TTL_SECONDS)
    return items


def citations_from(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert evidence into the citation shape the API and frontend already use."""
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("document_id") or item.get("source_id"))
        if key in seen:
            continue
        seen.add(key)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        citations.append(
            {
                "source_type": item.get("source_type"),
                "source_id": item.get("source_id"),
                "document_id": item.get("document_id"),
                "title": metadata.get("title") or item.get("title") or item.get("source_id"),
                "authority": item.get("authority", "reference"),
                "retrieval": item.get("retrieval"),
                "knowledge_version": metadata.get("knowledge_version") or metadata.get("version"),
            }
        )
    return citations
