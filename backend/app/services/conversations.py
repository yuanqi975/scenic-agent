"""Conversation persistence.

Extracted from ``main.py`` so the orchestrator can own the request flow. Reads are
park-scoped and tolerate a missing database, matching the original behaviour.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import text

from ..core.config import PARK_ID
from ..core.db import enabled, engine
from ..core.db_write import CONVERSATION_QUEUE, drain, enqueue


def create_conversation(conversation_id: str | None) -> str:
    """Reuse an existing conversation id or open a new one."""
    cid = conversation_id or uuid.uuid4().hex
    if not enabled():
        return cid

    def write() -> None:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO conversations(conversation_id,park_id) VALUES (:id,:park_id) ON CONFLICT (conversation_id) DO NOTHING"
                    ),
                    {"id": cid, "park_id": PARK_ID},
                )
        except Exception:
            pass

    # Clients commonly generate an id before sending the first message.  That id
    # still needs its parent row; otherwise messages may be written without a
    # retrievable conversation and GET /conversations/{id} returns 404.
    enqueue(CONVERSATION_QUEUE, write, key=f"conversation_create:{cid}")
    return cid


def recent_history(conversation_id: str, limit: int = 10) -> list[dict[str, str]]:
    """Read the last few turns for pronoun resolution (「它几点关门」)."""
    if not enabled():
        return []
    # Conversation writes are intentionally queued to keep the response path fast.
    # Flush this queue before reading so a rapid follow-up request cannot race the
    # previous turn and receive an empty context.
    drain(CONVERSATION_QUEUE)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """SELECT role, content FROM conversation_messages
                       WHERE conversation_id=:cid ORDER BY id DESC LIMIT :limit"""
                ),
                {"cid": conversation_id, "limit": limit},
            ).fetchall()
    except Exception:
        return []
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


def append_messages(
    conversation_id: str,
    message: str,
    answer: str,
    citations: list[dict[str, Any]],
    *,
    agent_name: str | None = None,
) -> None:
    """Persist the visitor turn and the assistant turn.

    ``agent_name`` is written onto the assistant row so a conversation can be replayed
    agent-by-agent (the project's guide suggested this column).

    Queued rather than written inline: this is the last thing a request does, and a
    synchronous connect here would add a full connect timeout to the visitor's
    latency whenever the database is unreachable.
    """
    payload = json.dumps(citations, ensure_ascii=False, default=str)
    if not enabled():
        return

    def write() -> None:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """INSERT INTO conversation_messages(conversation_id,role,content,citations)
                           VALUES (:cid,'user',:message,CAST(:citations AS jsonb)),
                                 (:cid,'assistant',:answer,CAST(:citations AS jsonb))"""
                    ),
                    {"cid": conversation_id, "message": message, "answer": answer, "citations": payload},
                )
                if agent_name:
                    try:
                        conn.execute(
                            text(
                                """UPDATE conversation_messages SET agent_name=:agent
                                   WHERE id = (SELECT max(id) FROM conversation_messages WHERE conversation_id=:cid)"""
                            ),
                            {"agent": agent_name, "cid": conversation_id},
                        )
                    except Exception:
                        # The column is optional; ignore it when the schema predates it.
                        pass
                conn.execute(
                    text("UPDATE conversations SET updated_at=now() WHERE conversation_id=:cid"),
                    {"cid": conversation_id},
                )
        except Exception:
            pass

    enqueue(CONVERSATION_QUEUE, write, key="conversation_messages")


def conversation_history(conversation_id: str) -> dict[str, Any] | None:
    """Full transcript for the public history endpoint."""
    if not enabled():
        return None
    # Message persistence is deliberately queued so chat responses are not delayed by
    # PostgreSQL. A history read, however, has a read-after-write contract: flush only
    # the conversation queue before querying so an immediate GET sees the turn that
    # produced its conversation id.
    drain(CONVERSATION_QUEUE)
    try:
        with engine.connect() as conn:
            conversation = conn.execute(
                text(
                    "SELECT conversation_id,created_at,updated_at FROM conversations WHERE conversation_id=:id AND park_id=:park_id"
                ),
                {"id": conversation_id, "park_id": PARK_ID},
            ).first()
            if not conversation:
                return None
            rows = conn.execute(
                text(
                    "SELECT role,content,citations,created_at FROM conversation_messages WHERE conversation_id=:id ORDER BY id"
                ),
                {"id": conversation_id},
            ).fetchall()
    except Exception:
        return None
    return {
        "conversation_id": conversation[0],
        "created_at": conversation[1],
        "updated_at": conversation[2],
        "messages": [
            {"role": row[0], "content": row[1], "citations": row[2], "created_at": row[3]} for row in rows
        ],
    }
