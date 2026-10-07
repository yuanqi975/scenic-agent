"""Conversation memory tool (read-only)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from ..core.config import PARK_ID
from ..core.db import engine
from .registry import tool


class GetConversationHistoryArgs(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64, description="当前会话 ID")
    limit: int = Field(default=10, ge=1, le=20, description="读取最近多少条消息（默认覆盖最近 5 轮），用于理解指代")


@tool(
    name="get_conversation_history",
    description="读取当前会话的最近若干轮对话，用于理解游客话里的指代与追问。",
    args_schema=GetConversationHistoryArgs,
    allowed=("supervisor", "knowledge_agent", "feedback_agent", "response_agent"),
    tags=("read", "memory"),
)
def get_conversation_history(args: GetConversationHistoryArgs) -> dict[str, Any]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """SELECT role, content, created_at FROM conversation_messages
                       WHERE conversation_id=:cid ORDER BY id DESC LIMIT :limit"""
                ),
                {"cid": args.conversation_id, "limit": args.limit},
            ).fetchall()
    except Exception:
        return {"conversation_id": args.conversation_id, "available": False, "messages": []}
    messages = [{"role": row[0], "content": row[1], "created_at": str(row[2])} for row in reversed(rows)]
    return {"conversation_id": args.conversation_id, "available": bool(messages), "messages": messages}


def recent_history(conversation_id: str, limit: int = 10) -> list[dict[str, str]]:
    """Plain ``[{role, content}]`` history used when prompting the model."""
    payload = get_conversation_history(
        GetConversationHistoryArgs(conversation_id=conversation_id, limit=limit)
    )
    return [{"role": item["role"], "content": item["content"]} for item in payload.get("messages", [])]


def park_id() -> str:
    return PARK_ID
