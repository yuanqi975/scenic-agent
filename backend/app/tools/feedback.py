"""Feedback tools.

``create_feedback_candidate`` is the **only** write tool in the system. Two rules make
it safe:

* it is whitelisted for ``feedback_agent`` alone;
* it refuses to write unless the visitor explicitly asked to submit, and it keys the
  resulting record on a hash of (conversation, content) so a repeated confirmation
  cannot create a duplicate candidate.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from ..core.config import PARK_ID
from ..core.db import enabled, engine
from .registry import tool

#: How long an unconfirmed draft stays resumable.
INTENT_TTL_MINUTES = 30


class CreateFeedbackCandidateArgs(BaseModel):
    content: str = Field(min_length=3, max_length=2000, description="游客反馈的原始内容，保留地点与问题描述")
    confirmed: bool = Field(
        default=False,
        description="游客是否已明确表示要提交该反馈。false 时本工具只登记草稿并返回追问话术，绝不写入数据库。",
    )
    confirmation_text: str | None = Field(
        default=None, max_length=200, description="游客确认提交的那句话原文，作为审计依据"
    )
    feedback_type: str | None = Field(default=None, max_length=20, description="反馈类型：评价 / 建议 / 设施问题 / 路线问题 / 投诉")
    related_attraction_id: str | None = Field(default=None, max_length=64, description="反馈涉及的景点 ID")
    related_facility_id: str | None = Field(default=None, max_length=64, description="反馈涉及的设施 ID")
    conversation_id: str | None = Field(default=None, max_length=64, description="当前会话 ID")


class PendingFeedbackArgs(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64, description="要查看的会话 ID")


def _idempotency_key(conversation_id: str | None, content: str) -> str:
    raw = f"{PARK_ID}:{conversation_id or 'anonymous'}:{content.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _stored_draft(conversation_id: str | None) -> dict[str, Any] | None:
    if not conversation_id or not enabled():
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """SELECT intent_id, draft, status FROM pending_feedback_intents
                       WHERE park_id=:park_id AND conversation_id=:cid
                         AND status='awaiting_confirmation'
                       ORDER BY created_at DESC LIMIT 1"""
                ),
                {"park_id": PARK_ID, "cid": conversation_id},
            ).first()
    except Exception:
        return None
    if not row:
        return None
    draft = row[1] if isinstance(row[1], dict) else json.loads(row[1])
    return {"intent_id": row[0], "draft": draft, "status": row[2]}


def _store_draft(
    conversation_id: str | None,
    content: str,
    *,
    feedback_type: str | None,
    related_attraction_id: str | None,
    related_facility_id: str | None,
) -> str:
    intent_id = f"intent_{uuid.uuid4().hex[:16]}"
    draft = {
        "content": content,
        "feedback_type": feedback_type or "建议",
        "related_attraction_id": related_attraction_id,
        "related_facility_id": related_facility_id,
    }
    if not conversation_id or not enabled():
        return intent_id
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO pending_feedback_intents
                       (intent_id, park_id, conversation_id, draft, idempotency_key, status, expires_at)
                       VALUES (:intent_id,:park_id,:cid,CAST(:draft AS jsonb),:key,'awaiting_confirmation',:expires)"""
                ),
                {
                    "intent_id": intent_id,
                    "park_id": PARK_ID,
                    "cid": conversation_id,
                    "draft": json.dumps(draft, ensure_ascii=False),
                    "key": _idempotency_key(conversation_id, content),
                    "expires": datetime.now(timezone.utc) + timedelta(minutes=INTENT_TTL_MINUTES),
                },
            )
    except Exception:
        # Drafting is best effort; the write path still enforces confirmation.
        pass
    return intent_id


@tool(
    name="create_feedback_candidate",
    description=(
        "登记游客反馈。confirmed=false 时只保存草稿并返回需要向游客确认的问句，不写入任何反馈记录；"
        "只有游客明确表示提交（confirmed=true）才会创建 feedbacks 与 feedback_candidates 待审核候选。"
        "本工具不能发布知识，发布必须由管理员审核。"
    ),
    args_schema=CreateFeedbackCandidateArgs,
    allowed=("feedback_agent",),
    write=True,
    tags=("write", "feedback"),
)
def create_feedback_candidate(args: CreateFeedbackCandidateArgs) -> dict[str, Any]:
    if not args.confirmed:
        intent_id = _store_draft(
            args.conversation_id,
            args.content,
            feedback_type=args.feedback_type,
            related_attraction_id=args.related_attraction_id,
            related_facility_id=args.related_facility_id,
        )
        return {
            "status": "awaiting_confirmation",
            "intent_id": intent_id,
            "written": False,
            "ask_visitor": (
                "需要我帮你把这条反馈提交给景区吗？回复「提交」我就登记为待审核反馈。"
            ),
        }

    key = _idempotency_key(args.conversation_id, args.content)
    if args.conversation_id:
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        """SELECT payload->>'feedback_id' FROM feedbacks
                           WHERE park_id=:park_id AND payload->>'idempotency_key'=:key LIMIT 1"""
                    ),
                    {"park_id": PARK_ID, "key": key},
                ).first()
        except Exception:
            row = None
        if row:
            return {
                "status": "duplicate",
                "written": False,
                "feedback_id": row[0],
                "message": "这条反馈之前已经提交过，未重复登记。",
            }

    feedback_id = f"feedback_runtime_{uuid.uuid4().hex[:12]}"
    candidate_id = f"candidate_runtime_{uuid.uuid4().hex[:12]}"
    submitted_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "feedback_id": feedback_id,
        "content": args.content,
        "feedback_type": args.feedback_type or "建议",
        "related_attraction_id": args.related_attraction_id,
        "related_facility_id": args.related_facility_id,
        "status": "pending_review",
        "submitted_at": submitted_at,
        "data_source": "visitor",
        "is_simulated": False,
        "idempotency_key": key,
        "confirmation_text": args.confirmation_text,
        "conversation_id": args.conversation_id,
    }
    candidate = {
        "candidate_id": candidate_id,
        "feedback_id": feedback_id,
        "candidate_fact": args.content,
        "related_attraction_id": args.related_attraction_id,
        "related_facility_id": args.related_facility_id,
        "status": "pending_review",
        "data_source": "visitor",
        "is_simulated": False,
        "park_id": PARK_ID,
        "created_at": submitted_at,
        "source_feedback_count": 1,
    }
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO feedbacks(park_id,feedback_id,payload) VALUES (:park_id,:id,CAST(:payload AS jsonb))"),
            {"park_id": PARK_ID, "id": feedback_id, "payload": json.dumps(payload, ensure_ascii=False)},
        )
        conn.execute(
            text(
                "INSERT INTO feedback_candidates(park_id,candidate_id,payload) VALUES (:park_id,:id,CAST(:payload AS jsonb))"
            ),
            {"park_id": PARK_ID, "id": candidate_id, "payload": json.dumps(candidate, ensure_ascii=False)},
        )
        if args.conversation_id:
            conn.execute(
                text(
                    """UPDATE pending_feedback_intents SET status='consumed'
                       WHERE park_id=:park_id AND conversation_id=:cid AND status='awaiting_confirmation'"""
                ),
                {"park_id": PARK_ID, "cid": args.conversation_id},
            )
    return {
        "status": "pending_review",
        "written": True,
        "feedback_id": feedback_id,
        "candidate_id": candidate_id,
        "message": "反馈已登记为待审核候选，管理员审核通过后才会进入知识库。",
    }


@tool(
    name="check_pending_feedback",
    description="查看当前会话是否已有待确认的反馈草稿，避免重复追问。",
    args_schema=PendingFeedbackArgs,
    allowed=("feedback_agent",),
    tags=("read", "feedback"),
)
def check_pending_feedback(args: PendingFeedbackArgs) -> dict[str, Any]:
    stored = _stored_draft(args.conversation_id)
    return {
        "pending": bool(stored),
        "intent_id": (stored or {}).get("intent_id"),
        "draft": (stored or {}).get("draft"),
    }
