"""Visitor feedback intake.

Two intake paths exist on purpose:

* ``POST /feedbacks`` keeps the original direct-write behaviour that admin tooling and
  the existing tests rely on;
* the ``feedback_agent`` route (tool ``create_feedback_candidate``) requires an explicit
  visitor confirmation before anything is written, which is the multi-agent behaviour.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text

from ..core.cache import enforce_public_rate_limit
from ..core.config import PARK_ID
from ..core.db import engine

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    content: str = Field(min_length=3, max_length=2000)
    feedback_type: str = "建议"
    related_attraction_id: str | None = None
    related_facility_id: str | None = None


@router.post("/feedbacks")
def feedback(request: FeedbackRequest, _: None = Depends(enforce_public_rate_limit)):
    """Record a visitor feedback together with its pending knowledge candidate."""
    feedback_id = f"feedback_runtime_{uuid.uuid4().hex[:12]}"
    submitted_at = datetime.now(timezone.utc).isoformat()
    payload = request.model_dump() | {
        "feedback_id": feedback_id,
        "status": "pending_review",
        "submitted_at": submitted_at,
        "data_source": "visitor",
        "is_simulated": False,
    }
    candidate_id = f"candidate_runtime_{uuid.uuid4().hex[:12]}"
    candidate = {
        "candidate_id": candidate_id,
        "feedback_id": feedback_id,
        "candidate_fact": request.content,
        "related_attraction_id": request.related_attraction_id,
        "related_facility_id": request.related_facility_id,
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
    return {"feedback_id": feedback_id, "candidate_id": candidate_id, "status": "pending_review"}
