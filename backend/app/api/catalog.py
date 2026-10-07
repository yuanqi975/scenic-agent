"""Public catalog endpoints: attractions and the standalone route recommendation."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.cache import enforce_public_rate_limit
from ..core.config import PARK_ID
from ..core.db import engine, payload_rows
from ..services.orchestrator import build_recommendation
from sqlalchemy import text

router = APIRouter(tags=["catalog"])


class RecommendationRequest(BaseModel):
    duration_minutes: int = Field(default=240, ge=30, le=1440)
    groups: list[str] = []
    preferences: list[str] = []
    weather: str = "晴"
    difficulty: str = "轻松"
    required_facilities: list[str] = []
    exclude_attraction_ids: list[str] = []


@router.get("/attractions")
def attractions(
    q: str | None = None,
    category: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    items = payload_rows("attractions", 100, 0)
    if q:
        items = [item for item in items if q.lower() in json.dumps(item, ensure_ascii=False).lower()]
    if category:
        items = [item for item in items if item.get("category") == category]
    return {"items": items[offset : offset + limit], "total": len(items)}


@router.get("/attractions/{attraction_id}")
def attraction_detail(attraction_id: str):
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT payload FROM attractions WHERE park_id=:park_id AND attraction_id=:id"),
                {"park_id": PARK_ID, "id": attraction_id},
            ).first()
    except Exception:
        row = None
    if not row:
        raise HTTPException(404, "景点不存在")
    return row[0]


@router.get("/notices")
def notices(
    active_only: bool = Query(True),
    notice_type: str | None = Query(None, max_length=40),
    limit: int = Query(20, ge=1, le=100),
):
    """Expose dated official notices for public clients and operations dashboards."""
    items = payload_rows("notices", 100, 0)
    if active_only:
        items = [item for item in items if item.get("status") == "active"]
    if notice_type:
        items = [item for item in items if item.get("notice_type") == notice_type]
    return {"items": items[:limit], "total": len(items), "knowledge_version": "v2.0"}


@router.post("/recommendations")
def recommendations(request: RecommendationRequest, _: None = Depends(enforce_public_rate_limit)):
    """Deterministic route recommendation.

    Kept as a standalone endpoint for the existing UI; the same algorithm is exposed to
    agents as the ``calculate_route`` tool, so both paths always agree.
    """
    result = build_recommendation(
        duration_minutes=request.duration_minutes,
        groups=request.groups,
        preferences=request.preferences,
        weather=request.weather,
        difficulty=request.difficulty,
    )
    if request.required_facilities:
        from ..tools.route import ValidateRouteArgs, calculate_route, validate_route, CalculateRouteArgs

        plan = calculate_route(
            CalculateRouteArgs(
                query=None,
                duration_minutes=request.duration_minutes,
                groups=request.groups,
                preferences=request.preferences,
                weather=request.weather,
                required_facilities=request.required_facilities,
                difficulty=request.difficulty,
                exclude_attraction_ids=request.exclude_attraction_ids,
            )
        )
        result["attractions"] = plan["attractions"]
        result["total_minutes"] = plan["total_minutes"]
        result["validation"] = validate_route(
            ValidateRouteArgs(
                plan=plan,
                duration_minutes=request.duration_minutes,
                required_facilities=request.required_facilities,
                groups=request.groups,
            )
        )
    return result
